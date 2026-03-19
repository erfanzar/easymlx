# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paged (unified) attention kernel for MLX.

This is a Metal implementation inspired by EasyDeL/ejkernel's unified attention
Triton kernels. The focus here is correctness and feature parity with the
block-tabled KV cache layout, not peak performance.
"""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from mlx import core as mx

from easymlx.caching import PageCacheView, PageMetadata

from .._attention_outputs import AttentionOutput
from .._base_operation import BaseOperation, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, RequirementsBuilder

_DEFAULT_USE_METAL = True
_FORCE_REFERENCE = False
_MAX_METAL_HEAD_DIM = 256


@dataclass(slots=True)
class UnifiedAttnMetadata:
    """Metadata required for paged (unified) attention execution.

    Attributes:
        block_tables: Integer array of shape ``[num_seqs, max_blocks_per_seq]``
            mapping sequences to KV cache block indices.
        kv_lens: Integer array of shape ``[num_seqs]`` with the actual KV
            length for each sequence.
        query_start_loc: Integer array of shape ``[num_seqs + 1]`` with
            cumulative query start positions.
        block_size: Number of KV slots per cache block.
        sliding_window: Optional sliding window size. None or 0 disables.
        is_single_token_decode: Whether each scheduled sequence contributes
            exactly one token in this step.
    """

    block_tables: mx.ArrayLike
    kv_lens: mx.ArrayLike
    query_start_loc: mx.ArrayLike
    block_size: int
    sliding_window: int | None = None
    is_single_token_decode: bool = False


@dataclass(slots=True)
class UnifiedAttnConfig:
    """Configuration for the unified attention kernel.

    Attributes:
        use_metal_kernel: Whether to prefer the Metal GPU kernel.
        allow_fallback: Whether to fall back to the NumPy reference
            implementation if the Metal kernel fails.
        threadgroup_size: Metal threadgroup size for the kernel dispatch.
    """

    use_metal_kernel: bool = True
    allow_fallback: bool = True
    threadgroup_size: int = 256


def _validate_inputs(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    kv_lens: mx.array,
    query_start_loc: mx.array,
) -> None:
    """Validate tensor shapes for paged attention inputs.

    Args:
        queries: Query tensor, expected rank 3.
        key_cache: Key cache tensor, expected rank 4.
        value_cache: Value cache tensor, expected rank 4.
        block_tables: Block table tensor, expected rank 2.
        kv_lens: KV lengths tensor, expected rank 1.
        query_start_loc: Query start locations, expected rank 1.

    Raises:
        ValueError: If any tensor has an unexpected shape.
    """
    if queries.ndim != 3:
        raise ValueError("queries must be rank-3: [total_tokens, num_q_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key_cache/value_cache must be rank-4: [num_blocks, block_size, num_kv_heads, head_dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if queries.shape[2] != key_cache.shape[3]:
        raise ValueError("queries head_dim must match key/value cache head_dim")
    if queries.shape[1] % key_cache.shape[2] != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be rank-2: [num_seqs, max_blocks_per_seq]")
    if kv_lens.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("kv_lens/query_start_loc must be rank-1")
    if query_start_loc.shape[0] != block_tables.shape[0] + 1:
        raise ValueError("query_start_loc must have length num_seqs + 1")


def _next_power_of_two(value: int) -> int:
    """Return the next power of two greater than or equal to *value*."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _floor_power_of_two(value: int) -> int:
    """Return the greatest power of two less than or equal to *value*."""
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


def _resolve_threadgroup_size(head_dim: int, requested: int) -> int:
    """Choose a reduction-friendly threadgroup size for the Metal kernel."""
    requested = max(32, min(int(requested), _MAX_METAL_HEAD_DIM))
    requested = _floor_power_of_two(requested)
    preferred = max(32, min(_MAX_METAL_HEAD_DIM, _next_power_of_two(head_dim)))
    return min(requested, preferred)


def _resolve_decode_threadgroup_size(head_dim: int, requested: int, total_tokens: int) -> int:
    """Choose a decode-friendly threadgroup size for the Metal kernel."""
    resolved = _resolve_threadgroup_size(head_dim, requested)
    if head_dim >= 128 and int(total_tokens) >= 4:
        return min(resolved, 128)
    return resolved


def _ensure_mx_array(value: mx.ArrayLike, *, dtype: mx.Dtype) -> mx.array:
    """Return *value* as an ``mx.array`` of the requested dtype.

    Avoids redundant casts in the hot path when callers already pass MLX
    arrays with the correct dtype.
    """
    if isinstance(value, mx.array) and value.dtype == dtype:
        return value
    return mx.array(value, dtype=dtype)


@lru_cache(maxsize=32)
def _float32_scalar_ptr(value: float) -> mx.array:
    """Return a cached 1-element float32 MLX array."""
    return mx.array([float(value)], dtype=mx.float32)


@lru_cache(maxsize=32)
def _int32_scalar_ptr(value: int) -> mx.array:
    """Return a cached 1-element int32 MLX array."""
    return mx.array([int(value)], dtype=mx.int32)


def _is_single_token_decode(query_start_loc: mx.ArrayLike, total_tokens: int) -> bool:
    """Return ``True`` when every scheduled sequence contributes exactly one token."""
    if total_tokens <= 0:
        return False
    qsl = np.asarray(query_start_loc, dtype=np.int32)
    if qsl.ndim != 1 or qsl.shape[0] != total_tokens + 1:
        return False
    if qsl[0] != 0:
        return False
    return bool(np.all(qsl[1:] - qsl[:-1] == 1))


def _paged_attention_reference(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    kv_lens: mx.array,
    query_start_loc: mx.array,
    *,
    softmax_scale: float,
    sliding_window: int,
) -> mx.array:
    """NumPy reference implementation of paged attention.

    This is a correct-but-slow fallback used when the Metal kernel is
    unavailable or fails. It loops over sequences and tokens explicitly.

    Args:
        queries: Query tensor of shape ``[total_tokens, num_q_heads, head_dim]``.
        key_cache: Key cache of shape ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Value cache with the same shape as ``key_cache``.
        block_tables: Block table of shape ``[num_seqs, max_blocks_per_seq]``.
        kv_lens: KV lengths of shape ``[num_seqs]``.
        query_start_loc: Query start locations of shape ``[num_seqs + 1]``.
        softmax_scale: Scaling factor applied to queries.
        sliding_window: Sliding window size (0 to disable).

    Returns:
        Output tensor of the same shape and dtype as ``queries``.

    Raises:
        ValueError: If ``num_q_heads`` is not divisible by ``num_kv_heads``.
    """
    q_np = np.asarray(queries)
    k_cache = np.asarray(key_cache)
    v_cache = np.asarray(value_cache)
    bt = np.asarray(block_tables)
    kv = np.asarray(kv_lens)
    qsl = np.asarray(query_start_loc)

    _total_tokens, num_q_heads, head_dim = q_np.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    num_seqs = bt.shape[0]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")

    num_queries_per_kv = num_q_heads // num_kv_heads
    out = np.zeros_like(q_np, dtype=np.float32)

    for seq_idx in range(num_seqs):
        q_start = int(qsl[seq_idx])
        q_end = int(qsl[seq_idx + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        seq_len = int(kv[seq_idx])
        if seq_len <= 0:
            continue

        k_seq = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        v_seq = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        for pos in range(seq_len):
            block = int(bt[seq_idx, pos // block_size])
            if block < 0 or block >= num_blocks:
                continue
            offset = pos % block_size
            k_seq[pos] = k_cache[block, offset]
            v_seq[pos] = v_cache[block, offset]

        for local_idx in range(q_len):
            token_idx = q_start + local_idx
            q_vec = q_np[token_idx].astype(np.float32) * softmax_scale
            context_len = seq_len - q_len
            max_k = min(seq_len - 1, context_len + local_idx)
            if max_k < 0:
                continue
            start_k = 0
            if sliding_window > 0:
                start_k = max(0, max_k - sliding_window + 1)

            k_slice = k_seq[start_k : max_k + 1]
            v_slice = v_seq[start_k : max_k + 1]
            if k_slice.size == 0:
                continue

            k_full = np.repeat(k_slice, repeats=num_queries_per_kv, axis=1)
            v_full = np.repeat(v_slice, repeats=num_queries_per_kv, axis=1)

            scores = np.einsum("hd,khd->hk", q_vec, k_full, optimize=True)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            out[token_idx] = np.einsum("hk,khd->hd", weights, v_full, optimize=True)

    return mx.array(out.astype(np.asarray(queries).dtype))


_METAL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""


_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint token_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;
    float q_scale = softmax_scale_ptr[0];

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    if ((int)token_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)token_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int num_seqs = query_start_loc_shape[0] - 1;
    int seq_idx = 0;
    for (int i = 0; i < num_seqs; ++i) {
        int start = query_start_loc[i];
        int end = query_start_loc[i + 1];
        if ((int)token_idx >= start && (int)token_idx < end) {
            seq_idx = i;
            break;
        }
    }

    int q_start = query_start_loc[seq_idx];
    int q_end = query_start_loc[seq_idx + 1];
    int q_len = q_end - q_start;
    if (q_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int q_pos = (int)token_idx - q_start;

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int context_len = seq_len - q_len;
    int max_k = context_len + q_pos;
    if (max_k >= seq_len) {
        max_k = seq_len - 1;
    }
    if (max_k < 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = max_k - sliding_window + 1;
        if (win_start > 0) {
            start_k = win_start;
        }
    }

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * q_scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;

    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride;
            float partial = 0.0f;

            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                partial += q_shared[d] * (float)key_cache[kv_base + d];
            }

            reduce_shared[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_shared[tid] += reduce_shared[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = reduce_shared[0];
                float m_new = fmax(m, score);
                float p = exp(score - m_new);
                float alpha = exp(m - m_new);
                alpha_shared = alpha;
                p_shared = p;
                l = l * alpha + p;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha = alpha_shared;
            float p = p_shared;
            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                out_shared[d] =
                    out_shared[d] * alpha + p * (float)value_cache[kv_base + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""


_METAL_DECODE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;
    float q_scale = softmax_scale_ptr[0];

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    if ((int)seq_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)seq_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = seq_len - sliding_window;
        if (win_start > 0) {
            start_k = win_start;
        }
    }
    int max_k = seq_len - 1;

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * q_scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride;
            float partial = 0.0f;

            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                partial += q_shared[d] * (float)key_cache[kv_base + d];
            }

            reduce_shared[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_shared[tid] += reduce_shared[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = reduce_shared[0];
                float m_new = fmax(m, score);
                float p = exp(score - m_new);
                float alpha = exp(m - m_new);
                alpha_shared = alpha;
                p_shared = p;
                l = l * alpha + p;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha = alpha_shared;
            float p = p_shared;
            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                out_shared[d] =
                    out_shared[d] * alpha + p * (float)value_cache[kv_base + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""


_METAL_DECODE_HEAD256_SOURCE = r"""
    constexpr uint kHeadDim = 256;
    constexpr uint kPack = 4;
    constexpr uint kThreads = kHeadDim / kPack;

    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup_id = simdgroup_index_in_threadgroup;
    float q_scale = softmax_scale_ptr[0];

    if (tid >= kThreads) {
        return;
    }

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (head_dim != (int)kHeadDim || total_tokens <= 0 || num_q_heads <= 0) {
        return;
    }
    if ((int)seq_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)seq_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = seq_len - sliding_window;
        if (win_start > 0) {
            start_k = win_start;
        }
    }
    int max_k = seq_len - 1;

    int q_vec_base = q_base + (int)tid * (int)kPack;
    float4 q_vec = float4(
        (float)queries[q_vec_base + 0] * q_scale,
        (float)queries[q_vec_base + 1] * q_scale,
        (float)queries[q_vec_base + 2] * q_scale,
        (float)queries[q_vec_base + 3] * q_scale
    );
    float4 out_vec = float4(0.0f);

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    threadgroup float simd_sums[2];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    float m = -INFINITY;
    float l = 0.0f;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride + (int)tid * (int)kPack;

            float4 k_vec = float4(
                (float)key_cache[kv_base + 0],
                (float)key_cache[kv_base + 1],
                (float)key_cache[kv_base + 2],
                (float)key_cache[kv_base + 3]
            );
            float partial = dot(q_vec, k_vec);

            float simd_total = simd_sum(partial);
            if (lane == 0) {
                simd_sums[simdgroup_id] = simd_total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup_id == 0) {
                float group_partial = tid < 2 ? simd_sums[tid] : 0.0f;
                float score = simd_sum(group_partial);
                if (tid == 0) {
                    float m_new = fmax(m, score);
                    float p = exp(score - m_new);
                    float alpha = exp(m - m_new);
                    alpha_shared = alpha;
                    p_shared = p;
                    l = l * alpha + p;
                    m = m_new;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 v_vec = float4(
                (float)value_cache[kv_base + 0],
                (float)value_cache[kv_base + 1],
                (float)value_cache[kv_base + 2],
                (float)value_cache[kv_base + 3]
            );
            out_vec = out_vec * alpha_shared + p_shared * v_vec;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 out_scaled = out_vec * inv_l_shared;
    int out_base = q_base + (int)tid * (int)kPack;
    out[out_base + 0] = (T)out_scaled.x;
    out[out_base + 1] = (T)out_scaled.y;
    out[out_base + 2] = (T)out_scaled.z;
    out[out_base + 3] = (T)out_scaled.w;
"""


_METAL_FP8_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint token_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;
    float q_scale = softmax_scale_ptr[0];

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    if ((int)token_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)token_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int num_seqs = query_start_loc_shape[0] - 1;
    int seq_idx = 0;
    for (int i = 0; i < num_seqs; ++i) {
        int start = query_start_loc[i];
        int end = query_start_loc[i + 1];
        if ((int)token_idx >= start && (int)token_idx < end) {
            seq_idx = i;
            break;
        }
    }

    int q_start = query_start_loc[seq_idx];
    int q_end = query_start_loc[seq_idx + 1];
    int q_len = q_end - q_start;
    if (q_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int q_pos = (int)token_idx - q_start;

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int context_len = seq_len - q_len;
    int max_k = context_len + q_pos;
    if (max_k >= seq_len) {
        max_k = seq_len - 1;
    }
    if (max_k < 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = max_k - sliding_window + 1;
        if (win_start > 0) {
            start_k = win_start;
        }
    }

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * q_scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;

    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        float ks = k_scale[block];
        float vs = v_scale[block];

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride;
            float partial = 0.0f;

            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                partial += q_shared[d] * fp8_e4m3_to_float((uchar)key_cache[kv_base + d]) * ks;
            }

            reduce_shared[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_shared[tid] += reduce_shared[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = reduce_shared[0];
                float m_new = fmax(m, score);
                float p = exp(score - m_new);
                float alpha = exp(m - m_new);
                alpha_shared = alpha;
                p_shared = p;
                l = l * alpha + p;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha = alpha_shared;
            float p = p_shared;
            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                out_shared[d] =
                    out_shared[d] * alpha + p * fp8_e4m3_to_float((uchar)value_cache[kv_base + d]) * vs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""


_METAL_DECODE_FP8_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;
    float q_scale = softmax_scale_ptr[0];

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    if ((int)seq_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)seq_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }
    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = seq_len - sliding_window;
        if (win_start > 0) {
            start_k = win_start;
        }
    }
    int max_k = seq_len - 1;

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * q_scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        float ks = k_scale[block];
        float vs = v_scale[block];

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride;
            float partial = 0.0f;

            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                partial += q_shared[d] * fp8_e4m3_to_float((uchar)key_cache[kv_base + d]) * ks;
            }

            reduce_shared[tid] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_shared[tid] += reduce_shared[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = reduce_shared[0];
                float m_new = fmax(m, score);
                float p = exp(score - m_new);
                float alpha = exp(m - m_new);
                alpha_shared = alpha;
                p_shared = p;
                l = l * alpha + p;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha = alpha_shared;
            float p = p_shared;
            for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
                out_shared[d] =
                    out_shared[d] * alpha + p * fp8_e4m3_to_float((uchar)value_cache[kv_base + d]) * vs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""


_METAL_DECODE_HEAD256_FP8_SOURCE = r"""
    constexpr uint kHeadDim = 256;
    constexpr uint kPack = 4;
    constexpr uint kThreads = kHeadDim / kPack;

    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup_id = simdgroup_index_in_threadgroup;
    float q_scale = softmax_scale_ptr[0];

    if (tid >= kThreads) {
        return;
    }

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (head_dim != (int)kHeadDim || total_tokens <= 0 || num_q_heads <= 0) {
        return;
    }
    if ((int)seq_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    int q_base = ((int)seq_idx * num_q_heads + (int)q_head) * head_dim;
    if (num_kv_heads <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        int out_base = q_base + (int)tid * (int)kPack;
        for (uint i = 0; i < kPack; ++i) {
            out[out_base + (int)i] = (T)0;
        }
        return;
    }

    int sliding_window = sliding_window_ptr[0];
    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = seq_len - sliding_window;
        if (win_start > 0) {
            start_k = win_start;
        }
    }
    int max_k = seq_len - 1;

    int q_vec_base = q_base + (int)tid * (int)kPack;
    float4 q_vec = float4(
        (float)queries[q_vec_base + 0] * q_scale,
        (float)queries[q_vec_base + 1] * q_scale,
        (float)queries[q_vec_base + 2] * q_scale,
        (float)queries[q_vec_base + 3] * q_scale
    );
    float4 out_vec = float4(0.0f);

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    int token_stride = num_kv_heads * head_dim;
    int start_block = start_k / block_size;
    int end_block = max_k / block_size;

    threadgroup float simd_sums[2];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    float m = -INFINITY;
    float l = 0.0f;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) {
            continue;
        }

        float ks = k_scale[block];
        float vs = v_scale[block];

        int offset_start = 0;
        int offset_end = block_size - 1;
        if (block_pos == start_block) {
            offset_start = start_k - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = max_k - block_pos * block_size;
        }

        int block_base = (block * block_size * num_kv_heads + kv_head) * head_dim;
        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int kv_base = block_base + offset * token_stride + (int)tid * (int)kPack;

            float4 k_vec = float4(
                fp8_e4m3_to_float((uchar)key_cache[kv_base + 0]) * ks,
                fp8_e4m3_to_float((uchar)key_cache[kv_base + 1]) * ks,
                fp8_e4m3_to_float((uchar)key_cache[kv_base + 2]) * ks,
                fp8_e4m3_to_float((uchar)key_cache[kv_base + 3]) * ks
            );
            float partial = dot(q_vec, k_vec);

            float simd_total = simd_sum(partial);
            if (lane == 0) {
                simd_sums[simdgroup_id] = simd_total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup_id == 0) {
                float group_partial = tid < 2 ? simd_sums[tid] : 0.0f;
                float score = simd_sum(group_partial);
                if (tid == 0) {
                    float m_new = fmax(m, score);
                    float p = exp(score - m_new);
                    float alpha = exp(m - m_new);
                    alpha_shared = alpha;
                    p_shared = p;
                    l = l * alpha + p;
                    m = m_new;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 v_vec = float4(
                fp8_e4m3_to_float((uchar)value_cache[kv_base + 0]) * vs,
                fp8_e4m3_to_float((uchar)value_cache[kv_base + 1]) * vs,
                fp8_e4m3_to_float((uchar)value_cache[kv_base + 2]) * vs,
                fp8_e4m3_to_float((uchar)value_cache[kv_base + 3]) * vs
            );
            out_vec = out_vec * alpha_shared + p_shared * v_vec;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 out_scaled = out_vec * inv_l_shared;
    int out_base = q_base + (int)tid * (int)kPack;
    out[out_base + 0] = (T)out_scaled.x;
    out[out_base + 1] = (T)out_scaled.y;
    out[out_base + 2] = (T)out_scaled.z;
    out[out_base + 3] = (T)out_scaled.w;
"""


_METAL_FP8_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline float fp8_e4m3_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 3) & 0xF;
  const uint man = v & 0x7;
  if (exp == 0 && man == 0) return s ? -0.0f : 0.0f;
  float sign = s ? -1.0f : 1.0f;
  if (exp == 0) {
    return sign * (float(man) / 8.0f) * exp2(-6.0f);
  }
  return sign * (1.0f + float(man) / 8.0f) * exp2(float(exp) - 7.0f);
}
"""

_METAL_FP8_INPUT_NAMES = [
    "queries",
    "key_cache",
    "value_cache",
    "block_tables",
    "kv_lens",
    "query_start_loc",
    "softmax_scale_ptr",
    "sliding_window_ptr",
    "k_scale",
    "v_scale",
]

_METAL_FP8_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged_fp8",
    input_names=_METAL_FP8_INPUT_NAMES,
    output_names=["out"],
    source=_METAL_FP8_SOURCE,
    header=_METAL_FP8_HEADER,
)

_METAL_DECODE_FP8_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged_decode_fp8",
    input_names=_METAL_FP8_INPUT_NAMES,
    output_names=["out"],
    source=_METAL_DECODE_FP8_SOURCE,
    header=_METAL_FP8_HEADER,
)

_METAL_DECODE_HEAD256_FP8_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged_decode_head256_fp8",
    input_names=_METAL_FP8_INPUT_NAMES,
    output_names=["out"],
    source=_METAL_DECODE_HEAD256_FP8_SOURCE,
    header=_METAL_FP8_HEADER,
)

_METAL_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "kv_lens",
        "query_start_loc",
        "softmax_scale_ptr",
        "sliding_window_ptr",
    ],
    output_names=["out"],
    source=_METAL_SOURCE,
    header=_METAL_HEADER,
)

_METAL_DECODE_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged_decode",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "kv_lens",
        "query_start_loc",
        "softmax_scale_ptr",
        "sliding_window_ptr",
    ],
    output_names=["out"],
    source=_METAL_DECODE_SOURCE,
    header=_METAL_HEADER,
)

_METAL_DECODE_HEAD256_KERNEL = mx.fast.metal_kernel(
    name="unified_attention_paged_decode_head256",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "kv_lens",
        "query_start_loc",
        "softmax_scale_ptr",
        "sliding_window_ptr",
    ],
    output_names=["out"],
    source=_METAL_DECODE_HEAD256_SOURCE,
    header=_METAL_HEADER,
)


def paged_attention(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    kv_lens: mx.array,
    query_start_loc: mx.array,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    use_metal: bool | None = None,
    single_token_decode: bool | None = None,
    threadgroup_size: int = 256,
    allow_fallback: bool = True,
    k_scale: mx.array | None = None,
    v_scale: mx.array | None = None,
) -> mx.array:
    """Execute paged attention over block-tabled KV caches.

    Attempts to use a Metal GPU kernel for performance, falling back
    to a NumPy reference implementation if the kernel is unavailable.

    Args:
        queries: Query tensor of shape ``[total_tokens, num_q_heads, head_dim]``.
        key_cache: Key cache of shape ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Value cache with the same shape as ``key_cache``.
        block_tables: Block table of shape ``[num_seqs, max_blocks_per_seq]``.
        kv_lens: KV lengths of shape ``[num_seqs]``.
        query_start_loc: Query start locations of shape ``[num_seqs + 1]``.
        softmax_scale: Scaling factor. Defaults to ``1/sqrt(head_dim)``.
        sliding_window: Sliding window size. None or 0 disables.
        use_metal: Whether to use the Metal kernel. None uses env defaults.
        single_token_decode: Explicit decode fast-path hint. When ``None``,
            this is inferred from ``query_start_loc``.
        threadgroup_size: Metal threadgroup size for kernel dispatch.
        k_scale: Per-block key scale factors for FP8 cache, shape ``[num_blocks]``.
        v_scale: Per-block value scale factors for FP8 cache, shape ``[num_blocks]``.

    Returns:
        Output tensor of the same shape as ``queries``.
    """
    _validate_inputs(queries, key_cache, value_cache, block_tables, kv_lens, query_start_loc)

    head_dim = int(queries.shape[-1])
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if sliding_window is None:
        sliding_window = 0
    if int(queries.shape[0]) == 0:
        return mx.zeros_like(queries)

    if use_metal is None:
        use_metal = _DEFAULT_USE_METAL and not _FORCE_REFERENCE

    is_fp8 = key_cache.dtype == mx.uint8 and k_scale is not None and v_scale is not None

    decode_fast_path = (
        _is_single_token_decode(query_start_loc, int(queries.shape[0]))
        if single_token_decode is None
        else bool(single_token_decode)
    )

    if use_metal and not _FORCE_REFERENCE and head_dim <= _MAX_METAL_HEAD_DIM:
        try:
            total_tokens = int(queries.shape[0])
            num_q_heads = int(queries.shape[1])
            use_head256_decode = decode_fast_path and head_dim == 256

            if is_fp8:
                kernel = (
                    _METAL_DECODE_HEAD256_FP8_KERNEL
                    if use_head256_decode
                    else _METAL_DECODE_FP8_KERNEL
                    if decode_fast_path
                    else _METAL_FP8_KERNEL
                )
            else:
                kernel = (
                    _METAL_DECODE_HEAD256_KERNEL
                    if use_head256_decode
                    else _METAL_DECODE_KERNEL
                    if decode_fast_path
                    else _METAL_KERNEL
                )
            block_tables = _ensure_mx_array(block_tables, dtype=mx.int32)
            kv_lens = _ensure_mx_array(kv_lens, dtype=mx.int32)
            query_start_loc = _ensure_mx_array(query_start_loc, dtype=mx.int32)
            softmax_scale_ptr = _float32_scalar_ptr(float(softmax_scale))
            sliding_window_ptr = _int32_scalar_ptr(int(sliding_window))

            effective_threadgroup_size = (
                64
                if use_head256_decode
                else (
                    _resolve_decode_threadgroup_size(head_dim, threadgroup_size, total_tokens)
                    if decode_fast_path
                    else _resolve_threadgroup_size(head_dim, threadgroup_size)
                )
            )

            inputs = [
                queries,
                key_cache,
                value_cache,
                block_tables,
                kv_lens,
                query_start_loc,
                softmax_scale_ptr,
                sliding_window_ptr,
            ]
            if is_fp8:
                inputs.extend([k_scale, v_scale])

            outputs = kernel(
                inputs=inputs,
                template=[("T", queries.dtype)],
                output_shapes=[queries.shape],
                output_dtypes=[queries.dtype],
                grid=(effective_threadgroup_size, total_tokens, num_q_heads),
                threadgroup=(effective_threadgroup_size, 1, 1),
            )
            return outputs[0]
        except Exception:
            if not allow_fallback:
                raise

    return _paged_attention_reference(
        queries,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        softmax_scale=float(softmax_scale),
        sliding_window=int(sliding_window),
    )


@OperationRegistry.register
class UnifiedAttention(BaseOperation):
    """Paged (unified) attention operation for block-tabled KV caches.

    Registered under ``"unified_attention"`` and dense/hybrid aliases.
    Requires ``PageCacheView`` and associated metadata when used for paged inputs.

    Args:
        metadata: Optional operation metadata for runtime configuration.
        use_metal: Whether to prefer the Metal GPU kernel. Defaults to ``True``.
    """

    def __init__(
        self,
        metadata: tp.Any | None = None,
        *,
        use_metal: bool = True,
    ):
        super().__init__(metadata=metadata)
        self.use_metal = bool(use_metal)

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names for this operation.

        Returns:
            A tuple of name aliases.
        """
        return ("unified_attention", "full_attention", "linear_attention")

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):
        return (
            RequirementsBuilder("unified_attention")
            .require_metadata(
                MetadataField.QUERY_START_LOC,
                MetadataField.BLOCK_TABLES,
                MetadataField.KV_LENS,
                MetadataField.BLOCK_SIZE,
            )
            .optional_metadata(MetadataField.SLIDING_WINDOW)
            .require_cache(PageCacheView)
            .build()
        )

    def forward_native(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        cache_metadata: UnifiedAttnMetadata | PageMetadata | None = None,
        cache_view: PageCacheView | None = None,
        scale: float | None = None,
        use_metal: bool | None = None,
        threadgroup_size: int = 256,
        **_: tp.Any,
    ) -> AttentionOutput:
        """Execute paged attention using cache metadata.

        Args:
            query: Query tensor (used as pre-scaled queries).
            key: Key cache tensor (block-formatted).
            value: Value cache tensor (block-formatted).
            cache_metadata: Paged attention metadata (required).
            cache_view: Optional page cache reference to include in output.
            scale: Softmax scaling factor.
            use_metal: Whether to use the Metal kernel. Defaults to
                the value set during initialization.
            threadgroup_size: Metal threadgroup size.
            **_: Additional keyword arguments (ignored).

        Returns:
            An ``AttentionOutput`` containing attention results and cache view.

        Raises:
            ValueError: If ``cache_metadata`` is None or incomplete.
        """
        if use_metal is None:
            use_metal = self.use_metal
        if cache_metadata is None:
            raise ValueError("UnifiedAttention requires cache_metadata.")
        if isinstance(cache_metadata, PageMetadata):
            if (
                cache_metadata.block_tables is None
                or cache_metadata.kv_lens is None
                or cache_metadata.block_size is None
            ):
                raise ValueError("PageMetadata must be resolved before unified attention execution.")
            cache_metadata = UnifiedAttnMetadata(
                block_tables=cache_metadata.block_tables,
                kv_lens=cache_metadata.kv_lens,
                query_start_loc=cache_metadata.query_start_loc,
                block_size=cache_metadata.block_size,
                sliding_window=cache_metadata.sliding_window,
                is_single_token_decode=cache_metadata.is_single_token_decode,
            )

        # FP8 path: pass per-block scale arrays to the Metal kernel
        # instead of dequantizing in Python.
        runtime_k_scale: mx.array | None = None
        runtime_v_scale: mx.array | None = None
        if cache_view is not None and getattr(cache_view, "cache_dtype_is_fp8", False) and key.dtype == mx.uint8:
            runtime_k_scale = getattr(cache_view, "k_scales", None)
            runtime_v_scale = getattr(cache_view, "v_scales", None)
            if runtime_k_scale is None or runtime_v_scale is None:
                # Fallback: dequantize in Python if scales are missing
                key, value = cache_view.dequantize_kv_cache(
                    key,
                    value,
                    out_dtype=query.dtype,
                )
                runtime_k_scale = None
                runtime_v_scale = None

        outputs = paged_attention(
            query,
            key,
            value,
            cache_metadata.block_tables,
            cache_metadata.kv_lens,
            cache_metadata.query_start_loc,
            softmax_scale=scale,
            sliding_window=cache_metadata.sliding_window,
            use_metal=use_metal,
            single_token_decode=cache_metadata.is_single_token_decode,
            threadgroup_size=threadgroup_size,
            k_scale=runtime_k_scale,
            v_scale=runtime_v_scale,
        )
        return AttentionOutput(attention_outputs=outputs, cache_view=cache_view)


__all__ = ("UnifiedAttention", "UnifiedAttnConfig", "UnifiedAttnMetadata", "paged_attention")
