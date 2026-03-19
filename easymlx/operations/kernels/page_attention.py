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

"""Dedicated paged attention kernel for MLX.

This is the standalone block-tabled paged runtime used by the
``PageAttention`` operation. It intentionally lives outside
``UnifiedAttention`` so the paged Metal path can evolve independently.
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
_DEFAULT_USE_MISTRAL = True
_MAX_METAL_HEAD_DIM = 256
_PACKED_DECODE_PARTITION_SIZE = 512
_PACKED_DECODE_V2_MIN_CONTEXT = 768


@dataclass(slots=True)
class PageAttnMetadata:
    """Metadata required for paged attention execution.

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
class PageAttnConfig:
    """Configuration for the page attention kernel.

    Attributes:
        use_metal_kernel: Whether to prefer the Metal GPU kernel.
        threadgroup_size: Metal threadgroup size for the kernel dispatch.
    """

    use_metal_kernel: bool = True
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
    if key_cache.ndim == 4 and value_cache.ndim == 4:
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache must have the same shape")
        if queries.shape[2] != key_cache.shape[3]:
            raise ValueError("queries head_dim must match key/value cache head_dim")
        if queries.shape[1] % key_cache.shape[2] != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")
    elif key_cache.ndim == 5 and value_cache.ndim == 4:
        page_vec_size = int(key_cache.shape[4])
        num_vecs = int(key_cache.shape[2])
        head_dim = int(value_cache.shape[2])
        if head_dim != num_vecs * page_vec_size:
            raise ValueError("packed page key cache shape is inconsistent with value cache head_dim")
        if queries.shape[2] != head_dim:
            raise ValueError("queries head_dim must match packed page cache head_dim")
        if int(key_cache.shape[1]) != int(value_cache.shape[1]):
            raise ValueError("packed page key/value caches must agree on num_kv_heads")
        if int(key_cache.shape[3]) != int(value_cache.shape[3]):
            raise ValueError("packed page key/value caches must agree on block_size")
        if queries.shape[1] % int(key_cache.shape[1]) != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")
    else:
        raise ValueError("key/value cache ranks must be either standard [4,4] or packed page-attention [5,4]")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be rank-2: [num_seqs, max_blocks_per_seq]")
    if kv_lens.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("kv_lens/query_start_loc must be rank-1")
    if query_start_loc.shape[0] != block_tables.shape[0] + 1:
        raise ValueError("query_start_loc must have length num_seqs + 1")


def _is_packed_page_cache_layout(key_cache: mx.array, value_cache: mx.array) -> bool:
    return key_cache.ndim == 5 and value_cache.ndim == 4


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


def _resolve_partitioned_window_info(
    kv_lens: mx.ArrayLike,
    *,
    sliding_window: int,
    partition_size: int,
) -> tuple[int, int]:
    """Return the maximum active KV window and partition count for decode."""
    kv_np = np.asarray(kv_lens, dtype=np.int32)
    if kv_np.size == 0:
        return 0, 1
    max_window = int(kv_np.max())
    if sliding_window > 0:
        max_window = min(max_window, int(sliding_window))
    if max_window <= 0:
        return 0, 1
    num_partitions = max(1, (max_window + int(partition_size) - 1) // int(partition_size))
    return max_window, num_partitions


@lru_cache(maxsize=1)
def _get_page_attention_header() -> str:
    from ._metal_header import PAGED_ATTENTION_HEADER

    return PAGED_ATTENTION_HEADER


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
    """NumPy reference implementation of paged attention for tests.

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


def _paged_attention_reference_packed(
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
    """Reference implementation for packed PageAttention cache layouts."""
    q_np = np.asarray(queries)
    k_cache = np.asarray(key_cache)
    v_cache = np.asarray(value_cache)
    bt = np.asarray(block_tables)
    kv = np.asarray(kv_lens)
    qsl = np.asarray(query_start_loc)

    _total_tokens, num_q_heads, head_dim = q_np.shape
    num_blocks, num_kv_heads, num_vecs, block_size, page_vec_size = k_cache.shape
    if num_vecs * page_vec_size != head_dim:
        raise ValueError("packed key cache shape does not match query head_dim")
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
            k_seq[pos] = k_cache[block, :, :, offset, :].reshape(num_kv_heads, head_dim)
            v_seq[pos] = v_cache[block, :, :, offset]

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


_METAL_PACKED_DECODE_HEAD256_SOURCE = r"""
    constexpr uint kHeadDim = 256;
    constexpr uint kPack = 16 / sizeof(T);
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

    int num_kv_heads = key_cache_shape[1];
    int block_size = key_cache_shape[3];
    int num_vecs = key_cache_shape[2];
    int packed_stride = key_cache_shape[4];
    if (num_vecs != (int)kThreads || packed_stride != (int)kPack) {
        return;
    }

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

    thread float q_vec[kPack];
    thread float out_vec[kPack];
    int q_vec_base = q_base + (int)tid * (int)kPack;
    for (uint i = 0; i < kPack; ++i) {
        q_vec[i] = (float)queries[q_vec_base + (int)i] * q_scale;
        out_vec[i] = 0.0f;
    }

    int max_blocks_per_seq = block_tables_shape[1];
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

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int packed_base =
                ((((block * num_kv_heads + kv_head) * num_vecs + (int)tid) * block_size + offset) * (int)kPack);
            float partial = 0.0f;
            for (uint i = 0; i < kPack; ++i) {
                partial += q_vec[i] * (float)key_cache[packed_base + (int)i];
            }

            float simd_total = simd_sum(partial);
            if (lane == 0) {
                simd_sums[simdgroup_id] = simd_total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup_id == 0) {
                float group_partial = tid < (kThreads / 32) ? simd_sums[tid] : 0.0f;
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

            int value_base =
                (((block * num_kv_heads + kv_head) * head_dim + (int)tid * (int)kPack) * block_size + offset);
            for (uint i = 0; i < kPack; ++i) {
                out_vec[i] = out_vec[i] * alpha_shared +
                    p_shared * (float)value_cache[value_base + (int)i * block_size];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int out_base = q_base + (int)tid * (int)kPack;
    for (uint i = 0; i < kPack; ++i) {
        out[out_base + (int)i] = (T)(out_vec[i] * inv_l_shared);
    }
"""


_METAL_PACKED_DECODE_HEAD256_PARTITIONED_SOURCE = r"""
    constexpr uint kHeadDim = 256;
    constexpr uint kPack = 16 / sizeof(T);
    constexpr uint kThreads = kHeadDim / kPack;

    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_partition_idx = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup_id = simdgroup_index_in_threadgroup;
    float q_scale = softmax_scale_ptr[0];

    if (tid >= kThreads) {
        return;
    }

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    int num_partitions = partition_count_ptr[0];
    if (head_dim != (int)kHeadDim || total_tokens <= 0 || num_q_heads <= 0 || num_partitions <= 0) {
        return;
    }

    int partition_idx = (int)head_partition_idx / num_q_heads;
    int q_head = (int)head_partition_idx - partition_idx * num_q_heads;
    if ((int)seq_idx >= total_tokens || q_head < 0 || q_head >= num_q_heads || partition_idx >= num_partitions) {
        return;
    }

    int num_kv_heads = key_cache_shape[1];
    int block_size = key_cache_shape[3];
    int num_vecs = key_cache_shape[2];
    int packed_stride = key_cache_shape[4];
    if (num_vecs != (int)kThreads || packed_stride != (int)kPack) {
        return;
    }

    int partition_meta_idx = ((int)seq_idx * num_q_heads + q_head) * num_partitions + partition_idx;
    int partition_out_base = partition_meta_idx * head_dim + (int)tid * (int)kPack;

    if (num_kv_heads <= 0) {
        for (uint i = 0; i < kPack; ++i) {
            tmp_out[partition_out_base + (int)i] = 0.0f;
        }
        if (tid == 0) {
            tmp_m[partition_meta_idx] = -INFINITY;
            tmp_l[partition_meta_idx] = 0.0f;
        }
        return;
    }

    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        for (uint i = 0; i < kPack; ++i) {
            tmp_out[partition_out_base + (int)i] = 0.0f;
        }
        if (tid == 0) {
            tmp_m[partition_meta_idx] = -INFINITY;
            tmp_l[partition_meta_idx] = 0.0f;
        }
        return;
    }

    int kv_head = q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        for (uint i = 0; i < kPack; ++i) {
            tmp_out[partition_out_base + (int)i] = 0.0f;
        }
        if (tid == 0) {
            tmp_m[partition_meta_idx] = -INFINITY;
            tmp_l[partition_meta_idx] = 0.0f;
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
    int partition_size = partition_size_ptr[0];
    int part_start = start_k + partition_idx * partition_size;
    if (part_start > max_k || partition_size <= 0) {
        for (uint i = 0; i < kPack; ++i) {
            tmp_out[partition_out_base + (int)i] = 0.0f;
        }
        if (tid == 0) {
            tmp_m[partition_meta_idx] = -INFINITY;
            tmp_l[partition_meta_idx] = 0.0f;
        }
        return;
    }
    int part_end = part_start + partition_size - 1;
    if (part_end > max_k) {
        part_end = max_k;
    }

    int q_base = ((int)seq_idx * num_q_heads + q_head) * head_dim;
    thread float q_vec[kPack];
    thread float out_vec[kPack];
    int q_vec_base = q_base + (int)tid * (int)kPack;
    for (uint i = 0; i < kPack; ++i) {
        q_vec[i] = (float)queries[q_vec_base + (int)i] * q_scale;
        out_vec[i] = 0.0f;
    }

    int max_blocks_per_seq = block_tables_shape[1];
    int start_block = part_start / block_size;
    int end_block = part_end / block_size;

    threadgroup float simd_sums[2];
    threadgroup float alpha_shared;
    threadgroup float p_shared;

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
            offset_start = part_start - block_pos * block_size;
        }
        if (block_pos == end_block) {
            offset_end = part_end - block_pos * block_size;
        }

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            int packed_base =
                ((((block * num_kv_heads + kv_head) * num_vecs + (int)tid) * block_size + offset) * (int)kPack);
            float partial = 0.0f;
            for (uint i = 0; i < kPack; ++i) {
                partial += q_vec[i] * (float)key_cache[packed_base + (int)i];
            }

            float simd_total = simd_sum(partial);
            if (lane == 0) {
                simd_sums[simdgroup_id] = simd_total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdgroup_id == 0) {
                float group_partial = tid < (kThreads / 32) ? simd_sums[tid] : 0.0f;
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

            int value_base =
                (((block * num_kv_heads + kv_head) * head_dim + (int)tid * (int)kPack) * block_size + offset);
            for (uint i = 0; i < kPack; ++i) {
                out_vec[i] = out_vec[i] * alpha_shared +
                    p_shared * (float)value_cache[value_base + (int)i * block_size];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = 0; i < kPack; ++i) {
        tmp_out[partition_out_base + (int)i] = out_vec[i];
    }
    if (tid == 0) {
        tmp_m[partition_meta_idx] = m;
        tmp_l[partition_meta_idx] = l;
    }
"""


_METAL_PACKED_DECODE_HEAD256_REDUCE_SOURCE = r"""
    constexpr uint kHeadDim = 256;
    constexpr uint kPack = 4;
    constexpr uint kThreads = kHeadDim / kPack;

    uint tid = thread_position_in_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;

    if (tid >= kThreads) {
        return;
    }

    int total_tokens = tmp_out_shape[0];
    int num_q_heads = tmp_out_shape[1];
    int head_dim = tmp_out_shape[3];
    int num_partitions = partition_count_ptr[0];
    if (head_dim != (int)kHeadDim || total_tokens <= 0 || num_q_heads <= 0 || num_partitions <= 0) {
        return;
    }
    if ((int)seq_idx >= total_tokens || (int)q_head >= num_q_heads) {
        return;
    }

    int partition_base = ((int)seq_idx * num_q_heads + (int)q_head) * num_partitions;
    int out_base = ((int)seq_idx * num_q_heads + (int)q_head) * head_dim + (int)tid * (int)kPack;

    threadgroup float global_m_shared;
    threadgroup float global_l_shared;

    if (tid == 0) {
        float global_m = -INFINITY;
        for (int part = 0; part < num_partitions; ++part) {
            global_m = fmax(global_m, tmp_m[partition_base + part]);
        }
        float global_l = 0.0f;
        if (global_m != -INFINITY) {
            for (int part = 0; part < num_partitions; ++part) {
                float part_l = tmp_l[partition_base + part];
                if (part_l > 0.0f) {
                    global_l += part_l * exp(tmp_m[partition_base + part] - global_m);
                }
            }
        }
        global_m_shared = global_m;
        global_l_shared = global_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 out_vec = float4(0.0f);
    if (global_l_shared > 0.0f) {
        for (int part = 0; part < num_partitions; ++part) {
            float part_l = tmp_l[partition_base + part];
            if (part_l <= 0.0f) {
                continue;
            }
            float weight = exp(tmp_m[partition_base + part] - global_m_shared);
            int part_vec_base =
                (((int)seq_idx * num_q_heads + (int)q_head) * num_partitions + part) * head_dim
                + (int)tid * (int)kPack;
            float4 part_vec = float4(
                tmp_out[part_vec_base + 0],
                tmp_out[part_vec_base + 1],
                tmp_out[part_vec_base + 2],
                tmp_out[part_vec_base + 3]
            );
            out_vec += weight * part_vec;
        }
        out_vec /= global_l_shared;
    }

    out[out_base + 0] = (T)out_vec.x;
    out[out_base + 1] = (T)out_vec.y;
    out[out_base + 2] = (T)out_vec.z;
    out[out_base + 3] = (T)out_vec.w;
"""


_METAL_KERNEL = mx.fast.metal_kernel(
    name="page_attention_paged",
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
    header=_get_page_attention_header(),
)

_METAL_DECODE_KERNEL = mx.fast.metal_kernel(
    name="page_attention_paged_decode",
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
    header=_get_page_attention_header(),
)

_METAL_DECODE_HEAD256_KERNEL = mx.fast.metal_kernel(
    name="page_attention_paged_decode_head256",
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
    header=_get_page_attention_header(),
)

_METAL_PACKED_DECODE_HEAD256_KERNEL = mx.fast.metal_kernel(
    name="page_attention_packed_decode_head256",
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
    source=_METAL_PACKED_DECODE_HEAD256_SOURCE,
    header=_get_page_attention_header(),
)

_METAL_PACKED_DECODE_HEAD256_PARTITIONED_KERNEL = mx.fast.metal_kernel(
    name="page_attention_packed_decode_head256_partitioned",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "kv_lens",
        "query_start_loc",
        "softmax_scale_ptr",
        "sliding_window_ptr",
        "partition_size_ptr",
        "partition_count_ptr",
    ],
    output_names=["tmp_out", "tmp_m", "tmp_l"],
    source=_METAL_PACKED_DECODE_HEAD256_PARTITIONED_SOURCE,
    header=_get_page_attention_header(),
)

_METAL_PACKED_DECODE_HEAD256_REDUCE_KERNEL = mx.fast.metal_kernel(
    name="page_attention_packed_decode_head256_reduce",
    input_names=["tmp_out", "tmp_m", "tmp_l", "partition_count_ptr"],
    output_names=["out"],
    source=_METAL_PACKED_DECODE_HEAD256_REDUCE_SOURCE,
    header=_get_page_attention_header(),
)


_PAGED_ATTENTION_V1_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;

    int num_seqs = context_lens_shape[0];
    int num_heads = queries_shape[1];
    int head_size = params[2];
    int block_size = params[3];
    int num_kv_heads = params[0];
    int max_blocks_per_seq = params[1];
    int q_stride = params[4];
    int kv_block_stride = params[5];
    int kv_head_stride = params[6];
    int sliding_window = params[7];
    float scale = scale_ptr[0];

    if ((int)seq_idx >= num_seqs || (int)head_idx >= num_heads) return;

    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head = (int)head_idx / num_queries_per_kv;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int context_len = context_lens[seq_idx];
    if (context_len <= 0) {
        int q_base = ((int)seq_idx * num_heads + (int)head_idx) * head_size;
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = context_len - sliding_window;
        if (win_start > 0) start_k = win_start;
    }

    int q_base = ((int)seq_idx * num_heads + (int)head_idx) * head_size;

    // Load query into shared memory
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;

    int num_context_blocks = (context_len + block_size - 1) / block_size;
    int start_block = start_k / block_size;
    int x_val = 16 / sizeof(T);  // elements per cache line

    for (int block_pos = start_block; block_pos < num_context_blocks; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) continue;

        int offset_start = (block_pos == start_block) ? (start_k - block_pos * block_size) : 0;
        int offset_end = (block_pos == num_context_blocks - 1)
            ? ((context_len - 1) % block_size) : (block_size - 1);

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            // Compute Q*K dot product
            float partial = 0.0f;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int x_idx = d / x_val;
                int x_offset = d % x_val;
                int k_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            x_idx * block_size * x_val + offset * x_val + x_offset;
                partial += q_shared[d] * (float)key_cache[k_idx];
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
                float score_val = reduce_shared[0];
                float m_new = fmax(m, score_val);
                float p_val = exp(score_val - m_new);
                float alpha_val = exp(m - m_new);
                alpha_shared = alpha_val;
                p_shared = p_val;
                l = l * alpha_val + p_val;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha_val = alpha_shared;
            float p_val = p_shared;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int v_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            d * block_size + offset;
                out_shared[d] = out_shared[d] * alpha_val + p_val * (float)value_cache[v_idx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""

_PAGED_ATTENTION_V1_FP8_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;

    int num_seqs = context_lens_shape[0];
    int num_heads = queries_shape[1];
    int head_size = params[2];
    int block_size = params[3];
    int num_kv_heads = params[0];
    int max_blocks_per_seq = params[1];
    int q_stride = params[4];
    int kv_block_stride = params[5];
    int kv_head_stride = params[6];
    int sliding_window = params[7];
    float scale = scale_ptr[0];

    if ((int)seq_idx >= num_seqs || (int)head_idx >= num_heads) return;

    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head = (int)head_idx / num_queries_per_kv;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int context_len = context_lens[seq_idx];
    if (context_len <= 0) {
        int q_base = ((int)seq_idx * num_heads + (int)head_idx) * head_size;
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            out[q_base + d] = (T)0;
        }
        return;
    }

    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = context_len - sliding_window;
        if (win_start > 0) start_k = win_start;
    }

    int q_base = ((int)seq_idx * num_heads + (int)head_idx) * head_size;

    // Load query into shared memory
    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;

    int num_context_blocks = (context_len + block_size - 1) / block_size;
    int start_block = start_k / block_size;
    int x_val = 16;  // FP8 uses sizeof(uchar)==1, so 16/1 = 16

    for (int block_pos = start_block; block_pos < num_context_blocks; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) continue;

        float ks = k_scale[block];
        float vs = v_scale[block];

        int offset_start = (block_pos == start_block) ? (start_k - block_pos * block_size) : 0;
        int offset_end = (block_pos == num_context_blocks - 1)
            ? ((context_len - 1) % block_size) : (block_size - 1);

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            // Compute Q*K dot product with FP8 dequantization
            float partial = 0.0f;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int x_idx = d / x_val;
                int x_offset = d % x_val;
                int k_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            x_idx * block_size * x_val + offset * x_val + x_offset;
                partial += q_shared[d] * fp8_e4m3_to_float((uchar)key_cache[k_idx]) * ks;
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
                float score_val = reduce_shared[0];
                float m_new = fmax(m, score_val);
                float p_val = exp(score_val - m_new);
                float alpha_val = exp(m - m_new);
                alpha_shared = alpha_val;
                p_shared = p_val;
                l = l * alpha_val + p_val;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha_val = alpha_shared;
            float p_val = p_shared;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int v_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            d * block_size + offset;
                out_shared[d] = out_shared[d] * alpha_val +
                    p_val * fp8_e4m3_to_float((uchar)value_cache[v_idx]) * vs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        out[q_base + d] = (T)(out_shared[d] * inv_l_shared);
    }
"""

_PAGED_ATTENTION_V2_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_partition_idx = threadgroup_position_in_grid.z;

    int num_seqs = context_lens_shape[0];
    int num_heads = queries_shape[1];
    int head_size = params[2];
    int block_size = params[3];
    int num_kv_heads = params[0];
    int max_blocks_per_seq = params[1];
    int q_stride = params[4];
    int kv_block_stride = params[5];
    int kv_head_stride = params[6];
    int sliding_window = params[7];
    float scale = scale_ptr[0];
    int partition_size = partition_size_ptr[0];
    int num_partitions = tmp_out_shape[2];

    int partition_idx = (int)head_partition_idx / num_heads;
    int head_idx = (int)head_partition_idx % num_heads;

    if ((int)seq_idx >= num_seqs || head_idx < 0 || head_idx >= num_heads || partition_idx >= num_partitions) return;

    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head = head_idx / num_queries_per_kv;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int context_len = context_lens[seq_idx];
    int meta_idx = ((int)seq_idx * num_heads + head_idx) * num_partitions + partition_idx;
    int out_base = meta_idx * head_size + (int)tid;

    if (context_len <= 0 || partition_idx * partition_size >= context_len) {
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            tmp_out[meta_idx * head_size + d] = 0.0f;
        }
        if (tid == 0) {
            exp_sums[meta_idx] = 0.0f;
            max_logits[meta_idx] = -INFINITY;
        }
        return;
    }

    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = context_len - sliding_window;
        if (win_start > 0) start_k = win_start;
    }

    int part_start = start_k + partition_idx * partition_size;
    int part_end = part_start + partition_size - 1;
    if (part_end >= context_len) part_end = context_len - 1;
    if (part_start > part_end) {
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            tmp_out[meta_idx * head_size + d] = 0.0f;
        }
        if (tid == 0) {
            exp_sums[meta_idx] = 0.0f;
            max_logits[meta_idx] = -INFINITY;
        }
        return;
    }

    int q_base = ((int)seq_idx * num_heads + head_idx) * head_size;
    int x_val = 16 / sizeof(T);

    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    int start_block = part_start / block_size;
    int end_block = part_end / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) continue;

        int offset_start = (block_pos == start_block) ? (part_start - block_pos * block_size) : 0;
        int offset_end = (block_pos == end_block) ? (part_end - block_pos * block_size) : (block_size - 1);

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            float partial = 0.0f;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int x_idx = d / x_val;
                int x_offset = d % x_val;
                int k_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            x_idx * block_size * x_val + offset * x_val + x_offset;
                partial += q_shared[d] * (float)key_cache[k_idx];
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
                float score_val = reduce_shared[0];
                float m_new = fmax(m, score_val);
                float p_val = exp(score_val - m_new);
                float alpha_val = exp(m - m_new);
                alpha_shared = alpha_val;
                p_shared = p_val;
                l = l * alpha_val + p_val;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha_val = alpha_shared;
            float p_val = p_shared;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int v_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            d * block_size + offset;
                out_shared[d] = out_shared[d] * alpha_val + p_val * (float)value_cache[v_idx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        tmp_out[meta_idx * head_size + d] = out_shared[d];
    }
    if (tid == 0) {
        exp_sums[meta_idx] = l;
        max_logits[meta_idx] = m;
    }
"""

_PAGED_ATTENTION_V2_FP8_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_partition_idx = threadgroup_position_in_grid.z;

    int num_seqs = context_lens_shape[0];
    int num_heads = queries_shape[1];
    int head_size = params[2];
    int block_size = params[3];
    int num_kv_heads = params[0];
    int max_blocks_per_seq = params[1];
    int q_stride = params[4];
    int kv_block_stride = params[5];
    int kv_head_stride = params[6];
    int sliding_window = params[7];
    float scale = scale_ptr[0];
    int partition_size = partition_size_ptr[0];
    int num_partitions = tmp_out_shape[2];

    int partition_idx = (int)head_partition_idx / num_heads;
    int head_idx = (int)head_partition_idx % num_heads;

    if ((int)seq_idx >= num_seqs || head_idx < 0 || head_idx >= num_heads || partition_idx >= num_partitions) return;

    int num_queries_per_kv = num_heads / num_kv_heads;
    int kv_head = head_idx / num_queries_per_kv;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int context_len = context_lens[seq_idx];
    int meta_idx = ((int)seq_idx * num_heads + head_idx) * num_partitions + partition_idx;
    int out_base = meta_idx * head_size + (int)tid;

    if (context_len <= 0 || partition_idx * partition_size >= context_len) {
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            tmp_out[meta_idx * head_size + d] = 0.0f;
        }
        if (tid == 0) {
            exp_sums[meta_idx] = 0.0f;
            max_logits[meta_idx] = -INFINITY;
        }
        return;
    }

    int start_k = 0;
    if (sliding_window > 0) {
        int win_start = context_len - sliding_window;
        if (win_start > 0) start_k = win_start;
    }

    int part_start = start_k + partition_idx * partition_size;
    int part_end = part_start + partition_size - 1;
    if (part_end >= context_len) part_end = context_len - 1;
    if (part_start > part_end) {
        for (int d = (int)tid; d < head_size; d += (int)tg_size) {
            tmp_out[meta_idx * head_size + d] = 0.0f;
        }
        if (tid == 0) {
            exp_sums[meta_idx] = 0.0f;
            max_logits[meta_idx] = -INFINITY;
        }
        return;
    }

    int q_base = ((int)seq_idx * num_heads + head_idx) * head_size;
    int x_val = 16;  // FP8: sizeof(uchar)==1, so 16/1 = 16

    threadgroup float q_shared[256];
    threadgroup float out_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * scale;
        out_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    int start_block = part_start / block_size;
    int end_block = part_end / block_size;

    for (int block_pos = start_block; block_pos <= end_block; ++block_pos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + block_pos];
        if (block < 0) continue;

        float ks = k_scale[block];
        float vs = v_scale[block];

        int offset_start = (block_pos == start_block) ? (part_start - block_pos * block_size) : 0;
        int offset_end = (block_pos == end_block) ? (part_end - block_pos * block_size) : (block_size - 1);

        for (int offset = offset_start; offset <= offset_end; ++offset) {
            float partial = 0.0f;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int x_idx = d / x_val;
                int x_offset = d % x_val;
                int k_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            x_idx * block_size * x_val + offset * x_val + x_offset;
                partial += q_shared[d] * fp8_e4m3_to_float((uchar)key_cache[k_idx]) * ks;
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
                float score_val = reduce_shared[0];
                float m_new = fmax(m, score_val);
                float p_val = exp(score_val - m_new);
                float alpha_val = exp(m - m_new);
                alpha_shared = alpha_val;
                p_shared = p_val;
                l = l * alpha_val + p_val;
                m = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float alpha_val = alpha_shared;
            float p_val = p_shared;
            for (int d = (int)tid; d < head_size; d += (int)tg_size) {
                int v_idx = block * kv_block_stride + kv_head * kv_head_stride +
                            d * block_size + offset;
                out_shared[d] = out_shared[d] * alpha_val +
                    p_val * fp8_e4m3_to_float((uchar)value_cache[v_idx]) * vs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        tmp_out[meta_idx * head_size + d] = out_shared[d];
    }
    if (tid == 0) {
        exp_sums[meta_idx] = l;
        max_logits[meta_idx] = m;
    }
"""

_PAGED_ATTENTION_V2_REDUCE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint seq_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;

    int num_seqs = tmp_out_shape[0];
    int num_heads = tmp_out_shape[1];
    int num_partitions = partition_count_ptr[0];
    int head_size = tmp_out_shape[3];

    if ((int)seq_idx >= num_seqs || (int)head_idx >= num_heads || num_partitions <= 0 || head_size <= 0) return;

    int partition_base = ((int)seq_idx * num_heads + (int)head_idx) * num_partitions;
    int out_offset = ((int)seq_idx * num_heads + (int)head_idx) * head_size;

    // Find global max
    threadgroup float global_m_shared;
    threadgroup float global_l_shared;

    if (tid == 0) {
        float global_m = -INFINITY;
        for (int part = 0; part < num_partitions; ++part) {
            global_m = fmax(global_m, max_logits[partition_base + part]);
        }
        float global_l = 0.0f;
        if (global_m != -INFINITY) {
            for (int part = 0; part < num_partitions; ++part) {
                float part_l = exp_sums[partition_base + part];
                if (part_l > 0.0f) {
                    global_l += part_l * exp(max_logits[partition_base + part] - global_m);
                }
            }
        }
        global_m_shared = global_m;
        global_l_shared = global_l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_size; d += (int)tg_size) {
        float acc = 0.0f;
        if (global_l_shared > 0.0f) {
            for (int part = 0; part < num_partitions; ++part) {
                float part_l = exp_sums[partition_base + part];
                if (part_l <= 0.0f) continue;
                float weight = exp(max_logits[partition_base + part] - global_m_shared);
                int part_vec_idx = (partition_base + part) * head_size + d;
                acc += tmp_out[part_vec_idx] * weight;
            }
            acc /= global_l_shared;
        }
        out[out_offset + d] = (T)acc;
    }
"""


_MISTRAL_PA_V1_KERNEL = mx.fast.metal_kernel(
    name="paged_attention_v1",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lens",
        "params",
        "scale_ptr",
    ],
    output_names=["out"],
    source=_PAGED_ATTENTION_V1_SOURCE,
    header=_get_page_attention_header(),
)

_MISTRAL_PA_V2_KERNEL = mx.fast.metal_kernel(
    name="paged_attention_v2",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lens",
        "params",
        "scale_ptr",
        "partition_size_ptr",
    ],
    output_names=["tmp_out", "exp_sums", "max_logits"],
    source=_PAGED_ATTENTION_V2_SOURCE,
    header=_get_page_attention_header(),
)

_MISTRAL_PA_V2_REDUCE_KERNEL = mx.fast.metal_kernel(
    name="paged_attention_v2_reduce",
    input_names=["tmp_out", "exp_sums", "max_logits", "partition_count_ptr"],
    output_names=["out"],
    source=_PAGED_ATTENTION_V2_REDUCE_SOURCE,
    header=_get_page_attention_header(),
)

_MISTRAL_PA_V1_FP8_KERNEL = mx.fast.metal_kernel(
    name="paged_attention_v1_fp8",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lens",
        "params",
        "scale_ptr",
        "k_scale",
        "v_scale",
    ],
    output_names=["out"],
    source=_PAGED_ATTENTION_V1_FP8_SOURCE,
    header=_get_page_attention_header(),
)

_MISTRAL_PA_V2_FP8_KERNEL = mx.fast.metal_kernel(
    name="paged_attention_v2_fp8",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "block_tables",
        "context_lens",
        "params",
        "scale_ptr",
        "partition_size_ptr",
        "k_scale",
        "v_scale",
    ],
    output_names=["tmp_out", "exp_sums", "max_logits"],
    source=_PAGED_ATTENTION_V2_FP8_SOURCE,
    header=_get_page_attention_header(),
)

# V2 reduce does not read cache, so it is shared between float and FP8 paths.


def _paged_attention_mistral(
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_tables: mx.array,
    kv_lens: mx.array,
    query_start_loc: mx.array,
    *,
    softmax_scale: float,
    sliding_window: int,
    threadgroup_size: int = 256,
    partition_size: int = 512,
    v2_min_context: int = 768,
    k_scale: mx.array | None = None,
    v_scale: mx.array | None = None,
) -> mx.array:
    """Execute paged attention using the mistral.rs-derived Metal kernels.

    This path uses the packed page cache layout (key_cache rank-5, value_cache
    rank-4) with the mistral.rs vector helper infrastructure.  It supports
    arbitrary head dimensions up to 256.

    When ``k_scale`` and ``v_scale`` are provided and the cache dtype is
    ``uint8``, FP8 dequantization is performed inside the Metal kernel.

    The function handles single-token decode only.  For prefill or non-packed
    layouts, the caller should fall back to the inline kernels.
    """
    total_tokens = int(queries.shape[0])
    num_q_heads = int(queries.shape[1])
    head_dim = int(queries.shape[2])

    is_fp8 = key_cache.dtype == mx.uint8 and k_scale is not None and v_scale is not None

    num_kv_heads = int(key_cache.shape[1])
    block_size = int(key_cache.shape[3])
    page_vec_size = int(key_cache.shape[4])
    num_vecs = int(key_cache.shape[2])

    kv_block_stride = num_kv_heads * num_vecs * block_size * page_vec_size
    kv_head_stride = num_vecs * block_size * page_vec_size
    q_stride = num_q_heads * head_dim
    max_blocks_per_seq = int(block_tables.shape[1])

    block_tables = _ensure_mx_array(block_tables, dtype=mx.int32)
    kv_lens = _ensure_mx_array(kv_lens, dtype=mx.int32)

    scale_ptr = _float32_scalar_ptr(float(softmax_scale))
    params = mx.array(
        [
            num_kv_heads,
            max_blocks_per_seq,
            head_dim,
            block_size,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            int(sliding_window),
        ],
        dtype=mx.int32,
    )

    effective_tg = min(256, max(32, _next_power_of_two(head_dim)))

    max_window, num_partitions = _resolve_partitioned_window_info(
        kv_lens,
        sliding_window=int(sliding_window),
        partition_size=partition_size,
    )
    use_v2 = max_window >= v2_min_context and num_partitions > 1

    if use_v2:
        partition_size_ptr = _int32_scalar_ptr(partition_size)
        partition_count_ptr = _int32_scalar_ptr(num_partitions)
        if is_fp8:
            kernel_v2 = _MISTRAL_PA_V2_FP8_KERNEL
            tmp_outputs = kernel_v2(
                inputs=[
                    queries,
                    key_cache,
                    value_cache,
                    block_tables,
                    kv_lens,
                    params,
                    scale_ptr,
                    partition_size_ptr,
                    k_scale,
                    v_scale,
                ],
                template=[("T", queries.dtype)],
                output_shapes=[
                    (total_tokens, num_q_heads, num_partitions, head_dim),
                    (total_tokens, num_q_heads, num_partitions),
                    (total_tokens, num_q_heads, num_partitions),
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
                grid=(effective_tg, total_tokens, num_q_heads * num_partitions),
                threadgroup=(effective_tg, 1, 1),
            )
        else:
            kernel_v2 = _MISTRAL_PA_V2_KERNEL
            tmp_outputs = kernel_v2(
                inputs=[
                    queries,
                    key_cache,
                    value_cache,
                    block_tables,
                    kv_lens,
                    params,
                    scale_ptr,
                    partition_size_ptr,
                ],
                template=[("T", queries.dtype)],
                output_shapes=[
                    (total_tokens, num_q_heads, num_partitions, head_dim),
                    (total_tokens, num_q_heads, num_partitions),
                    (total_tokens, num_q_heads, num_partitions),
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
                grid=(effective_tg, total_tokens, num_q_heads * num_partitions),
                threadgroup=(effective_tg, 1, 1),
            )
        reduce_kernel = _MISTRAL_PA_V2_REDUCE_KERNEL
        reduced = reduce_kernel(
            inputs=[tmp_outputs[0], tmp_outputs[1], tmp_outputs[2], partition_count_ptr],
            template=[("T", queries.dtype)],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(effective_tg, total_tokens, num_q_heads),
            threadgroup=(effective_tg, 1, 1),
        )
        return reduced[0]

    if is_fp8:
        kernel_v1 = _MISTRAL_PA_V1_FP8_KERNEL
        outputs = kernel_v1(
            inputs=[
                queries,
                key_cache,
                value_cache,
                block_tables,
                kv_lens,
                params,
                scale_ptr,
                k_scale,
                v_scale,
            ],
            template=[("T", queries.dtype)],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(effective_tg, total_tokens, num_q_heads),
            threadgroup=(effective_tg, 1, 1),
        )
    else:
        kernel_v1 = _MISTRAL_PA_V1_KERNEL
        outputs = kernel_v1(
            inputs=[
                queries,
                key_cache,
                value_cache,
                block_tables,
                kv_lens,
                params,
                scale_ptr,
            ],
            template=[("T", queries.dtype)],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(effective_tg, total_tokens, num_q_heads),
            threadgroup=(effective_tg, 1, 1),
        )
    return outputs[0]


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
    use_mistral: bool | None = None,
    single_token_decode: bool | None = None,
    threadgroup_size: int = 256,
    partition_size: int | None = None,
    v2_min_context: int | None = None,
    k_scale: mx.array | None = None,
    v_scale: mx.array | None = None,
) -> mx.array:
    """Execute paged attention over block-tabled KV caches.

    Args:
        queries: Query tensor of shape ``[total_tokens, num_q_heads, head_dim]``.
        key_cache: Key cache of shape ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Value cache with the same shape as ``key_cache``.
        block_tables: Block table of shape ``[num_seqs, max_blocks_per_seq]``.
        kv_lens: KV lengths of shape ``[num_seqs]``.
        query_start_loc: Query start locations of shape ``[num_seqs + 1]``.
        softmax_scale: Scaling factor. Defaults to ``1/sqrt(head_dim)``.
        sliding_window: Sliding window size. None or 0 disables.
        use_metal: Whether to use the Metal kernel. ``PageAttention`` is
            Metal-only; ``False`` raises.
        use_mistral: Whether to prefer the mistral.rs-derived Metal kernels
            for packed-layout decode.  Defaults to ``True``.  Set ``False``
            to fall back to the inline kernels.
        single_token_decode: Explicit decode fast-path hint. When ``None``,
            this is inferred from ``query_start_loc``.
        threadgroup_size: Metal threadgroup size for kernel dispatch.

    Returns:
        Output tensor of the same shape as ``queries``.

    Raises:
        NotImplementedError: If Metal execution is disabled or the requested
            shape/layout is not implemented by the PageAttention kernels.
    """
    _validate_inputs(queries, key_cache, value_cache, block_tables, kv_lens, query_start_loc)

    if partition_size is None:
        partition_size = _PACKED_DECODE_PARTITION_SIZE
    partition_size = max(256, partition_size)
    if v2_min_context is None:
        v2_min_context = _PACKED_DECODE_V2_MIN_CONTEXT
    v2_min_context = max((partition_size * 3) // 2, v2_min_context)

    head_dim = int(queries.shape[-1])
    packed_layout = _is_packed_page_cache_layout(key_cache, value_cache)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if sliding_window is None:
        sliding_window = 0
    if int(queries.shape[0]) == 0:
        return mx.zeros_like(queries)

    if use_metal is None:
        use_metal = _DEFAULT_USE_METAL
    if not use_metal:
        raise NotImplementedError("PageAttention is Metal-only; reference fallback is disabled.")

    decode_fast_path = (
        _is_single_token_decode(query_start_loc, int(queries.shape[0]))
        if single_token_decode is None
        else bool(single_token_decode)
    )

    if packed_layout and decode_fast_path and head_dim <= _MAX_METAL_HEAD_DIM:
        return _paged_attention_mistral(
            queries,
            key_cache,
            value_cache,
            block_tables,
            kv_lens,
            query_start_loc,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            threadgroup_size=threadgroup_size,
            partition_size=partition_size,
            v2_min_context=v2_min_context,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    if packed_layout and head_dim == 256:
        if decode_fast_path:
            total_tokens = int(queries.shape[0])
            num_q_heads = int(queries.shape[1])
            block_tables = _ensure_mx_array(block_tables, dtype=mx.int32)
            kv_lens = _ensure_mx_array(kv_lens, dtype=mx.int32)
            query_start_loc = _ensure_mx_array(query_start_loc, dtype=mx.int32)
            softmax_scale_ptr = _float32_scalar_ptr(float(softmax_scale))
            sliding_window_ptr = _int32_scalar_ptr(int(sliding_window))
            page_vec_size = int(key_cache.shape[4])
            effective_threadgroup_size = head_dim // page_vec_size
            max_window, num_partitions = _resolve_partitioned_window_info(
                kv_lens,
                sliding_window=int(sliding_window),
                partition_size=partition_size,
            )
            use_partitioned_decode = max_window >= v2_min_context and num_partitions > 1
            if use_partitioned_decode:
                partition_size_ptr = _int32_scalar_ptr(partition_size)
                partition_count_ptr = _int32_scalar_ptr(num_partitions)
                tmp_outputs = _METAL_PACKED_DECODE_HEAD256_PARTITIONED_KERNEL(
                    inputs=[
                        queries,
                        key_cache,
                        value_cache,
                        block_tables,
                        kv_lens,
                        query_start_loc,
                        softmax_scale_ptr,
                        sliding_window_ptr,
                        partition_size_ptr,
                        partition_count_ptr,
                    ],
                    template=[("T", queries.dtype)],
                    output_shapes=[
                        (total_tokens, num_q_heads, num_partitions, head_dim),
                        (total_tokens, num_q_heads, num_partitions),
                        (total_tokens, num_q_heads, num_partitions),
                    ],
                    output_dtypes=[mx.float32, mx.float32, mx.float32],
                    grid=(effective_threadgroup_size, total_tokens, num_q_heads * num_partitions),
                    threadgroup=(effective_threadgroup_size, 1, 1),
                )
                reduced = _METAL_PACKED_DECODE_HEAD256_REDUCE_KERNEL(
                    inputs=[tmp_outputs[0], tmp_outputs[1], tmp_outputs[2], partition_count_ptr],
                    template=[("T", queries.dtype)],
                    output_shapes=[queries.shape],
                    output_dtypes=[queries.dtype],
                    grid=(64, total_tokens, num_q_heads),
                    threadgroup=(64, 1, 1),
                )
                return reduced[0]

            outputs = _METAL_PACKED_DECODE_HEAD256_KERNEL(
                inputs=[
                    queries,
                    key_cache,
                    value_cache,
                    block_tables,
                    kv_lens,
                    query_start_loc,
                    softmax_scale_ptr,
                    sliding_window_ptr,
                ],
                template=[("T", queries.dtype)],
                output_shapes=[queries.shape],
                output_dtypes=[queries.dtype],
                grid=(effective_threadgroup_size, total_tokens, num_q_heads),
                threadgroup=(effective_threadgroup_size, 1, 1),
            )
            return outputs[0]
        raise NotImplementedError(
            "PageAttention packed layout currently only implements single-token decode for head_dim=256."
        )

    if not packed_layout and head_dim <= _MAX_METAL_HEAD_DIM:
        total_tokens = int(queries.shape[0])
        num_q_heads = int(queries.shape[1])
        use_head256_decode = decode_fast_path and head_dim == 256
        kernel = (
            _METAL_DECODE_HEAD256_KERNEL
            if use_head256_decode
            else _METAL_DECODE_KERNEL if decode_fast_path else _METAL_KERNEL
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

        outputs = kernel(
            inputs=[
                queries,
                key_cache,
                value_cache,
                block_tables,
                kv_lens,
                query_start_loc,
                softmax_scale_ptr,
                sliding_window_ptr,
            ],
            template=[("T", queries.dtype)],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(effective_threadgroup_size, total_tokens, num_q_heads),
            threadgroup=(effective_threadgroup_size, 1, 1),
        )
        return outputs[0]
    raise NotImplementedError(
        "PageAttention Metal kernels do not support this shape/layout: "
        f"packed_layout={packed_layout}, head_dim={head_dim}, decode_fast_path={decode_fast_path}."
    )


@OperationRegistry.register
class PageAttention(BaseOperation):
    """Paged attention operation for block-tabled KV caches.

    Registered under ``"page_attention"`` and ``"paged_attention"``.

    Args:
        metadata: Optional operation metadata for runtime configuration.
        partition_size: Partition size for the partitioned decode kernel.
            Defaults to ``512``.
        v2_min_context: Minimum context length before the partitioned
            (v2) decode path is used. Defaults to ``768``.
    """

    def __init__(
        self,
        metadata: tp.Any | None = None,
        *,
        partition_size: int = 512,
        v2_min_context: int = 768,
    ):
        super().__init__(metadata=metadata)
        self.partition_size = max(256, int(partition_size))
        self.v2_min_context = max((self.partition_size * 3) // 2, int(v2_min_context))

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names for this operation.

        Returns:
            A tuple of name aliases.
        """
        return ("page_attention", "paged_attention")

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):
        return (
            RequirementsBuilder("page_attention")
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
        cache_metadata: PageAttnMetadata | PageMetadata | None = None,
        cache_view: PageCacheView | None = None,
        scale: float | None = None,
        use_metal: bool = _DEFAULT_USE_METAL,
        threadgroup_size: int = 256,
        **_: tp.Any,
    ) -> AttentionOutput:
        """Execute page attention using cache metadata.

        Args:
            query: Query tensor (used as pre-scaled queries).
            key: Key cache tensor (block-formatted).
            value: Value cache tensor (block-formatted).
            cache_metadata: Paged attention metadata (required).
            cache_view: Optional page cache reference to include in output.
            scale: Softmax scaling factor.
            use_metal: Whether to use the Metal kernel.
            threadgroup_size: Metal threadgroup size.
            **_: Additional keyword arguments (ignored).

        Returns:
            An ``AttentionOutput`` containing attention results and cache view.

        Raises:
            ValueError: If ``cache_metadata`` is None or incomplete.
        """
        if cache_metadata is None:
            raise ValueError("PageAttention requires cache_metadata.")
        if isinstance(cache_metadata, PageMetadata):
            if (
                cache_metadata.block_tables is None
                or cache_metadata.kv_lens is None
                or cache_metadata.block_size is None
            ):
                raise ValueError("PageMetadata must be resolved before page attention execution.")
            cache_metadata = PageAttnMetadata(
                block_tables=cache_metadata.block_tables,
                kv_lens=cache_metadata.kv_lens,
                query_start_loc=cache_metadata.query_start_loc,
                block_size=cache_metadata.block_size,
                sliding_window=cache_metadata.sliding_window,
                is_single_token_decode=cache_metadata.is_single_token_decode,
            )

        runtime_key_cache = key
        runtime_value_cache = value
        runtime_k_scale: mx.array | None = None
        runtime_v_scale: mx.array | None = None
        if cache_view is not None:
            page_key_cache = getattr(cache_view.cache, "page_key_cache", None)
            page_value_cache = getattr(cache_view.cache, "page_value_cache", None)
            if (
                page_key_cache is not None
                and page_value_cache is not None
                and bool(cache_metadata.is_single_token_decode)
            ):
                runtime_key_cache = page_key_cache
                runtime_value_cache = page_value_cache

        # FP8 path: pass per-block scale arrays to the Metal kernel
        # instead of dequantizing in Python.
        if (
            cache_view is not None
            and getattr(cache_view, "cache_dtype_is_fp8", False)
            and runtime_key_cache.dtype == mx.uint8
        ):
            runtime_k_scale = getattr(cache_view, "k_scales", None)
            runtime_v_scale = getattr(cache_view, "v_scales", None)
            if runtime_k_scale is None or runtime_v_scale is None:
                # Fallback: dequantize in Python if scales are missing
                runtime_key_cache, runtime_value_cache = cache_view.dequantize_kv_cache(
                    runtime_key_cache,
                    runtime_value_cache,
                    out_dtype=query.dtype,
                )
                runtime_k_scale = None
                runtime_v_scale = None

        outputs = paged_attention(
            query,
            runtime_key_cache,
            runtime_value_cache,
            cache_metadata.block_tables,
            cache_metadata.kv_lens,
            cache_metadata.query_start_loc,
            softmax_scale=scale,
            sliding_window=cache_metadata.sliding_window,
            use_metal=use_metal,
            single_token_decode=cache_metadata.is_single_token_decode,
            threadgroup_size=threadgroup_size,
            partition_size=self.partition_size,
            v2_min_context=self.v2_min_context,
            k_scale=runtime_k_scale,
            v_scale=runtime_v_scale,
        )
        return AttentionOutput(attention_outputs=outputs, cache_view=cache_view)


def page_attention(*args: tp.Any, **kwargs: tp.Any) -> mx.array:
    """Alias for the standalone paged-attention function."""
    return paged_attention(*args, **kwargs)


PageAttn = PageAttention

__all__ = (
    "PageAttention",
    "PageAttn",
    "PageAttnConfig",
    "PageAttnMetadata",
    "page_attention",
    "paged_attention",
)
