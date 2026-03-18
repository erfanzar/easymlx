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
import os
import typing as tp
from dataclasses import dataclass

import numpy as np
from mlx import core as mx

from easymlx.caching import PageCache, PageMetadata

from .._attention_outputs import AttentionOutput
from .._base_operation import BaseOperation, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, RequirementsBuilder


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Args:
        name: Name of the environment variable.
        default: Value returned when the variable is not set.

    Returns:
        True unless the variable is set to a falsy string
        (``"0"``, ``"false"``, ``"no"``, or ``"off"``).
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


_DEFAULT_USE_METAL = _env_flag("EASYMLX_UNIFIED_ATTENTION_USE_METAL", True)
_FORCE_REFERENCE = _env_flag("EASYMLX_UNIFIED_ATTENTION_FORCE_REFERENCE", False)


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
    """

    block_tables: mx.ArrayLike
    kv_lens: mx.ArrayLike
    query_start_loc: mx.ArrayLike
    block_size: int
    sliding_window: int | None = None


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
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be rank-2: [num_seqs, max_blocks_per_seq]")
    if kv_lens.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("kv_lens/query_start_loc must be rank-1")
    if query_start_loc.shape[0] != block_tables.shape[0] + 1:
        raise ValueError("query_start_loc must have length num_seqs + 1")


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
    uint elem = thread_position_in_grid.x;
    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    uint elems_per_token = (uint)(num_q_heads * head_dim);
    uint token_idx = elem / elems_per_token;
    if (token_idx >= (uint)total_tokens) {
        return;
    }
    uint rem = elem - token_idx * elems_per_token;
    int q_head = (int)(rem / head_dim);
    int dim_idx = (int)(rem - (uint)(q_head * head_dim));

    int num_kv_heads = key_cache_shape[2];
    if (num_kv_heads <= 0) {
        out[elem] = (T)0;
        return;
    }
    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        out[elem] = (T)0;
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
        out[elem] = (T)0;
        return;
    }
    int q_pos = (int)token_idx - q_start;

    int seq_len = kv_lens[seq_idx];
    if (seq_len <= 0) {
        out[elem] = (T)0;
        return;
    }

    int context_len = seq_len - q_len;
    int max_k = context_len + q_pos;
    if (max_k >= seq_len) {
        max_k = seq_len - 1;
    }
    if (max_k < 0) {
        out[elem] = (T)0;
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
    int q_base = ((int)token_idx * num_q_heads + q_head) * head_dim;

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (int kpos = start_k; kpos <= max_k; ++kpos) {
        int block = block_tables[seq_idx * max_blocks_per_seq + (kpos / block_size)];
        if (block < 0) {
            continue;
        }
        int offset = kpos - (kpos / block_size) * block_size;
        int kv_base = ((block * block_size + offset) * num_kv_heads + kv_head) * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float qv = (float)queries[q_base + d];
            float kv = (float)key_cache[kv_base + d];
            score += qv * kv;
        }

        float m_new = fmax(m, score);
        float p = exp(score - m_new);
        float alpha = exp(m - m_new);
        float v_val = (float)value_cache[kv_base + dim_idx];
        acc = acc * alpha + p * v_val;
        l = l * alpha + p;
        m = m_new;
    }

    float out_val = (l <= 0.0f) ? 0.0f : (acc / l);
    out[elem] = (T)out_val;
"""


_METAL_KERNEL: tp.Any | None = None


def _get_metal_kernel():
    """Lazily compile and cache the Metal paged attention kernel.

    Returns:
        The compiled ``mx.fast.metal_kernel`` object.
    """
    global _METAL_KERNEL
    if _METAL_KERNEL is None:
        _METAL_KERNEL = mx.fast.metal_kernel(
            name="unified_attention_paged",
            input_names=[
                "queries",
                "key_cache",
                "value_cache",
                "block_tables",
                "kv_lens",
                "query_start_loc",
                "sliding_window_ptr",
            ],
            output_names=["out"],
            source=_METAL_SOURCE,
            header=_METAL_HEADER,
        )
    return _METAL_KERNEL


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
    threadgroup_size: int = 256,
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
        threadgroup_size: Metal threadgroup size for kernel dispatch.

    Returns:
        Output tensor of the same shape as ``queries``.
    """
    _validate_inputs(queries, key_cache, value_cache, block_tables, kv_lens, query_start_loc)

    head_dim = int(queries.shape[-1])
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if sliding_window is None:
        sliding_window = 0

    if use_metal is None:
        use_metal = _DEFAULT_USE_METAL and not _FORCE_REFERENCE

    if use_metal and not _FORCE_REFERENCE:
        try:
            kernel = _get_metal_kernel()
            q_scaled = queries * float(softmax_scale)
            block_tables = mx.array(block_tables, dtype=mx.int32)
            kv_lens = mx.array(kv_lens, dtype=mx.int32)
            query_start_loc = mx.array(query_start_loc, dtype=mx.int32)
            sliding_window_ptr = mx.array([int(sliding_window)], dtype=mx.int32)

            total_tokens = int(q_scaled.shape[0])
            num_q_heads = int(q_scaled.shape[1])
            head_dim = int(q_scaled.shape[2])
            total_elems = int(total_tokens * num_q_heads * head_dim)

            outputs = kernel(
                inputs=[q_scaled, key_cache, value_cache, block_tables, kv_lens, query_start_loc, sliding_window_ptr],
                template=[("T", q_scaled.dtype)],
                output_shapes=[q_scaled.shape],
                output_dtypes=[q_scaled.dtype],
                grid=(total_elems, 1, 1),
                threadgroup=(int(threadgroup_size), 1, 1),
            )
            return outputs[0]
        except Exception:
            if not _env_flag("EASYMLX_UNIFIED_ATTENTION_ALLOW_FALLBACK", True):
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

    Registered under ``"unified_attention"``, ``"paged_attention"``, and
    ``"page_attention"``. Requires ``PageCache`` and associated metadata.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names for this operation.

        Returns:
            A tuple of name aliases.
        """
        return ("unified_attention", "paged_attention", "page_attention")

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
            .require_cache(PageCache)
            .build()
        )

    def forward_native(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        cache_metadata: UnifiedAttnMetadata | PageMetadata | None = None,
        cache_view: PageCache | None = None,
        scale: float | None = None,
        use_metal: bool = _DEFAULT_USE_METAL,
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
            use_metal: Whether to use the Metal kernel.
            threadgroup_size: Metal threadgroup size.
            **_: Additional keyword arguments (ignored).

        Returns:
            An ``AttentionOutput`` containing attention results and cache view.

        Raises:
            ValueError: If ``cache_metadata`` is None or incomplete.
        """
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
            )

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
            threadgroup_size=threadgroup_size,
        )
        return AttentionOutput(attention_outputs=outputs, cache_view=cache_view)


__all__ = ("UnifiedAttention", "UnifiedAttnConfig", "UnifiedAttnMetadata", "paged_attention")
