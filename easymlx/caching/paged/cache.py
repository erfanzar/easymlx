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

"""Paged KV cache helpers for serving runtimes.

All storage uses ``mx.array`` — no numpy dependency for cache state.
"""

from __future__ import annotations

import logging
import math
import typing as tp
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)


_USE_METAL_KERNELS: bool = True


_RESHAPE_AND_CACHE_SOURCE = r"""
    uint gid = thread_position_in_grid.x;
    uint tg_total = threads_per_threadgroup.x;

    int key_cache_elems = key_cache_in_shape[0];
    int val_cache_elems = value_cache_in_shape[0];
    int num_tokens = slot_mapping_shape[0];

    int key_stride = params[0];
    int value_stride = params[1];
    int num_heads = params[2];
    int head_size = params[3];
    int block_size = params[4];
    int x_val = params[5];

    // Phase 1: copy entire cache from input to output
    for (int i = (int)gid; i < key_cache_elems; i += (int)(tg_total * threadgroup_position_in_grid.x + tg_total)) {
        key_cache_out[i] = key_cache_in[i];
    }
    for (int i = (int)gid; i < val_cache_elems; i += (int)(tg_total * threadgroup_position_in_grid.x + tg_total)) {
        value_cache_out[i] = value_cache_in[i];
    }

    // Phase 2: scatter-write new tokens
    int token_idx = (int)threadgroup_position_in_grid.y;
    if (token_idx >= num_tokens) return;

    int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;

    int block_idx = slot_idx / block_size;
    int block_offset = slot_idx % block_size;
    int n = num_heads * head_size;
    uint tid = thread_position_in_threadgroup.x;

    for (int i = (int)tid; i < n; i += (int)tg_total) {
        int src_key_idx = token_idx * key_stride + i;
        int src_value_idx = token_idx * value_stride + i;

        int head_idx = i / head_size;
        int head_offset = i % head_size;
        int x_idx = head_offset / x_val;
        int x_offset = head_offset % x_val;

        int tgt_key_idx =
            block_idx * num_heads * (head_size / x_val) * block_size * x_val +
            head_idx * (head_size / x_val) * block_size * x_val +
            x_idx * block_size * x_val +
            block_offset * x_val + x_offset;
        int tgt_value_idx =
            block_idx * num_heads * head_size * block_size +
            head_idx * head_size * block_size +
            head_offset * block_size + block_offset;

        key_cache_out[tgt_key_idx] = key[src_key_idx];
        value_cache_out[tgt_value_idx] = value[src_value_idx];
    }
"""

_COPY_BLOCKS_SOURCE = r"""
    uint gid = thread_position_in_grid.x;
    uint tg_size = threads_per_threadgroup.x;

    int key_total = key_cache_in_shape[0];
    int val_total = value_cache_in_shape[0];
    int num_pairs = block_mapping_shape[0] / 2;

    // Phase 1: copy entire cache
    for (int i = (int)gid; i < key_total; i += (int)(tg_size * (threadgroup_position_in_grid.x + 1))) {
        key_cache_out[i] = key_cache_in[i];
    }
    for (int i = (int)gid; i < val_total; i += (int)(tg_size * (threadgroup_position_in_grid.x + 1))) {
        value_cache_out[i] = value_cache_in[i];
    }

    // Phase 2: copy specified block pairs
    int pair_idx = (int)threadgroup_position_in_grid.y;
    if (pair_idx >= num_pairs) return;

    int numel_per_block_key = params[0];
    int numel_per_block_value = params[1];

    int src_block = block_mapping[2 * pair_idx];
    int dst_block = block_mapping[2 * pair_idx + 1];

    uint tid = thread_position_in_threadgroup.x;
    int src_key_off = src_block * numel_per_block_key;
    int dst_key_off = dst_block * numel_per_block_key;
    for (int i = (int)tid; i < numel_per_block_key; i += (int)tg_size) {
        key_cache_out[dst_key_off + i] = key_cache_in[src_key_off + i];
    }

    int src_val_off = src_block * numel_per_block_value;
    int dst_val_off = dst_block * numel_per_block_value;
    for (int i = (int)tid; i < numel_per_block_value; i += (int)tg_size) {
        value_cache_out[dst_val_off + i] = value_cache_in[src_val_off + i];
    }
"""

_GATHER_KV_CACHE_SOURCE = r"""
    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;

    int num_tokens_total = params[0];
    int num_seqs = params[1];
    int block_size = params[2];
    int block_table_stride = params[3];
    int num_kv_heads = params[4];
    int head_size = params[5];
    int x_val = params[6];

    int token_id = (int)gid;
    if (token_id >= num_tokens_total) return;

    // Binary search cu_seq_lens to find batch_id
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= token_id) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int batch_id = lo;
    int batch_offset = token_id - cu_seq_lens[batch_id];
    int block_table_id = batch_offset / block_size;
    int slot = batch_offset % block_size;
    int block_id = block_table[batch_id * block_table_stride + block_table_id];

    int n = num_kv_heads * head_size;
    long out_base = (long)token_id * num_kv_heads * head_size;

    long k_block_stride = (long)num_kv_heads * (head_size / x_val) * block_size * x_val;
    long k_head_stride = (long)(head_size / x_val) * block_size * x_val;
    long v_block_stride = (long)num_kv_heads * head_size * block_size;
    long v_head_stride = (long)head_size * block_size;

    for (int i = (int)tid; i < n; i += (int)tg_size) {
        int head_idx = i / head_size;
        int d = i % head_size;

        int x_idx = d / x_val;
        int x_offset = d % x_val;
        long k_src_idx = (long)block_id * k_block_stride +
                         head_idx * k_head_stride +
                         x_idx * block_size * x_val +
                         slot * x_val + x_offset;

        long v_src_idx = (long)block_id * v_block_stride +
                         head_idx * v_head_stride +
                         d * block_size + slot;

        k_out[out_base + i] = (T)key_cache[k_src_idx];
        v_out[out_base + i] = (T)value_cache[v_src_idx];
    }
"""

_KV_SCALE_UPDATE_SOURCE = r"""
    uint gid = thread_position_in_grid.x;
    uint total_threads = threads_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;

    long num_elements = (long)num_elems_ptr[0];

    // Per-thread local maxima
    float local_max_k = 0.0f;
    float local_max_v = 0.0f;

    // Strided loop covering entire array
    for (long idx = (long)gid; idx < num_elements; idx += (long)total_threads) {
        float avk = abs((float)k[idx]);
        float avv = abs((float)v[idx]);
        local_max_k = fmax(local_max_k, avk);
        local_max_v = fmax(local_max_v, avv);
    }

    // Parallel reduction in threadgroup to find block maxima
    threadgroup float shared_k[256];
    threadgroup float shared_v[256];
    shared_k[tid] = local_max_k;
    shared_v[tid] = local_max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_k[tid] = fmax(shared_k[tid], shared_k[tid + s]);
            shared_v[tid] = fmax(shared_v[tid], shared_v[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the final scales
    if (tid == 0) {
        float div_const = 240.0f;
        float candidate_k = shared_k[0] / div_const;
        float candidate_v = shared_v[0] / div_const;
        // Use max with existing value for multi-threadgroup accumulation
        k_scale_out[threadgroup_position_in_grid.x] = candidate_k;
        v_scale_out[threadgroup_position_in_grid.x] = candidate_v;
    }
"""


def _get_metal_header() -> str:
    from easymlx.operations.kernels._metal_header import PAGED_ATTENTION_HEADER

    return PAGED_ATTENTION_HEADER


_RESHAPE_AND_CACHE_KERNEL = mx.fast.metal_kernel(
    name="reshape_and_cache",
    input_names=["key", "value", "key_cache_in", "value_cache_in", "slot_mapping", "params"],
    output_names=["key_cache_out", "value_cache_out"],
    source=_RESHAPE_AND_CACHE_SOURCE,
    header=_get_metal_header(),
)

_COPY_BLOCKS_KERNEL = mx.fast.metal_kernel(
    name="copy_blocks",
    input_names=["key_cache_in", "value_cache_in", "block_mapping", "params"],
    output_names=["key_cache_out", "value_cache_out"],
    source=_COPY_BLOCKS_SOURCE,
    header=_get_metal_header(),
)

_GATHER_KV_CACHE_KERNEL = mx.fast.metal_kernel(
    name="gather_kv_cache",
    input_names=["key_cache", "value_cache", "block_table", "cu_seq_lens", "params"],
    output_names=["k_out", "v_out"],
    source=_GATHER_KV_CACHE_SOURCE,
    header=_get_metal_header(),
)

_KV_SCALE_UPDATE_KERNEL = mx.fast.metal_kernel(
    name="kv_scale_update",
    input_names=["k", "v", "num_elems_ptr"],
    output_names=["k_scale_out", "v_scale_out"],
    source=_KV_SCALE_UPDATE_SOURCE,
    header=_get_metal_header(),
)


def _reshape_and_cache_metal(
    key: mx.array,
    value: mx.array,
    page_key_cache: mx.array,
    page_value_cache: mx.array,
    slot_mapping: mx.array,
    *,
    num_kv_heads: int,
    head_dim: int,
    page_vec_size: int,
    block_size: int,
) -> tuple[mx.array, mx.array]:
    """Reshape-and-cache using the mistral.rs Metal kernel.

    Copies the existing cache and scatter-writes new tokens in a single
    Metal dispatch.

    Returns (updated_page_key_cache, updated_page_value_cache).
    """
    num_tokens = int(key.shape[0])
    if num_tokens == 0:
        return page_key_cache, page_value_cache

    kernel = _RESHAPE_AND_CACHE_KERNEL
    key_stride = num_kv_heads * head_dim
    value_stride = num_kv_heads * head_dim
    params = mx.array(
        [key_stride, value_stride, num_kv_heads, head_dim, block_size, page_vec_size],
        dtype=mx.int32,
    )

    key_flat = key.reshape(num_tokens, key_stride)
    val_flat = value.reshape(num_tokens, value_stride)
    slot_mapping = slot_mapping.astype(mx.int32)

    tg_size = min(256, max(32, num_kv_heads * head_dim))
    results = kernel(
        inputs=[
            key_flat,
            val_flat,
            page_key_cache.reshape(-1),
            page_value_cache.reshape(-1),
            slot_mapping,
            params,
        ],
        output_shapes=[page_key_cache.reshape(-1).shape, page_value_cache.reshape(-1).shape],
        output_dtypes=[page_key_cache.dtype, page_value_cache.dtype],
        grid=(tg_size, num_tokens, 1),
        threadgroup=(tg_size, 1, 1),
    )
    return results[0].reshape(page_key_cache.shape), results[1].reshape(page_value_cache.shape)


def _copy_blocks(
    key_cache: mx.array,
    value_cache: mx.array,
    block_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Copy cache blocks using the Metal kernel.

    Returns (new_key_cache, new_value_cache).
    """
    if block_mapping.shape[0] == 0:
        return key_cache, value_cache

    num_pairs = int(block_mapping.shape[0])
    block_mapping_flat = block_mapping.reshape(-1).astype(mx.int32)

    kernel = _COPY_BLOCKS_KERNEL
    numel_key = 1
    for d in key_cache.shape[1:]:
        numel_key *= int(d)
    numel_val = 1
    for d in value_cache.shape[1:]:
        numel_val *= int(d)

    params = mx.array([numel_key, numel_val], dtype=mx.int32)
    tg_size = min(256, max(32, max(numel_key, numel_val)))
    results = kernel(
        inputs=[
            key_cache.reshape(-1),
            value_cache.reshape(-1),
            block_mapping_flat,
            params,
        ],
        output_shapes=[key_cache.reshape(-1).shape, value_cache.reshape(-1).shape],
        output_dtypes=[key_cache.dtype, value_cache.dtype],
        grid=(tg_size, num_pairs, 1),
        threadgroup=(tg_size, 1, 1),
    )
    return results[0].reshape(key_cache.shape), results[1].reshape(value_cache.shape)


def _try_gather_kv_cache_metal(
    key_cache: mx.array,
    value_cache: mx.array,
    block_table: mx.array,
    cu_seq_lens: mx.array,
    *,
    num_tokens: int,
    num_seqs: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    x_val: int,
    out_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array] | None:
    """Attempt to run the gather_kv_cache Metal kernel.

    Returns (k_out, v_out) on success, None if Metal is unavailable.
    """
    if not _USE_METAL_KERNELS:
        return None

    if num_tokens == 0:
        return (
            mx.zeros((0, num_kv_heads, head_size), dtype=out_dtype),
            mx.zeros((0, num_kv_heads, head_size), dtype=out_dtype),
        )

    block_table_stride = int(block_table.shape[1])
    params = mx.array(
        [num_tokens, num_seqs, block_size, block_table_stride, num_kv_heads, head_size, x_val],
        dtype=mx.int32,
    )

    kernel = _GATHER_KV_CACHE_KERNEL
    tg_size = min(256, max(32, num_kv_heads * head_size))
    outputs = kernel(
        inputs=[key_cache, value_cache, block_table.astype(mx.int32), cu_seq_lens.astype(mx.int32), params],
        template=[("T", out_dtype)],
        output_shapes=[
            (num_tokens, num_kv_heads, head_size),
            (num_tokens, num_kv_heads, head_size),
        ],
        output_dtypes=[out_dtype, out_dtype],
        grid=(num_tokens, 1, 1),
        threadgroup=(tg_size, 1, 1),
    )
    return outputs[0], outputs[1]


def _try_kv_scale_update_metal(
    k: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array] | None:
    """Compute FP8 scaling factors for K and V using the Metal kernel.

    Returns (k_scale, v_scale) as scalar float32 arrays on success,
    None if Metal is unavailable.
    """
    if not _USE_METAL_KERNELS:
        return None

    num_elements = 1
    for d in k.shape:
        num_elements *= int(d)
    if num_elements == 0:
        return mx.array([0.0], dtype=mx.float32), mx.array([0.0], dtype=mx.float32)

    num_elems_ptr = mx.array([num_elements], dtype=mx.int32)
    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)

    tg_size = min(256, max(32, num_elements))
    num_tgs = 1

    kernel = _KV_SCALE_UPDATE_KERNEL
    outputs = kernel(
        inputs=[k_flat, v_flat, num_elems_ptr],
        template=[("T", k.dtype)],
        output_shapes=[(num_tgs,), (num_tgs,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(tg_size * num_tgs, 1, 1),
        threadgroup=(tg_size, 1, 1),
    )
    k_scale = mx.max(outputs[0])
    v_scale = mx.max(outputs[1])
    return k_scale, v_scale


def _fp8_quantize(x: mx.array, scale: mx.array) -> mx.array:
    """Quantize a float tensor to FP8 E4M3 using a pre-computed scale.

    Args:
        x: Input tensor in float16/bfloat16/float32.
        scale: Scalar scale factor (max_abs / 240).

    Returns:
        Quantized uint8 tensor in E4M3 format.
    """
    safe_scale = mx.maximum(scale, mx.array(1e-12, dtype=mx.float32))
    return mx.to_fp8(x / safe_scale)


def _fp8_dequantize(
    x: mx.array,
    scale: mx.array,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Dequantize a FP8 E4M3 uint8 tensor back to float.

    Args:
        x: Quantized uint8 tensor.
        scale: Scalar scale factor used during quantization.
        dtype: Target output dtype.

    Returns:
        Dequantized float tensor.
    """
    return mx.from_fp8(x, dtype=dtype) * scale


class PageCacheView:
    """Block-table based KV cache backed by ``mx.array``.

    Attributes
    ----------
    key_cache, value_cache : mx.array
        Shape ``[num_blocks, block_size, num_kv_heads, head_dim]``.
    block_tables : mx.array
        Shape ``[num_seqs, blocks_per_seq]``, dtype ``int32``.
    kv_lens : mx.array
        Shape ``[num_seqs]``, dtype ``int32`` — current KV length per sequence.
    page_key_cache, page_value_cache : mx.array | None
        Optional packed views used by the dedicated ``PageAttention`` runtime.
    block_size, num_kv_heads, head_dim : int
        Scalar configuration.
    """

    __slots__ = (
        "block_size",
        "block_tables",
        "cache_dtype_is_fp8",
        "head_dim",
        "k_scales",
        "key_cache",
        "kv_lens",
        "num_kv_heads",
        "page_key_cache",
        "page_value_cache",
        "page_vec_size",
        "v_scales",
        "value_cache",
    )

    def __init__(
        self,
        *,
        key_cache: mx.array,
        value_cache: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        page_key_cache: mx.array | None = None,
        page_value_cache: mx.array | None = None,
        page_vec_size: int | None = None,
        cache_dtype_is_fp8: bool = False,
        k_scales: mx.array | None = None,
        v_scales: mx.array | None = None,
    ) -> None:
        """Initialize a paged KV cache from pre-allocated arrays.

        Args:
            key_cache: Key storage of shape
                ``[num_blocks, block_size, num_kv_heads, head_dim]``.
            value_cache: Value storage with the same shape as *key_cache*.
            block_tables: Block index mapping of shape
                ``[num_seqs, blocks_per_seq]``, dtype ``int32``.
            kv_lens: Current KV length per sequence, shape ``[num_seqs]``,
                dtype ``int32``.
            block_size: Number of tokens stored in each block.
            num_kv_heads: Number of key/value attention heads.
            head_dim: Dimensionality of each attention head.
            cache_dtype_is_fp8: If ``True``, key/value caches store FP8
                E4M3 data as ``uint8`` and require per-block scale factors.
            k_scales: Per-block key scale factors of shape
                ``[num_blocks]``, dtype ``float32``. Required when
                ``cache_dtype_is_fp8`` is ``True``.
            v_scales: Per-block value scale factors of shape
                ``[num_blocks]``, dtype ``float32``. Required when
                ``cache_dtype_is_fp8`` is ``True``.

        Raises:
            ValueError: If array shapes are inconsistent with the provided
                scalar parameters.
        """
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = block_tables
        self.kv_lens = kv_lens
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_key_cache = page_key_cache
        self.page_value_cache = page_value_cache
        self.page_vec_size = int(page_vec_size) if page_vec_size is not None else 0
        self.cache_dtype_is_fp8 = bool(cache_dtype_is_fp8)
        self.k_scales = k_scales
        self.v_scales = v_scales
        self._validate_shape()

    @property
    def cache(self) -> PageCacheView:
        """Return self for compatibility with code that accesses ``view.cache``.

        Previously, a wrapper class exposed the view as ``.cache``.  Now that
        models receive views directly, this property allows operation kernels
        to use ``cache_view.cache`` uniformly.
        """
        return self

    @property
    def blocks_per_seq(self) -> int:
        """Return the number of blocks allocated per sequence."""
        return int(self.block_tables.shape[1])

    @property
    def num_seqs(self) -> int:
        """Return the number of sequence slots in this cache."""
        return int(self.block_tables.shape[0])

    @property
    def max_seq_len(self) -> int:
        """Return the maximum sequence length this cache can hold."""
        return int(self.blocks_per_seq * self.block_size)

    def _validate_shape(self) -> None:
        """Validate internal array shapes for consistency.

        Raises:
            ValueError: If any array dimensions do not match the expected
                configuration.
        """
        if self.block_tables.ndim != 2:
            raise ValueError("block_tables must be rank-2: [num_seqs, blocks_per_seq]")
        if self.kv_lens.ndim != 1:
            raise ValueError("kv_lens must be rank-1: [num_seqs]")
        if self.key_cache.shape != self.value_cache.shape:
            raise ValueError("key_cache and value_cache must have the same shape.")
        if self.key_cache.ndim != 4:
            raise ValueError("key_cache/value_cache must be rank-4.")

        num_seqs, blocks_per_seq = self.block_tables.shape
        if self.kv_lens.shape[0] != num_seqs:
            raise ValueError("kv_lens length must match block_tables first dimension.")
        if self.key_cache.shape[0] != blocks_per_seq * num_seqs:
            raise ValueError("key_cache block count does not match configured sequences.")
        if self.key_cache.shape[1] != self.block_size:
            raise ValueError("key_cache block size does not match block_size.")
        if self.key_cache.shape[2] != self.num_kv_heads:
            raise ValueError("key_cache num_kv_heads does not match configured value.")
        if self.key_cache.shape[3] != self.head_dim:
            raise ValueError("key_cache head_dim does not match configured value.")
        if (self.page_key_cache is None) != (self.page_value_cache is None):
            raise ValueError("page_key_cache and page_value_cache must either both be set or both be None.")
        if self.page_key_cache is not None:
            if self.page_vec_size <= 0:
                raise ValueError("page_vec_size must be positive when packed page caches are provided.")
            if self.head_dim % self.page_vec_size != 0:
                raise ValueError("head_dim must be divisible by page_vec_size.")
            num_vecs = self.head_dim // self.page_vec_size
            expected_k_shape = (
                self.key_cache.shape[0],
                self.num_kv_heads,
                num_vecs,
                self.block_size,
                self.page_vec_size,
            )
            expected_v_shape = (
                self.key_cache.shape[0],
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )
            if tuple(self.page_key_cache.shape) != expected_k_shape:
                raise ValueError("page_key_cache shape does not match packed PageAttention layout.")
            if tuple(self.page_value_cache.shape) != expected_v_shape:
                raise ValueError("page_value_cache shape does not match packed PageAttention layout.")

    @staticmethod
    def _page_vec_size(dtype: mx.Dtype, head_dim: int = 128) -> int:
        if dtype == mx.float32:
            vs = 4
        elif dtype in {mx.float16, mx.bfloat16}:
            vs = 8
        else:
            vs = 1
        while vs > 1 and head_dim % vs != 0:
            vs //= 2
        return max(vs, 1)

    def can_append(self, seq_idx: int, num_new_tokens: int) -> bool:
        """Check whether *num_new_tokens* can be appended to a sequence.

        Args:
            seq_idx: Zero-based sequence slot index.
            num_new_tokens: Number of new tokens to append.

        Returns:
            ``True`` if the sequence has enough remaining capacity.

        Raises:
            ValueError: If *num_new_tokens* is negative.
            IndexError: If *seq_idx* is out of range.
        """
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative.")
        if seq_idx < 0 or seq_idx >= self.num_seqs:
            raise IndexError(f"Invalid seq_idx {seq_idx}")
        return int(self.kv_lens[seq_idx].item()) + int(num_new_tokens) <= self.max_seq_len

    @classmethod
    def init(
        cls,
        *,
        num_seqs: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: mx.Dtype | type | str = mx.float16,
        cache_dtype: str | None = None,
    ) -> PageCacheView:
        """Allocate a new zero-initialized ragged pages cache view.

        Block tables are set up with a simple contiguous layout where each
        sequence owns ``ceil(max_seq_len / block_size)`` consecutive blocks.

        Args:
            num_seqs: Number of sequence slots to allocate.
            max_seq_len: Maximum sequence length each slot can hold.
            num_kv_heads: Number of key/value attention heads.
            head_dim: Dimensionality of each attention head.
            block_size: Number of tokens per cache block.
            dtype: Data type for key/value storage (used when
                ``cache_dtype`` is ``None`` or ``"auto"``).
            cache_dtype: Override for cache storage dtype. ``"fp8"``
                allocates uint8 caches with per-block float32 scales
                for FP8 E4M3 quantization. ``None`` or ``"auto"``
                falls through to *dtype*.

        Returns:
            A freshly allocated :class:`PageCacheView`.
        """
        _DTYPE_MAP = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        if not isinstance(dtype, mx.Dtype):
            import numpy as np

            if dtype is np.float16 or str(dtype) == "float16":
                dtype = mx.float16
            elif dtype is np.float32 or str(dtype) == "float32":
                dtype = mx.float32
            elif str(dtype) in _DTYPE_MAP:
                dtype = _DTYPE_MAP[str(dtype)]
            else:
                dtype = mx.float16

        is_fp8 = str(cache_dtype or "auto").lower() == "fp8"

        if num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = blocks_per_seq * num_seqs

        rows = []
        for seq_idx in range(num_seqs):
            start = seq_idx * blocks_per_seq
            rows.append(mx.arange(start, start + blocks_per_seq, dtype=mx.int32))
        block_tables = mx.stack(rows)

        storage_dtype = mx.uint8 if is_fp8 else dtype

        key_cache = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=storage_dtype)
        value_cache = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=storage_dtype)

        # FP8 per-block scale factors (one scalar per block)
        k_scales: mx.array | None = None
        v_scales: mx.array | None = None
        if is_fp8:
            k_scales = mx.zeros((num_blocks,), dtype=mx.float32)
            v_scales = mx.zeros((num_blocks,), dtype=mx.float32)

        # Page caches (packed layout for Metal PageAttention kernel) always
        # use the model's float dtype, not uint8.  The standard 4D caches
        # store FP8 data as uint8, and the Metal kernels have native FP8
        # dequantization support using per-block k_scales/v_scales.
        page_vec_size = 0
        page_key_cache = None
        page_value_cache = None
        kv_lens = mx.zeros((num_seqs,), dtype=mx.int32)

        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            kv_lens=kv_lens,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_key_cache=page_key_cache,
            page_value_cache=page_value_cache,
            page_vec_size=page_vec_size,
            cache_dtype_is_fp8=is_fp8,
            k_scales=k_scales,
            v_scales=v_scales,
        )

    allocate = init

    def append(self, seq_idx: int, key: mx.array, value: mx.array) -> None:
        """Append tokens to a sequence's KV cache.

        Parameters
        ----------
        seq_idx : int
            Which sequence slot to append to.
        key, value : mx.array
            Shape ``[num_tokens, num_kv_heads, head_dim]``.
        """
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 3:
            raise ValueError("key/value must be rank-3: [num_tokens, num_kv_heads, head_dim]")
        if key.shape[1] != self.num_kv_heads or key.shape[2] != self.head_dim:
            raise ValueError("key/value head dims do not match cache")
        if seq_idx < 0 or seq_idx >= self.num_seqs:
            raise IndexError(f"Invalid seq_idx {seq_idx}")

        start = int(self.kv_lens[seq_idx].item())
        num_tokens = int(key.shape[0])
        if start + num_tokens > self.max_seq_len:
            raise ValueError("Cannot append tokens: sequence length would exceed cache capacity")

        for i in range(num_tokens):
            pos = start + i
            block = int(self.block_tables[seq_idx, pos // self.block_size].item())
            if block < 0:
                continue
            offset = pos % self.block_size
            self.key_cache[block, offset] = key[i]
            self.value_cache[block, offset] = value[i]
            if self.page_key_cache is not None and self.page_value_cache is not None:
                packed_key = key[i].reshape(self.num_kv_heads, self.head_dim // self.page_vec_size, self.page_vec_size)
                self.page_key_cache[block, :, :, offset, :] = packed_key
                self.page_value_cache[block, :, :, offset] = value[i]
        self.kv_lens[seq_idx] = start + num_tokens

    def reset(self, seq_idx: int) -> None:
        """Zero out all blocks for a single sequence and reset its KV length.

        Args:
            seq_idx: Zero-based sequence slot index.

        Raises:
            IndexError: If *seq_idx* is out of range.
        """
        if seq_idx < 0 or seq_idx >= self.num_seqs:
            raise IndexError(f"Invalid seq_idx {seq_idx}")
        block_ids = self.block_tables[seq_idx]
        for i in range(block_ids.shape[0]):
            bid = int(block_ids[i].item())
            if bid >= 0:
                self.key_cache["bid"] = mx.zeros(
                    (self.block_size, self.num_kv_heads, self.head_dim),
                    dtype=self.key_cache.dtype,
                )
                self.value_cache["bid"] = mx.zeros(
                    (self.block_size, self.num_kv_heads, self.head_dim),
                    dtype=self.value_cache.dtype,
                )
                if self.page_key_cache is not None and self.page_value_cache is not None:
                    self.page_key_cache["bid"] = mx.zeros(
                        (self.num_kv_heads, self.head_dim // self.page_vec_size, self.block_size, self.page_vec_size),
                        dtype=self.page_key_cache.dtype,
                    )
                    self.page_value_cache["bid"] = mx.zeros(
                        (self.num_kv_heads, self.head_dim, self.block_size),
                        dtype=self.page_value_cache.dtype,
                    )
        self.kv_lens[seq_idx] = 0

    def reset_batch(self, seq_indices: Sequence["int"]) -> None:
        """Reset multiple sequence slots in batch.

        Args:
            seq_indices: Sequence slot indices to reset.
        """
        for seq_idx in seq_indices:
            self.reset(int(seq_idx))

    def clear(self, seq_idx: int | None = None) -> None:
        """Clear the cache, optionally for a single sequence only.

        Args:
            seq_idx: If provided, only clear this sequence slot. If ``None``,
                zero out the entire cache (all blocks and all KV lengths).
        """
        if seq_idx is None:
            self.key_cache = mx.zeros_like(self.key_cache)
            self.value_cache = mx.zeros_like(self.value_cache)
            if self.page_key_cache is not None:
                self.page_key_cache = mx.zeros_like(self.page_key_cache)
            if self.page_value_cache is not None:
                self.page_value_cache = mx.zeros_like(self.page_value_cache)
            self.kv_lens = mx.zeros_like(self.kv_lens)
            return
        self.reset(int(seq_idx))

    def clear_many(self, seq_indices: Iterable["int"]) -> None:
        """Clear multiple sequence slots.

        Args:
            seq_indices: Iterable of sequence slot indices to clear.
        """
        for seq_idx in seq_indices:
            self.reset(int(seq_idx))

    def copy_blocks(self, block_mapping: mx.array) -> None:
        """Copy cache blocks for Copy-on-Write (CoW) operations.

        Uses a Metal kernel when available, with a Python fallback.

        Args:
            block_mapping: An ``mx.array`` of shape ``[num_pairs, 2]`` with
                dtype ``int32`` where each row is ``(src_block, dst_block)``.
        """
        if block_mapping.ndim != 2 or block_mapping.shape[1] != 2:
            raise ValueError("block_mapping must be rank-2 with shape [num_pairs, 2]")
        num_pairs = int(block_mapping.shape[0])
        if num_pairs == 0:
            return

        self.key_cache, self.value_cache = _copy_blocks(
            self.key_cache,
            self.value_cache,
            block_mapping,
        )

        if self.page_key_cache is not None and self.page_value_cache is not None:
            page_vec_size = int(self.page_vec_size or 0)
            if page_vec_size > 0:
                self.page_key_cache, self.page_value_cache = _copy_blocks(
                    self.page_key_cache,
                    self.page_value_cache,
                    block_mapping,
                )

    def gather_kv_cache(
        self,
        block_table: mx.array,
        cu_seq_lens: mx.array,
        *,
        out_dtype: mx.Dtype | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Gather contiguous K/V tensors from paged cache blocks.

        Reads scattered page data and produces dense tensors suitable for
        standard (non-paged) attention. Uses the packed page cache layout
        when available, falling back to the standard layout.

        Args:
            block_table: Block table of shape ``[num_seqs, max_blocks_per_seq]``,
                dtype ``int32``.
            cu_seq_lens: Cumulative sequence lengths of shape ``[num_seqs + 1]``,
                dtype ``int32``.
            out_dtype: Output data type. Defaults to the cache dtype.

        Returns:
            A tuple ``(k_out, v_out)`` each of shape
            ``[total_tokens, num_kv_heads, head_dim]``.
        """
        if out_dtype is None:
            out_dtype = self.key_cache.dtype

        cu_np = cu_seq_lens.tolist() if isinstance(cu_seq_lens, mx.array) else list(cu_seq_lens)
        num_seqs = len(cu_np) - 1
        total_tokens = int(cu_np[-1])

        page_vec_size = int(self.page_vec_size or 0)

        if self.page_key_cache is not None and self.page_value_cache is not None and page_vec_size > 0:
            result = _try_gather_kv_cache_metal(
                self.page_key_cache,
                self.page_value_cache,
                block_table,
                cu_seq_lens,
                num_tokens=total_tokens,
                num_seqs=num_seqs,
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                x_val=page_vec_size,
                out_dtype=out_dtype,
            )
            if result is not None:
                return result

        raise RuntimeError(
            "gather_kv_cache Metal kernel failed and no fallback is available. "
            "Ensure packed page caches (page_key_cache/page_value_cache) are allocated."
        )

    def compute_kv_scales(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array] | None:
        """Compute FP8 scaling factors for key/value data.

        Uses a Metal kernel for efficient parallel reduction when available.
        This is needed when writing to an FP8-quantized KV cache to compute
        the appropriate scale factors (max_abs / 240).

        Args:
            keys: Key tensor to compute scale for.
            values: Value tensor to compute scale for.

        Returns:
            A tuple ``(k_scale, v_scale)`` of scalar float32 arrays, or
            ``None`` if FP8 scaling is not applicable (e.g. non-FP8 cache).
        """
        return _try_kv_scale_update_metal(keys, values)

    def dequantize_kv_cache(
        self,
        key_cache: mx.array,
        value_cache: mx.array,
        *,
        block_tables: mx.array | None = None,
        out_dtype: mx.Dtype = mx.float16,
    ) -> tuple[mx.array, mx.array]:
        """Dequantize FP8 key/value caches for attention computation.

        When ``cache_dtype_is_fp8`` is ``False``, returns the inputs
        unchanged. Otherwise, applies per-block FP8-to-float dequantization
        using the stored scale factors.

        Args:
            key_cache: Key cache tensor (uint8 when FP8, else float).
            value_cache: Value cache tensor.
            block_tables: Optional block table for selective dequant.
                If ``None``, dequantizes the entire cache.
            out_dtype: Target float dtype for dequantized output.

        Returns:
            ``(dequantized_keys, dequantized_values)`` in *out_dtype*.
        """
        if not self.cache_dtype_is_fp8 or self.k_scales is None or self.v_scales is None:
            return key_cache, value_cache

        num_blocks = key_cache.shape[0]
        key_cache.shape[1]

        # Broadcast per-block scales to [num_blocks, 1, 1, 1]
        k_scale = self.k_scales[:num_blocks].reshape(num_blocks, 1, 1, 1)
        v_scale = self.v_scales[:num_blocks].reshape(num_blocks, 1, 1, 1)

        deq_keys = mx.from_fp8(key_cache, dtype=out_dtype) * k_scale
        deq_values = mx.from_fp8(value_cache, dtype=out_dtype) * v_scale

        return deq_keys, deq_values

    def concatenate_to_cache(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        cache_metadata: tp.Any,
        rope: tp.Any = None,
        slot_ids: tp.Sequence[int] | None = None,
        query_lens: tp.Sequence[int] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, tp.Any]:
        """Append K/V to the paged cache and optionally apply RoPE.

        All operations are pure MLX and compatible with ``mx.compile``.

        Args:
            queries: Query tensor, shape ``[total_tokens, num_heads, head_dim]``.
            keys: Key tensor, shape ``[total_tokens, num_kv_heads, head_dim]``.
            values: Value tensor, shape ``[total_tokens, num_kv_heads, head_dim]``.
            cache_metadata: A :class:`PageMetadata` with query start locations
                and (optionally) block tables / kv lengths. ``slot_ids`` are read
                from this metadata when available.
            rope: Optional rotary position embedding callable.
            slot_ids: Explicit slot IDs override. When ``None``, uses
                ``cache_metadata.slot_ids``.
            query_lens: Optional pre-computed query lengths (one per slot).
                When provided, cumulative start locations are derived from
                these instead of ``cache_metadata.query_start_loc``.

        Returns:
            A four-tuple of ``(prepared_queries, key_cache, value_cache,
            resolved_metadata)`` where ``key_cache`` and ``value_cache`` are the
            full block-level cache arrays, and ``resolved_metadata`` has updated
            block tables and KV lengths for the selected slots.
        """
        if slot_ids is None:
            slot_ids = getattr(cache_metadata, "slot_ids", None) or ()
        resolved_slot_ids = tuple(int(s) for s in slot_ids)
        slot_array = mx.array(list(resolved_slot_ids), dtype=mx.int32)

        if query_lens is not None:
            qlens = [int(q) for q in query_lens]
            starts = [0]
            for q in qlens:
                starts.append(starts[-1] + q)
            seq_ranges = [(starts[i], starts[i + 1]) for i in range(len(qlens))]
        else:
            qsl = cache_metadata.query_start_loc
            if isinstance(qsl, mx.array):
                qsl_list = qsl.tolist()
            else:
                qsl_list = list(qsl)
            seq_ranges = [(int(qsl_list[i]), int(qsl_list[i + 1])) for i in range(len(resolved_slot_ids))]

        computed_query_lens = [max(0, se - ss) for ss, se in seq_ranges]
        is_single_token_decode = (
            bool(computed_query_lens)
            and queries.shape[0] == len(computed_query_lens)
            and all(ql == 1 for ql in computed_query_lens)
        )

        block_size = self.block_size
        base_kv_lens = self.kv_lens[slot_array]
        page_key_cache = self.page_key_cache
        page_value_cache = self.page_value_cache
        page_vec_size = int(self.page_vec_size or 0)

        if is_single_token_decode:
            token_positions = base_kv_lens.astype(mx.int32)
            prepared_queries = queries
            prepared_keys = keys
            prepared_values = values

            if rope is not None:
                offsets = token_positions
                q_batched = mx.expand_dims(queries, axis=1).transpose(0, 2, 1, 3)
                k_batched = mx.expand_dims(keys, axis=1).transpose(0, 2, 1, 3)
                prepared_queries = rope(q_batched, offset=offsets).transpose(0, 2, 1, 3)[:, 0]
                prepared_keys = rope(k_batched, offset=offsets).transpose(0, 2, 1, 3)[:, 0]

            block_ids = self.block_tables[slot_array, token_positions // block_size]
            in_block_offsets = token_positions % block_size

            if self.cache_dtype_is_fp8 and self.k_scales is not None and self.v_scales is not None:
                # Compute per-token scales and quantize to FP8 E4M3
                scale_result = self.compute_kv_scales(prepared_keys, prepared_values)
                if scale_result is not None:
                    k_scale, v_scale = scale_result
                else:
                    k_scale = mx.array(1.0, dtype=mx.float32)
                    v_scale = mx.array(1.0, dtype=mx.float32)
                quantized_keys = _fp8_quantize(prepared_keys, k_scale)
                quantized_values = _fp8_quantize(prepared_values, v_scale)
                self.key_cache[block_ids, in_block_offsets] = quantized_keys
                self.value_cache[block_ids, in_block_offsets] = quantized_values
                # Update per-block scales (use max of existing and new)
                self.k_scales[block_ids] = mx.maximum(self.k_scales[block_ids], k_scale)
                self.v_scales[block_ids] = mx.maximum(self.v_scales[block_ids], v_scale)
            else:
                self.key_cache[block_ids, in_block_offsets] = prepared_keys
                self.value_cache[block_ids, in_block_offsets] = prepared_values

            if page_key_cache is not None and page_value_cache is not None and page_vec_size > 0:
                num_vecs = self.head_dim // page_vec_size
                packed_keys = prepared_keys.reshape(-1, self.num_kv_heads, num_vecs, page_vec_size)
                self.page_key_cache[block_ids, :, :, in_block_offsets] = packed_keys
                self.page_value_cache[block_ids, :, :, in_block_offsets] = prepared_values
            self.kv_lens[slot_array] = token_positions + 1

            resolved_metadata = cache_metadata.with_cache_state(
                block_tables=self.block_tables[slot_array],
                kv_lens=self.kv_lens[slot_array],
                block_size=block_size,
            )
            return prepared_queries, self.key_cache, self.value_cache, resolved_metadata

        query_chunks: list[mx.array] = []
        key_chunks: list[mx.array] = []
        value_chunks: list[mx.array] = []
        slot_parts: list[mx.array] = []
        position_parts: list[mx.array] = []

        for seq_idx, slot in enumerate(resolved_slot_ids):
            seq_start, seq_end = seq_ranges[seq_idx]
            num_tokens = seq_end - seq_start
            if num_tokens <= 0:
                continue

            q_chunk = queries[seq_start:seq_end]
            k_chunk = keys[seq_start:seq_end]
            v_chunk = values[seq_start:seq_end]
            offset = base_kv_lens[seq_idx]

            if rope is not None:
                q_4d = q_chunk[None].transpose(0, 2, 1, 3)
                k_4d = k_chunk[None].transpose(0, 2, 1, 3)
                q_4d = rope(q_4d, offset=offset).transpose(0, 2, 1, 3)[0]
                k_4d = rope(k_4d, offset=offset).transpose(0, 2, 1, 3)[0]
                q_chunk = q_4d
                k_chunk = k_4d

            query_chunks.append(q_chunk)
            key_chunks.append(k_chunk)
            value_chunks.append(v_chunk)
            slot_parts.append(mx.full((num_tokens,), slot, dtype=mx.int32))
            position_parts.append(offset + mx.arange(num_tokens, dtype=mx.int32))

        prepared_queries = (
            mx.concatenate(query_chunks, axis=0)
            if query_chunks
            else mx.zeros((0, queries.shape[1], queries.shape[2]), dtype=queries.dtype)
        )
        prepared_keys = (
            mx.concatenate(key_chunks, axis=0)
            if key_chunks
            else mx.zeros((0, keys.shape[1], keys.shape[2]), dtype=keys.dtype)
        )
        prepared_values = (
            mx.concatenate(value_chunks, axis=0)
            if value_chunks
            else mx.zeros((0, values.shape[1], values.shape[2]), dtype=values.dtype)
        )

        if slot_parts:
            token_slots = mx.concatenate(slot_parts, axis=0)
            token_positions = mx.concatenate(position_parts, axis=0)
            block_ids = self.block_tables[token_slots, token_positions // block_size]
            in_block_offsets = token_positions % block_size

            slot_mapping = block_ids * block_size + in_block_offsets

            if self.cache_dtype_is_fp8 and self.k_scales is not None and self.v_scales is not None:
                scale_result = self.compute_kv_scales(prepared_keys, prepared_values)
                if scale_result is not None:
                    k_scale, v_scale = scale_result
                else:
                    k_scale = mx.array(1.0, dtype=mx.float32)
                    v_scale = mx.array(1.0, dtype=mx.float32)
                quantized_keys = _fp8_quantize(prepared_keys, k_scale)
                quantized_values = _fp8_quantize(prepared_values, v_scale)
                self.key_cache[block_ids, in_block_offsets] = quantized_keys
                self.value_cache[block_ids, in_block_offsets] = quantized_values
                # Update per-block scales (broadcast to all touched blocks)
                self.k_scales[block_ids] = mx.maximum(self.k_scales[block_ids], k_scale)
                self.v_scales[block_ids] = mx.maximum(self.v_scales[block_ids], v_scale)
            else:
                self.key_cache[block_ids, in_block_offsets] = prepared_keys
                self.value_cache[block_ids, in_block_offsets] = prepared_values

            if page_key_cache is not None and page_value_cache is not None and page_vec_size > 0:
                slot_mapping = block_ids * block_size + in_block_offsets
                if self.cache_dtype_is_fp8:
                    page_keys_for_metal = prepared_keys
                    page_values_for_metal = prepared_values
                else:
                    page_keys_for_metal = prepared_keys
                    page_values_for_metal = prepared_values
                self.page_key_cache, self.page_value_cache = _reshape_and_cache_metal(
                    page_keys_for_metal,
                    page_values_for_metal,
                    self.page_key_cache,
                    self.page_value_cache,
                    slot_mapping,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_vec_size=page_vec_size,
                    block_size=block_size,
                )

        self.kv_lens[slot_array] = base_kv_lens + mx.array(computed_query_lens, dtype=mx.int32)
        resolved_metadata = cache_metadata.with_cache_state(
            block_tables=self.block_tables[slot_array],
            kv_lens=self.kv_lens[slot_array],
            block_size=block_size,
        )
        return prepared_queries, self.key_cache, self.value_cache, resolved_metadata


class PageCacheConfig:
    """Static configuration for a ragged-pages (paged) KV cache.

    Matches EasyDeL's ``PageCacheConfig`` interface.

    Attributes:
        num_hidden_layers: Number of transformer layers.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimensionality of each attention head.
        page_size: Number of tokens per cache page/block.
        max_num_reqs: Maximum number of concurrent sequences.
        max_model_length: Maximum sequence length.
        memory_utilization: Fraction of available memory to use.
        dtype: Data type for cache tensors.
        cache_dtype: Override for cache storage dtype. ``"fp8"`` enables
            FP8 E4M3 quantized KV cache. ``None`` or ``"auto"`` uses *dtype*.
    """

    __slots__ = (
        "cache_dtype",
        "dtype",
        "head_dim",
        "max_model_length",
        "max_num_reqs",
        "memory_utilization",
        "num_hidden_layers",
        "num_kv_heads",
        "page_size",
    )

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_num_reqs: int = 4,
        max_model_length: int = 4096,
        memory_utilization: float = 0.85,
        dtype: mx.Dtype = mx.float16,
        cache_dtype: str | None = None,
    ) -> None:
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_length = int(max_model_length)
        self.memory_utilization = float(memory_utilization)
        self.dtype = dtype
        self.cache_dtype = cache_dtype

    @property
    def num_pages(self) -> int:
        """Total number of pages needed."""
        blocks_per_seq = math.ceil(self.max_model_length / self.page_size)
        return blocks_per_seq * self.max_num_reqs

    @classmethod
    def create(cls, **kwargs) -> PageCacheConfig:
        """Factory method matching EasyDeL's ``BaseCacheConfig.create``."""
        return cls(**kwargs)


class PageCache:
    """Multi-layer container for ragged-pages (paged) KV caches.

    Matches EasyDeL's ``PageCache`` interface — a list of
    per-layer ``PageCacheView`` instances sharing the same config.

    Attributes:
        views: Per-layer cache views.
        config: Shared configuration.
    """

    __slots__ = ("config", "views")

    def __init__(self, views: list[PageCacheView], config: PageCacheConfig | None = None) -> None:
        self.views = views
        self.config = config

    def __getitem__(self, index: int) -> PageCacheView:
        return self.views[index]

    def __setitem__(self, index: int, view: PageCacheView) -> None:
        self.views[index] = view

    def __len__(self) -> int:
        return len(self.views)

    def __iter__(self):
        return iter(self.views)

    @classmethod
    def init_cache(cls, config: PageCacheConfig) -> PageCache:
        """Initialize all layers from a shared config.

        Args:
            config: Shared ragged-pages configuration.

        Returns:
            A :class:`PageCache` with one view per layer.
        """
        views = [
            PageCacheView.init(
                num_seqs=config.max_num_reqs,
                max_seq_len=config.max_model_length,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                block_size=config.page_size,
                dtype=config.dtype,
                cache_dtype=config.cache_dtype,
            )
            for _ in range(config.num_hidden_layers)
        ]
        return cls(views=views, config=config)

    def clear(self) -> None:
        """Clear all layer caches."""
        for view in self.views:
            view.clear()

    def copy_blocks(self, block_mapping: mx.array) -> None:
        """Copy cache blocks across all layers for CoW operations.

        Args:
            block_mapping: An ``mx.array`` of shape ``[num_pairs, 2]`` with
                dtype ``int32`` where each row is ``(src_block, dst_block)``.
        """
        for view in self.views:
            view.copy_blocks(block_mapping)


def build_query_start_loc(query_lens: Iterable["int"]) -> mx.array:
    """Build cumulative query start locations from per-sequence lengths.

    Args:
        query_lens: Per-sequence query lengths.

    Returns:
        An ``mx.array`` of shape ``[num_seqs + 1]`` with dtype ``int32``
        containing cumulative offsets starting from zero.

    Example:
        >>> build_query_start_loc([3, 5, 2])
        array([0, 3, 8, 10], dtype=int32)
    """
    offsets = [0]
    for length in query_lens:
        offsets.append(offsets[-1] + int(length))
    return mx.array(offsets, dtype=mx.int32)


@dataclass(slots=True)
class PageMetadata:
    """Kernel-agnostic paged-attention metadata.

    Carries all the information a paged-attention kernel needs to locate
    cached key/value data across block tables.

    Attributes:
        query_start_loc: Cumulative query start offsets of shape
            ``[num_seqs + 1]``, dtype ``int32``.
        block_tables: Block-to-physical-page mapping of shape
            ``[num_seqs, blocks_per_seq]``, or ``None`` before resolution.
        kv_lens: Per-sequence KV cache lengths of shape ``[num_seqs]``,
            or ``None`` before resolution.
        block_size: Number of tokens per cache block, or ``None`` before
            resolution.
        sliding_window: Optional sliding-window size for local attention.
        is_single_token_decode: Whether every active sequence contributes
            exactly one token in this step.
    """

    query_start_loc: mx.ArrayLike
    block_tables: mx.ArrayLike | None = None
    kv_lens: mx.ArrayLike | None = None
    block_size: int | None = None
    sliding_window: int | None = None
    is_single_token_decode: bool = False
    slot_ids: tuple[int, ...] | None = None

    def with_cache_state(
        self,
        *,
        block_tables: mx.ArrayLike,
        kv_lens: mx.ArrayLike,
        block_size: int,
    ) -> PageMetadata:
        """Return a new :class:`PageMetadata` with resolved cache state.

        Args:
            block_tables: Block table array for the selected sequences.
            kv_lens: KV lengths array for the selected sequences.
            block_size: Number of tokens per block.

        Returns:
            A new :class:`PageMetadata` with the provided cache state fields
            populated, preserving the original ``query_start_loc`` and
            ``sliding_window``.
        """
        return PageMetadata(
            query_start_loc=self.query_start_loc,
            block_tables=block_tables,
            kv_lens=kv_lens,
            block_size=int(block_size),
            sliding_window=self.sliding_window,
            is_single_token_decode=self.is_single_token_decode,
            slot_ids=self.slot_ids,
        )

    def with_sliding_window(self, sliding_window: int | None) -> PageMetadata:
        """Return a copy with a different sliding-window setting.

        Args:
            sliding_window: New sliding-window size, or ``None`` to disable.

        Returns:
            A new :class:`PageMetadata` with the updated sliding window.
        """
        return PageMetadata(
            query_start_loc=self.query_start_loc,
            block_tables=self.block_tables,
            kv_lens=self.kv_lens,
            block_size=self.block_size,
            sliding_window=sliding_window,
            is_single_token_decode=self.is_single_token_decode,
            slot_ids=self.slot_ids,
        )


__all__ = (
    "PageCache",
    "PageCacheConfig",
    "PageCacheView",
    "PageMetadata",
    "build_query_start_loc",
)
