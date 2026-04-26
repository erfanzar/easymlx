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

from __future__ import annotations

import functools
import math
import typing as tp
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .cache import PageCacheView, PageMetadata

_TURBOQUANT_DEFAULT_BITS = 3
_TURBOQUANT_DEFAULT_SEED = 42
_MAX_TURBOQUANT_METAL_HEAD_DIM = 256


def _float32_scalar_ptr(value: float) -> mx.array:
    return mx.array([float(value)], dtype=mx.float32)


def _int32_scalar_ptr(value: int) -> mx.array:
    return mx.array([int(value)], dtype=mx.int32)


def _resolve_decode_threadgroup_size(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    return 256


def _ensure_mx(value: tp.Any, dtype: mx.Dtype | None = None) -> mx.array:
    if isinstance(value, mx.array):
        return value if dtype is None or value.dtype == dtype else value.astype(dtype)
    return mx.array(value, dtype=dtype) if dtype is not None else mx.array(value)


def is_turboquant_cache_dtype(cache_dtype: str | None) -> bool:
    """Return ``True`` when *cache_dtype* requests TurboQuant storage."""
    cache_dtype = str(cache_dtype or "auto").lower()
    return cache_dtype.startswith("turboquant") or cache_dtype.startswith("tq")


def resolve_turboquant_bits(
    cache_dtype: str | None,
    cache_bits: int | None = None,
) -> int | None:
    """Resolve the TurboQuant bit-width from cache settings."""
    if not is_turboquant_cache_dtype(cache_dtype):
        return None

    cache_dtype = str(cache_dtype or "").lower()
    digits = "".join(ch for ch in cache_dtype if ch.isdigit())
    bits = int(digits) if digits else int(cache_bits or _TURBOQUANT_DEFAULT_BITS)
    if bits not in (2, 3, 4):
        raise ValueError("TurboQuant cache_bits must be 2, 3, or 4.")
    return bits


def _packed_word_count(value_count: int, bits: int) -> int:
    return (int(value_count) * int(bits) + 31) // 32


@functools.lru_cache(maxsize=64)
def _bit_pack_plan(count: int, bits: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if bits <= 0:
        raise ValueError("bits must be positive.")

    word_count = _packed_word_count(count, bits)
    bit_offsets = np.arange(count, dtype=np.int32) * int(bits)
    word_indices = bit_offsets // 32
    shifts = (bit_offsets % 32).astype(np.uint32)
    spill = shifts.astype(np.int32) + int(bits) - 32
    has_spill = spill > 0
    spill_shifts = np.where(has_spill, int(bits) - spill, 0).astype(np.uint32)
    next_indices = np.minimum(word_indices + 1, word_count - 1)
    word_ids = np.arange(word_count, dtype=np.int32)[:, None]
    lower_masks = (word_ids == word_indices[None, :]).astype(np.uint32)
    spill_masks = ((word_ids == next_indices[None, :]) & has_spill[None, :]).astype(np.uint32)
    return (
        mx.array(shifts, dtype=mx.uint32),
        mx.array(spill_shifts, dtype=mx.uint32),
        mx.array(lower_masks, dtype=mx.uint32),
        mx.array(spill_masks, dtype=mx.uint32),
    )


def _pack_bits(values: mx.array, bits: int) -> mx.array:
    values = _ensure_mx(values, mx.uint32)
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension.")
    count = int(values.shape[-1])
    word_count = _packed_word_count(count, bits)
    flat = values.reshape(-1, count)
    mask = (1 << bits) - 1
    shifts, spill_shifts, lower_masks, spill_masks = _bit_pack_plan(count, bits)
    current = flat & mask
    lower_contrib = mx.left_shift(current, shifts[None, :])
    packed = mx.sum(lower_contrib[:, None, :] * lower_masks[None, :, :], axis=-1).astype(mx.uint32)
    if bool(mx.any(spill_masks).item()):
        spill_contrib = mx.right_shift(current, spill_shifts[None, :])
        packed = mx.bitwise_or(
            packed,
            mx.sum(spill_contrib[:, None, :] * spill_masks[None, :, :], axis=-1).astype(mx.uint32),
        )

    return packed.reshape((*values.shape[:-1], word_count))


def _unpack_bits(words: mx.array, bits: int, count: int) -> mx.array:
    words = _ensure_mx(words, mx.uint32)
    if words.ndim == 0:
        raise ValueError("words must have at least one dimension.")
    word_count = _packed_word_count(count, bits)
    if int(words.shape[-1]) != word_count:
        raise ValueError("Packed word count does not match the requested output length.")

    flat = words.reshape(-1, word_count)
    mask = (1 << bits) - 1

    bit_offsets = mx.arange(count, dtype=mx.int32) * int(bits)
    word_indices = bit_offsets // 32
    shifts = (bit_offsets % 32).astype(mx.uint32)

    unpacked = mx.right_shift(mx.take(flat, word_indices, axis=1), shifts[None, :])

    spill = shifts.astype(mx.int32) + int(bits) - 32
    has_spill = spill > 0
    if bool(mx.any(has_spill).item()):
        next_indices = mx.minimum(word_indices + 1, word_count - 1)
        spill_shifts = mx.where(
            has_spill,
            mx.full(spill.shape, int(bits), dtype=mx.int32) - spill,
            mx.zeros_like(spill),
        ).astype(mx.uint32)
        unpacked = mx.bitwise_or(
            unpacked,
            mx.where(
                has_spill[None, :],
                mx.left_shift(mx.take(flat, next_indices, axis=1), spill_shifts[None, :]),
                mx.zeros_like(unpacked),
            ),
        )

    unpacked = unpacked & mask

    return unpacked.reshape((*words.shape[:-1], count))


def _pack_sign_bits(signs: mx.array) -> mx.array:
    signs = _ensure_mx(signs, mx.bool_)
    if signs.ndim == 0:
        raise ValueError("signs must have at least one dimension.")
    return _pack_bits(signs.astype(mx.uint32), 1)


def _unpack_sign_bits(words: mx.array, count: int) -> mx.array:
    unpacked = _unpack_bits(words, 1, count).astype(mx.float32)
    return unpacked * 2.0 - 1.0


@functools.lru_cache(maxsize=32)
def _orthogonal_matrix(head_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((head_dim, head_dim), dtype=np.float32)
    q, r = np.linalg.qr(gaussian)
    diag_sign = np.sign(np.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.reshape(1, -1)
    return np.asarray(q, dtype=np.float32)


@functools.lru_cache(maxsize=32)
def _qjl_projection(head_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal((head_dim, head_dim), dtype=np.float32), dtype=np.float32)


@functools.lru_cache(maxsize=64)
def _lloyd_max_centroids(head_dim: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    if bits < 1:
        raise ValueError("bits must be positive.")

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(head_dim)
    grid = np.linspace(-6.0 * sigma, 6.0 * sigma, 32768, dtype=np.float64)
    pdf = np.exp(-0.5 * (grid / sigma) ** 2)
    pdf /= pdf.sum()

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = np.linspace(lo, hi, n_levels, endpoint=False, dtype=np.float64)
    centroids += (hi - lo) / (2 * n_levels)

    for _ in range(128):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        bins = np.searchsorted(boundaries, grid, side="right")
        weight_sum = np.bincount(bins, weights=pdf, minlength=n_levels).astype(np.float64)
        weighted_sum = np.bincount(bins, weights=pdf * grid, minlength=n_levels).astype(np.float64)
        new_centroids = np.where(weight_sum > 1e-12, weighted_sum / weight_sum, centroids)
        if np.max(np.abs(new_centroids - centroids)) < 1e-9:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    return np.asarray(centroids, dtype=np.float32), np.asarray(boundaries, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TurboQuantParams:
    head_dim: int
    bits: int
    mse_bits: int
    key_rotation: mx.array
    value_rotation: mx.array
    projection: mx.array
    key_centroids: mx.array
    key_boundaries: mx.array
    value_centroids: mx.array
    value_boundaries: mx.array


@functools.lru_cache(maxsize=32)
def get_turboquant_params(
    head_dim: int,
    bits: int,
    seed: int = _TURBOQUANT_DEFAULT_SEED,
) -> TurboQuantParams:
    mse_bits = max(bits - 1, 1)
    key_centroids, key_boundaries = _lloyd_max_centroids(head_dim, mse_bits)
    value_centroids, value_boundaries = _lloyd_max_centroids(head_dim, bits)
    return TurboQuantParams(
        head_dim=int(head_dim),
        bits=int(bits),
        mse_bits=int(mse_bits),
        key_rotation=mx.array(_orthogonal_matrix(head_dim, seed), dtype=mx.float32),
        value_rotation=mx.array(_orthogonal_matrix(head_dim, seed + 100), dtype=mx.float32),
        projection=mx.array(_qjl_projection(head_dim, seed + 1), dtype=mx.float32),
        key_centroids=mx.array(key_centroids, dtype=mx.float32),
        key_boundaries=mx.array(key_boundaries, dtype=mx.float32),
        value_centroids=mx.array(value_centroids, dtype=mx.float32),
        value_boundaries=mx.array(value_boundaries, dtype=mx.float32),
    )


def _bucketize(values: mx.array, boundaries: mx.array) -> mx.array:
    comparisons = mx.expand_dims(values, axis=-1) >= boundaries
    return mx.sum(comparisons, axis=-1).astype(mx.uint32)


def _quantize_keys(keys: mx.array, params: TurboQuantParams) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    keys = _ensure_mx(keys, mx.float32)
    head_dim = int(params.head_dim)
    flat = keys.reshape(-1, head_dim)
    norms = mx.linalg.norm(flat, axis=-1, keepdims=True)
    safe_norms = mx.maximum(norms, 1e-8)
    normalized = flat / safe_norms

    rotated = normalized @ params.key_rotation.T
    indices = _bucketize(rotated, params.key_boundaries)
    reconstructed_rotated = params.key_centroids[indices]
    reconstructed = (reconstructed_rotated @ params.key_rotation) * norms

    residual = flat - reconstructed
    residual_norm = mx.linalg.norm(residual, axis=-1)
    projected = residual @ params.projection.T
    sign_words = _pack_sign_bits(projected >= 0)
    packed_indices = _pack_bits(indices, params.mse_bits)

    prefix_shape = keys.shape[:-1]
    return (
        packed_indices.reshape((*prefix_shape, packed_indices.shape[-1])),
        norms.reshape(prefix_shape).astype(mx.float16),
        residual_norm.reshape(prefix_shape).astype(mx.float16),
        sign_words.reshape((*prefix_shape, sign_words.shape[-1])),
    )


def _quantize_values(values: mx.array, params: TurboQuantParams) -> tuple[mx.array, mx.array]:
    values = _ensure_mx(values, mx.float32)
    head_dim = int(params.head_dim)
    flat = values.reshape(-1, head_dim)
    norms = mx.linalg.norm(flat, axis=-1, keepdims=True)
    safe_norms = mx.maximum(norms, 1e-8)
    normalized = flat / safe_norms

    rotated = normalized @ params.value_rotation.T
    indices = _bucketize(rotated, params.value_boundaries)
    packed_indices = _pack_bits(indices, params.bits)

    prefix_shape = values.shape[:-1]
    return (
        packed_indices.reshape((*prefix_shape, packed_indices.shape[-1])),
        norms.reshape(prefix_shape).astype(mx.float16),
    )


def _decode_key_mse(
    packed_indices: mx.array,
    norms: mx.array,
    params: TurboQuantParams,
) -> mx.array:
    indices = _unpack_bits(packed_indices, params.mse_bits, params.head_dim)
    rotated = params.key_centroids[indices]
    return (rotated @ params.key_rotation) * _ensure_mx(norms, mx.float32)[..., None]


def _decode_value_vectors(
    packed_indices: mx.array,
    norms: mx.array,
    params: TurboQuantParams,
) -> mx.array:
    indices = _unpack_bits(packed_indices, params.bits, params.head_dim)
    rotated = params.value_centroids[indices]
    return (rotated @ params.value_rotation) * _ensure_mx(norms, mx.float32)[..., None]


def _decode_qjl_signs(
    packed_signs: mx.array,
    params: TurboQuantParams,
) -> mx.array:
    return _unpack_sign_bits(packed_signs, params.head_dim)


def _build_block_tables(num_seqs: int, blocks_per_seq: int) -> mx.array:
    rows = []
    for seq_idx in range(num_seqs):
        start = seq_idx * blocks_per_seq
        rows.append(mx.arange(start, start + blocks_per_seq, dtype=mx.int32))
    return mx.stack(rows)


_TURBOQUANT_METAL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""


_TURBOQUANT_METAL_DECODE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint token_idx = threadgroup_position_in_grid.y;
    uint q_head = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int total_tokens = queries_shape[0];
    int num_q_heads = queries_shape[1];
    int head_dim = queries_shape[2];
    if (total_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0) {
        return;
    }
    if ((int)token_idx >= total_tokens || (int)q_head >= num_q_heads || head_dim > 256) {
        return;
    }

    int num_kv_heads = key_cache_shape[2];
    if (num_kv_heads <= 0) {
        return;
    }
    int num_queries_per_kv = num_q_heads / num_kv_heads;
    if (num_queries_per_kv <= 0) {
        return;
    }
    int kv_head = (int)q_head / num_queries_per_kv;
    if (kv_head >= num_kv_heads) {
        kv_head = num_kv_heads - 1;
    }

    int num_seqs = query_start_loc_shape[0] - 1;
    if ((int)token_idx >= num_seqs) {
        return;
    }

    int q_start = query_start_loc[token_idx];
    int q_end = query_start_loc[token_idx + 1];
    if (q_end - q_start != 1) {
        return;
    }

    int seq_len = kv_lens[token_idx];
    int q_base = (q_start * num_q_heads + (int)q_head) * head_dim;
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

    int block_size = key_cache_shape[1];
    int max_blocks_per_seq = block_tables_shape[1];
    int key_words = key_cache_shape[3];
    int value_words = value_cache_shape[3];
    int sign_words = key_qjl_signs_shape[3];
    uint key_mask = (1u << (uint)key_bits_ptr[0]) - 1u;
    uint value_mask = (1u << (uint)value_bits_ptr[0]) - 1u;
    float q_scale = softmax_scale_ptr[0];
    float correction_scale = correction_scale_ptr[0];

    threadgroup float q_shared[256];
    threadgroup float q_key_rot_shared[256];
    threadgroup float q_proj_shared[256];
    threadgroup float out_rot_shared[256];
    threadgroup float reduce_shared[256];
    threadgroup float reduce2_shared[256];
    threadgroup float alpha_shared;
    threadgroup float p_shared;
    threadgroup float inv_l_shared;

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        q_shared[d] = (float)queries[q_base + d] * q_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        float q_key = 0.0f;
        float q_proj = 0.0f;
        int row_base = d * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            float qv = q_shared[j];
            q_key += qv * key_rotation[row_base + j];
            q_proj += qv * projection[row_base + j];
        }
        q_key_rot_shared[d] = q_key;
        q_proj_shared[d] = q_proj;
        out_rot_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start_k; pos < seq_len; ++pos) {
        int block = block_tables[token_idx * max_blocks_per_seq + (pos / block_size)];
        if (block < 0) {
            continue;
        }
        int offset = pos % block_size;
        int kv_index = (block * block_size + offset) * num_kv_heads + kv_head;
        int key_base = kv_index * key_words;
        int value_base = kv_index * value_words;
        int sign_base = kv_index * sign_words;

        float key_norm = (float)key_norms[kv_index];
        float value_norm = (float)value_norms[kv_index];
        float residual_norm = (float)key_residual_norms[kv_index];
        float score_partial = 0.0f;
        float qjl_partial = 0.0f;

        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            int key_bit_offset = d * key_bits_ptr[0];
            int key_word_idx = key_bit_offset / 32;
            int key_shift = key_bit_offset % 32;
            uint key_current = key_cache[key_base + key_word_idx] >> (uint)key_shift;
            int key_spill = key_shift + key_bits_ptr[0] - 32;
            if (key_spill > 0) {
                key_current |= key_cache[key_base + key_word_idx + 1] << (uint)(key_bits_ptr[0] - key_spill);
            }
            uint key_idx = key_current & key_mask;
            float key_centroid = key_centroids[key_idx];
            score_partial += q_key_rot_shared[d] * key_centroid;

            int sign_word_idx = d / 32;
            int sign_shift = d % 32;
            float sign = ((key_qjl_signs[sign_base + sign_word_idx] >> (uint)sign_shift) & 1u) ? 1.0f : -1.0f;
            qjl_partial += q_proj_shared[d] * sign;
        }

        reduce_shared[tid] = score_partial;
        reduce2_shared[tid] = qjl_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_shared[tid] += reduce_shared[tid + stride];
                reduce2_shared[tid] += reduce2_shared[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float score = reduce_shared[0] * key_norm + correction_scale * reduce2_shared[0] * residual_norm;
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
            int value_bit_offset = d * value_bits_ptr[0];
            int value_word_idx = value_bit_offset / 32;
            int value_shift = value_bit_offset % 32;
            uint value_current = value_cache[value_base + value_word_idx] >> (uint)value_shift;
            int value_spill = value_shift + value_bits_ptr[0] - 32;
            if (value_spill > 0) {
                value_current |= value_cache[value_base + value_word_idx + 1] << (uint)(value_bits_ptr[0] - value_spill);
            }
            uint value_idx = value_current & value_mask;
            float value_centroid = value_centroids[value_idx] * value_norm;
            out_rot_shared[d] = out_rot_shared[d] * alpha + p * value_centroid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        inv_l_shared = l > 0.0f ? (1.0f / l) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        float out_value = 0.0f;
        for (int j = 0; j < head_dim; ++j) {
            out_value += (out_rot_shared[j] * inv_l_shared) * value_rotation[j * head_dim + d];
        }
        out[q_base + d] = (T)out_value;
    }
"""


_TURBOQUANT_METAL_DECODE_KERNEL = mx.fast.metal_kernel(
    name="turboquant_paged_decode",
    input_names=[
        "queries",
        "key_cache",
        "value_cache",
        "key_norms",
        "value_norms",
        "key_residual_norms",
        "key_qjl_signs",
        "block_tables",
        "kv_lens",
        "query_start_loc",
        "key_rotation",
        "value_rotation",
        "projection",
        "key_centroids",
        "value_centroids",
        "softmax_scale_ptr",
        "correction_scale_ptr",
        "sliding_window_ptr",
        "key_bits_ptr",
        "value_bits_ptr",
    ],
    output_names=["out"],
    source=_TURBOQUANT_METAL_DECODE_SOURCE,
    header=_TURBOQUANT_METAL_HEADER,
)


class TurboQuantPageCacheView(PageCacheView):
    """Paged cache storing TurboQuant-compressed keys and values."""

    __slots__ = (
        "cache_dtype_is_turboquant",
        "key_norms",
        "key_qjl_signs",
        "key_residual_norms",
        "turboquant_bits",
        "turboquant_seed",
        "value_norms",
    )

    def __init__(
        self,
        *,
        key_cache: mx.array,
        value_cache: mx.array,
        key_norms: mx.array,
        key_residual_norms: mx.array,
        key_qjl_signs: mx.array,
        value_norms: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        turboquant_bits: int,
        turboquant_seed: int = _TURBOQUANT_DEFAULT_SEED,
    ) -> None:
        self.cache_dtype_is_turboquant = True
        self.turboquant_bits = int(turboquant_bits)
        self.turboquant_seed = int(turboquant_seed)
        self.key_norms = key_norms
        self.key_residual_norms = key_residual_norms
        self.key_qjl_signs = key_qjl_signs
        self.value_norms = value_norms
        super().__init__(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            kv_lens=kv_lens,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_key_cache=None,
            page_value_cache=None,
            page_vec_size=0,
            cache_dtype_is_fp8=False,
            k_scales=None,
            v_scales=None,
        )

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
        cache_bits: int | None = None,
        turboquant_seed: int = _TURBOQUANT_DEFAULT_SEED,
    ) -> TurboQuantPageCacheView:
        del dtype
        bits = resolve_turboquant_bits(cache_dtype, cache_bits)
        if bits is None:
            raise ValueError("TurboQuantPageCacheView.init requires a TurboQuant cache_dtype.")
        if num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = blocks_per_seq * num_seqs
        params = get_turboquant_params(head_dim, bits, turboquant_seed)
        key_words = _packed_word_count(head_dim, params.mse_bits)
        value_words = _packed_word_count(head_dim, params.bits)
        sign_words = _packed_word_count(head_dim, 1)

        return cls(
            key_cache=mx.zeros((num_blocks, block_size, num_kv_heads, key_words), dtype=mx.uint32),
            value_cache=mx.zeros((num_blocks, block_size, num_kv_heads, value_words), dtype=mx.uint32),
            key_norms=mx.zeros((num_blocks, block_size, num_kv_heads), dtype=mx.float16),
            key_residual_norms=mx.zeros((num_blocks, block_size, num_kv_heads), dtype=mx.float16),
            key_qjl_signs=mx.zeros((num_blocks, block_size, num_kv_heads, sign_words), dtype=mx.uint32),
            value_norms=mx.zeros((num_blocks, block_size, num_kv_heads), dtype=mx.float16),
            block_tables=_build_block_tables(num_seqs, blocks_per_seq),
            kv_lens=mx.zeros((num_seqs,), dtype=mx.int32),
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            turboquant_bits=bits,
            turboquant_seed=turboquant_seed,
        )

    allocate = init

    def _validate_shape(self) -> None:
        if self.block_tables.ndim != 2:
            raise ValueError("block_tables must be rank-2: [num_seqs, blocks_per_seq]")
        if self.kv_lens.ndim != 1:
            raise ValueError("kv_lens must be rank-1: [num_seqs]")
        if self.key_cache.ndim != 4 or self.value_cache.ndim != 4:
            raise ValueError("TurboQuant key/value caches must be rank-4.")

        num_seqs, blocks_per_seq = self.block_tables.shape
        num_blocks = blocks_per_seq * num_seqs
        if self.kv_lens.shape[0] != num_seqs:
            raise ValueError("kv_lens length must match block_tables first dimension.")
        if self.key_cache.shape[0] != num_blocks or self.value_cache.shape[0] != num_blocks:
            raise ValueError("TurboQuant cache block count does not match configured sequences.")
        if self.key_cache.shape[1] != self.block_size or self.value_cache.shape[1] != self.block_size:
            raise ValueError("TurboQuant cache block size does not match block_size.")
        if self.key_cache.shape[2] != self.num_kv_heads or self.value_cache.shape[2] != self.num_kv_heads:
            raise ValueError("TurboQuant cache num_kv_heads does not match configured value.")
        dense_aux_shape = (num_blocks, self.block_size, self.num_kv_heads)
        if tuple(self.key_norms.shape) != dense_aux_shape:
            raise ValueError("key_norms shape does not match TurboQuant cache geometry.")
        if tuple(self.key_residual_norms.shape) != dense_aux_shape:
            raise ValueError("key_residual_norms shape does not match TurboQuant cache geometry.")
        if tuple(self.value_norms.shape) != dense_aux_shape:
            raise ValueError("value_norms shape does not match TurboQuant cache geometry.")
        if self.key_qjl_signs.ndim != 4:
            raise ValueError("key_qjl_signs must be rank-4.")
        if tuple(self.key_qjl_signs.shape[:3]) != dense_aux_shape:
            raise ValueError("key_qjl_signs shape does not match TurboQuant cache geometry.")
        if self.page_key_cache is not None or self.page_value_cache is not None:
            raise ValueError("TurboQuant cache does not use packed Metal page caches.")

    def _params(self) -> TurboQuantParams:
        return get_turboquant_params(self.head_dim, self.turboquant_bits, self.turboquant_seed)

    def _write_compressed(
        self,
        block_ids: mx.array,
        in_block_offsets: mx.array,
        keys: mx.array,
        values: mx.array,
    ) -> None:
        params = self._params()
        packed_keys, key_norms, key_residual_norms, key_qjl_signs = _quantize_keys(keys, params)
        packed_values, value_norms = _quantize_values(values, params)

        self.key_cache[block_ids, in_block_offsets] = packed_keys.astype(mx.uint32)
        self.value_cache[block_ids, in_block_offsets] = packed_values.astype(mx.uint32)
        self.key_norms[block_ids, in_block_offsets] = key_norms.astype(mx.float16)
        self.key_residual_norms[block_ids, in_block_offsets] = key_residual_norms.astype(mx.float16)
        self.key_qjl_signs[block_ids, in_block_offsets] = key_qjl_signs.astype(mx.uint32)
        self.value_norms[block_ids, in_block_offsets] = value_norms.astype(mx.float16)

    def append(self, seq_idx: int, key: mx.array, value: mx.array) -> None:
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

        positions = start + mx.arange(num_tokens, dtype=mx.int32)
        block_ids = self.block_tables[seq_idx, positions // self.block_size]
        in_block_offsets = positions % self.block_size
        self._write_compressed(block_ids, in_block_offsets, key, value)
        self.kv_lens[seq_idx] = start + num_tokens

    def reset(self, seq_idx: int) -> None:
        if seq_idx < 0 or seq_idx >= self.num_seqs:
            raise IndexError(f"Invalid seq_idx {seq_idx}")
        block_ids = self.block_tables[seq_idx]
        for i in range(block_ids.shape[0]):
            block_id = int(block_ids[i].item())
            if block_id < 0:
                continue
            self.key_cache[block_id] = mx.zeros_like(self.key_cache[block_id])
            self.value_cache[block_id] = mx.zeros_like(self.value_cache[block_id])
            self.key_norms[block_id] = mx.zeros_like(self.key_norms[block_id])
            self.key_residual_norms[block_id] = mx.zeros_like(self.key_residual_norms[block_id])
            self.key_qjl_signs[block_id] = mx.zeros_like(self.key_qjl_signs[block_id])
            self.value_norms[block_id] = mx.zeros_like(self.value_norms[block_id])
        self.kv_lens[seq_idx] = 0

    def clear(self, seq_idx: int | None = None) -> None:
        if seq_idx is None:
            self.key_cache = mx.zeros_like(self.key_cache)
            self.value_cache = mx.zeros_like(self.value_cache)
            self.key_norms = mx.zeros_like(self.key_norms)
            self.key_residual_norms = mx.zeros_like(self.key_residual_norms)
            self.key_qjl_signs = mx.zeros_like(self.key_qjl_signs)
            self.value_norms = mx.zeros_like(self.value_norms)
            self.kv_lens = mx.zeros_like(self.kv_lens)
            return
        self.reset(int(seq_idx))

    def reset_batch(self, seq_indices: Sequence[int]) -> None:
        for seq_idx in seq_indices:
            self.reset(int(seq_idx))

    def clear_many(self, seq_indices: Iterable[int]) -> None:
        for seq_idx in seq_indices:
            self.reset(int(seq_idx))

    def copy_blocks(self, block_mapping: mx.array) -> None:
        if block_mapping.ndim != 2 or block_mapping.shape[1] != 2:
            raise ValueError("block_mapping must be rank-2 with shape [num_pairs, 2]")
        for src_block, dst_block in block_mapping.tolist():
            if int(src_block) < 0 or int(dst_block) < 0:
                continue
            self.key_cache[int(dst_block)] = self.key_cache[int(src_block)]
            self.value_cache[int(dst_block)] = self.value_cache[int(src_block)]
            self.key_norms[int(dst_block)] = self.key_norms[int(src_block)]
            self.key_residual_norms[int(dst_block)] = self.key_residual_norms[int(src_block)]
            self.key_qjl_signs[int(dst_block)] = self.key_qjl_signs[int(src_block)]
            self.value_norms[int(dst_block)] = self.value_norms[int(src_block)]

    def dequantize_kv_cache(
        self,
        key_cache: mx.array,
        value_cache: mx.array,
        *,
        block_tables: mx.array | None = None,
        out_dtype: mx.Dtype = mx.float16,
    ) -> tuple[mx.array, mx.array]:
        del block_tables
        params = self._params()
        dense_keys = _decode_key_mse(_ensure_mx(key_cache, mx.uint32), self.key_norms, params)
        dense_values = _decode_value_vectors(_ensure_mx(value_cache, mx.uint32), self.value_norms, params)
        return dense_keys.astype(out_dtype), dense_values.astype(out_dtype)

    def gather_kv_cache(
        self,
        block_table: mx.array,
        cu_seq_lens: mx.array,
        *,
        out_dtype: mx.Dtype | None = None,
    ) -> tuple[mx.array, mx.array]:
        if out_dtype is None:
            out_dtype = mx.float16

        dense_keys, dense_values = self.dequantize_kv_cache(self.key_cache, self.value_cache, out_dtype=out_dtype)
        block_table = _ensure_mx(block_table, mx.int32)
        cu_seq_lens = _ensure_mx(cu_seq_lens, mx.int32)
        total_tokens = int(cu_seq_lens[-1].item()) if cu_seq_lens.size > 0 else 0

        gathered_k = mx.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=out_dtype)
        gathered_v = mx.zeros((total_tokens, self.num_kv_heads, self.head_dim), dtype=out_dtype)
        num_seqs = max(0, int(cu_seq_lens.shape[0]) - 1)
        for seq_idx in range(num_seqs):
            seq_start = int(cu_seq_lens[seq_idx].item())
            seq_end = int(cu_seq_lens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            if seq_len <= 0:
                continue
            positions = mx.arange(seq_len, dtype=mx.int32)
            blocks = block_table[seq_idx, positions // self.block_size]
            offsets = positions % self.block_size
            gathered_k[seq_start:seq_end] = dense_keys[blocks, offsets]
            gathered_v[seq_start:seq_end] = dense_values[blocks, offsets]

        return gathered_k, gathered_v

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
        if slot_ids is None:
            slot_ids = getattr(cache_metadata, "slot_ids", None) or ()
        resolved_slot_ids = tuple(int(s) for s in slot_ids)
        slot_array = mx.array(list(resolved_slot_ids), dtype=mx.int32)

        if query_lens is not None:
            qlens = [int(q) for q in query_lens]
            starts = [0]
            for q_len in qlens:
                starts.append(starts[-1] + q_len)
            seq_ranges = [(starts[i], starts[i + 1]) for i in range(len(qlens))]
        else:
            qsl = cache_metadata.query_start_loc
            qsl_list = qsl.tolist() if isinstance(qsl, mx.array) else list(qsl)
            seq_ranges = [(int(qsl_list[i]), int(qsl_list[i + 1])) for i in range(len(resolved_slot_ids))]

        computed_query_lens = [max(0, seq_end - seq_start) for seq_start, seq_end in seq_ranges]
        is_single_token_decode = (
            bool(computed_query_lens)
            and queries.shape[0] == len(computed_query_lens)
            and all(q_len == 1 for q_len in computed_query_lens)
        )

        block_size = self.block_size
        base_kv_lens = self.kv_lens[slot_array]

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
            self._write_compressed(block_ids, in_block_offsets, prepared_keys, prepared_values)

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
                q_chunk = rope(q_4d, offset=offset).transpose(0, 2, 1, 3)[0]
                k_chunk = rope(k_4d, offset=offset).transpose(0, 2, 1, 3)[0]

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
            self._write_compressed(block_ids, in_block_offsets, prepared_keys, prepared_values)

        self.kv_lens[slot_array] = base_kv_lens + mx.array(computed_query_lens, dtype=mx.int32)
        resolved_metadata = cache_metadata.with_cache_state(
            block_tables=self.block_tables[slot_array],
            kv_lens=self.kv_lens[slot_array],
            block_size=block_size,
        )
        return prepared_queries, self.key_cache, self.value_cache, resolved_metadata

    def turboquant_attention(
        self,
        queries: mx.array,
        cache_metadata: PageMetadata,
        *,
        softmax_scale: float | None = None,
        use_metal: bool | None = None,
    ) -> mx.array:
        if cache_metadata.block_tables is None or cache_metadata.kv_lens is None:
            raise ValueError("TurboQuant attention requires resolved block_tables and kv_lens.")

        params = self._params()
        queries = _ensure_mx(queries, mx.float32)
        block_tables = _ensure_mx(cache_metadata.block_tables, mx.int32)
        kv_lens = _ensure_mx(cache_metadata.kv_lens, mx.int32)
        query_start_loc = _ensure_mx(cache_metadata.query_start_loc, mx.int32)

        num_tokens, num_q_heads, head_dim = queries.shape
        num_kv_heads = self.num_kv_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")

        if bool(getattr(cache_metadata, "is_single_token_decode", False)):
            if use_metal is None:
                use_metal = True
            if use_metal and head_dim <= _MAX_TURBOQUANT_METAL_HEAD_DIM:
                try:
                    return self._turboquant_single_token_attention_metal(
                        queries,
                        block_tables,
                        kv_lens,
                        query_start_loc,
                        params,
                        softmax_scale=float(softmax_scale),
                        sliding_window=int(cache_metadata.sliding_window or 0),
                    )
                except Exception:
                    pass
            return self._turboquant_single_token_attention(
                queries,
                block_tables,
                kv_lens,
                query_start_loc,
                params,
                softmax_scale=float(softmax_scale),
                sliding_window=int(cache_metadata.sliding_window or 0),
            )

        num_queries_per_kv = num_q_heads // num_kv_heads
        correction_scale = math.sqrt(math.pi / 2.0) / head_dim
        sliding_window = int(cache_metadata.sliding_window or 0)
        out = mx.zeros((num_tokens, num_q_heads, head_dim), dtype=mx.float32)

        for seq_idx in range(max(0, int(query_start_loc.shape[0]) - 1)):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = int(kv_lens[seq_idx].item())
            if seq_len <= 0:
                continue

            positions = mx.arange(seq_len, dtype=mx.int32)
            blocks = block_tables[seq_idx, positions // self.block_size]
            offsets = positions % self.block_size

            seq_key_words = self.key_cache[blocks, offsets]
            seq_value_words = self.value_cache[blocks, offsets]
            seq_key_norms = self.key_norms[blocks, offsets]
            seq_value_norms = self.value_norms[blocks, offsets]
            seq_residual_norms = self.key_residual_norms[blocks, offsets]
            seq_sign_words = self.key_qjl_signs[blocks, offsets]

            key_mse = _decode_key_mse(seq_key_words, seq_key_norms, params)
            values = _decode_value_vectors(seq_value_words, seq_value_norms, params)
            signs = _decode_qjl_signs(seq_sign_words, params)

            if num_queries_per_kv != 1:
                key_mse = mx.repeat(key_mse, repeats=num_queries_per_kv, axis=1)
                values = mx.repeat(values, repeats=num_queries_per_kv, axis=1)
                signs = mx.repeat(signs, repeats=num_queries_per_kv, axis=1)
                seq_residual_norms = mx.repeat(seq_residual_norms, repeats=num_queries_per_kv, axis=1)

            for local_idx in range(q_len):
                token_idx = q_start + local_idx
                q_vec = queries[token_idx] * float(softmax_scale)
                q_projected = q_vec @ params.projection.T
                context_len = seq_len - q_len
                max_k = min(seq_len - 1, context_len + local_idx)
                if max_k < 0:
                    continue
                start_k = 0
                if sliding_window > 0:
                    start_k = max(0, max_k - sliding_window + 1)

                k_slice = key_mse[start_k : max_k + 1]
                v_slice = values[start_k : max_k + 1]
                sign_slice = signs[start_k : max_k + 1]
                seq_residual_norms[start_k : max_k + 1]
                if int(k_slice.shape[0]) == 0:
                    continue

                term1 = mx.einsum("hd,khd->hk", q_vec, k_slice)
                qjl_ip = mx.einsum("hd,khd->hk", q_projected, sign_slice)
                scores = term1 + (
                    correction_scale * qjl_ip * mx.transpose(seq_residual_norms[start_k : max_k + 1], (1, 0))
                )
                scores = scores - mx.max(scores, axis=-1, keepdims=True)
                weights = mx.exp(scores)
                weights = weights / mx.maximum(mx.sum(weights, axis=-1, keepdims=True), 1e-12)
                out[token_idx] = mx.einsum("hk,khd->hd", weights, v_slice)

        return out.astype(queries.dtype)

    def _turboquant_single_token_attention(
        self,
        queries: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        query_start_loc: mx.array,
        params: TurboQuantParams,
        *,
        softmax_scale: float,
        sliding_window: int,
    ) -> mx.array:
        num_tokens, num_q_heads, head_dim = queries.shape
        num_kv_heads = self.num_kv_heads
        num_queries_per_kv = num_q_heads // num_kv_heads
        correction_scale = math.sqrt(math.pi / 2.0) / head_dim
        out = mx.zeros((num_tokens, num_q_heads, head_dim), dtype=mx.float32)

        for seq_idx in range(max(0, int(query_start_loc.shape[0]) - 1)):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            if q_end - q_start != 1:
                raise ValueError("Single-token TurboQuant decode fast path requires exactly one query per sequence.")

            seq_len = int(kv_lens[seq_idx].item())
            if seq_len <= 0:
                continue

            start_k = max(0, seq_len - sliding_window) if sliding_window > 0 else 0
            positions = mx.arange(start_k, seq_len, dtype=mx.int32)
            blocks = block_tables[seq_idx, positions // self.block_size]
            offsets = positions % self.block_size

            seq_key_words = self.key_cache[blocks, offsets]
            seq_value_words = self.value_cache[blocks, offsets]
            seq_key_norms = _ensure_mx(self.key_norms[blocks, offsets], mx.float32)
            seq_value_norms = _ensure_mx(self.value_norms[blocks, offsets], mx.float32)
            seq_residual_norms = _ensure_mx(self.key_residual_norms[blocks, offsets], mx.float32)
            seq_sign_words = self.key_qjl_signs[blocks, offsets]

            q_vec = queries[q_start] * float(softmax_scale)
            q_key_rot = (q_vec @ params.key_rotation.T).reshape(num_kv_heads, num_queries_per_kv, head_dim)
            q_projected = (q_vec @ params.projection.T).reshape(num_kv_heads, num_queries_per_kv, head_dim)

            key_indices = _unpack_bits(seq_key_words, params.mse_bits, params.head_dim)
            value_indices = _unpack_bits(seq_value_words, params.bits, params.head_dim)
            key_rotated = params.key_centroids[key_indices]
            value_rotated = params.value_centroids[value_indices] * seq_value_norms[..., None]
            sign_vectors = _decode_qjl_signs(seq_sign_words, params)

            term1 = mx.einsum("gqd,kgd,kg->gqk", q_key_rot, key_rotated, seq_key_norms)
            qjl_ip = mx.einsum("gqd,kgd,kg->gqk", q_projected, sign_vectors, seq_residual_norms)
            scores = term1 + correction_scale * qjl_ip
            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            weights = mx.exp(scores)
            weights = weights / mx.maximum(mx.sum(weights, axis=-1, keepdims=True), 1e-12)

            rotated_out = mx.einsum("gqk,kgd->gqd", weights, value_rotated)
            out[q_start] = (rotated_out.reshape(-1, head_dim) @ params.value_rotation).reshape(num_q_heads, head_dim)

        return out.astype(queries.dtype)

    def _turboquant_single_token_attention_metal(
        self,
        queries: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        query_start_loc: mx.array,
        params: TurboQuantParams,
        *,
        softmax_scale: float,
        sliding_window: int,
    ) -> mx.array:
        total_tokens = int(queries.shape[0])
        num_q_heads = int(queries.shape[1])
        head_dim = int(queries.shape[2])
        correction_scale = math.sqrt(math.pi / 2.0) / head_dim
        threadgroup_size = _resolve_decode_threadgroup_size(head_dim)

        outputs = _TURBOQUANT_METAL_DECODE_KERNEL(
            inputs=[
                queries,
                self.key_cache,
                self.value_cache,
                self.key_norms,
                self.value_norms,
                self.key_residual_norms,
                self.key_qjl_signs,
                block_tables,
                kv_lens,
                query_start_loc,
                params.key_rotation,
                params.value_rotation,
                params.projection,
                params.key_centroids,
                params.value_centroids,
                _float32_scalar_ptr(float(softmax_scale)),
                _float32_scalar_ptr(float(correction_scale)),
                _int32_scalar_ptr(int(sliding_window)),
                _int32_scalar_ptr(int(params.mse_bits)),
                _int32_scalar_ptr(int(params.bits)),
            ],
            template=[("T", queries.dtype)],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(threadgroup_size, total_tokens, num_q_heads),
            threadgroup=(threadgroup_size, 1, 1),
        )
        return outputs[0]


__all__ = (
    "TurboQuantPageCacheView",
    "get_turboquant_params",
    "is_turboquant_cache_dtype",
    "resolve_turboquant_bits",
)
