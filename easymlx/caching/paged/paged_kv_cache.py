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

"""Paged KV cache helpers for unified attention (serving-only).

All storage uses ``mx.array`` — no numpy dependency for cache state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import mlx.core as mx


class PagedKVCache:
    """Block-table based KV cache backed by ``mx.array``.

    Attributes
    ----------
    key_cache, value_cache : mx.array
        Shape ``[num_blocks, block_size, num_kv_heads, head_dim]``.
    block_tables : mx.array
        Shape ``[num_seqs, blocks_per_seq]``, dtype ``int32``.
    kv_lens : mx.array
        Shape ``[num_seqs]``, dtype ``int32`` — current KV length per sequence.
    block_size, num_kv_heads, head_dim : int
        Scalar configuration.
    """

    __slots__ = ("block_size", "block_tables", "head_dim", "key_cache", "kv_lens", "num_kv_heads", "value_cache")

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
        self._validate_shape()

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
    def allocate(
        cls,
        *,
        num_seqs: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: mx.Dtype | type | str = mx.float16,
    ) -> PagedKVCache:
        """Allocate a new zero-initialized paged KV cache.

        Block tables are set up with a simple contiguous layout where each
        sequence owns ``ceil(max_seq_len / block_size)`` consecutive blocks.

        Args:
            num_seqs: Number of sequence slots to allocate.
            max_seq_len: Maximum sequence length each slot can hold.
            num_kv_heads: Number of key/value attention heads.
            head_dim: Dimensionality of each attention head.
            block_size: Number of tokens per cache block.
            dtype: Data type for key/value storage. Accepts ``mx.Dtype``,
                numpy dtypes, or string names (``"float16"``, ``"float32"``,
                ``"bfloat16"``).

        Returns:
            A freshly allocated :class:`PagedKVCache` with all KV lengths
            set to zero.

        Raises:
            ValueError: If *num_seqs*, *max_seq_len*, or *block_size* are
                not positive.
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

        key_cache = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
        value_cache = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
        kv_lens = mx.zeros((num_seqs,), dtype=mx.int32)

        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            kv_lens=kv_lens,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

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


__all__ = ("PagedKVCache", "build_query_start_loc")
