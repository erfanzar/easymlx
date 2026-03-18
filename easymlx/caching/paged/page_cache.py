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

"""Paged cache views for attention execution.

All hot-path array ops are pure MLX — no .item() calls — so the entire
forward path is compatible with ``mx.compile``.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx

from .page_metadata import PageMetadata
from .paged_kv_cache import PagedKVCache


class PageCache:
    """Per-call view over a paged KV cache for a selected set of slots.

    Designed to be captured in an ``mx.compile`` closure.  The query_start_loc
    is set once at construction from *integer* query_lens so that slice bounds
    are plain Python ints (compile-safe).
    """

    def __init__(
        self,
        cache: PagedKVCache,
        slot_ids: tp.Sequence[int],
        query_lens: tp.Sequence[int] | None = None,
    ):
        """Initialize a per-call page cache view.

        Args:
            cache: The underlying :class:`PagedKVCache` storage.
            slot_ids: Sequence slot indices that this view operates on.
                Must be unique and within ``[0, cache.num_seqs)``.
            query_lens: Optional per-sequence query lengths. When provided,
                cumulative start locations are pre-computed as plain Python
                ints for ``mx.compile`` compatibility.

        Raises:
            ValueError: If *slot_ids* are not unique or contain out-of-range
                values.
        """
        self.cache = cache
        self.slot_ids = tuple(int(s) for s in slot_ids)
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must be unique for a paged cache view.")
        if any(s < 0 or s >= self.cache.num_seqs for s in self.slot_ids):
            raise ValueError("slot_ids contain an out-of-range cache slot.")
        self._slot_array = mx.array(list(self.slot_ids), dtype=mx.int32)

        if query_lens is not None:
            qlens = [int(q) for q in query_lens]
            starts = [0]
            for q in qlens:
                starts.append(starts[-1] + q)
            self._seq_ranges: list[tuple[int, int]] = [(starts[i], starts[i + 1]) for i in range(len(qlens))]
        else:
            self._seq_ranges = []

    @property
    def kv_lens(self) -> mx.array:
        """Return the per-sequence KV lengths from the underlying cache."""
        return self.cache.kv_lens

    @property
    def block_tables(self) -> mx.array:
        """Return the block tables from the underlying cache."""
        return self.cache.block_tables

    @property
    def offset(self) -> int:
        """Return the current sequence offset (max kv_len across active slots).

        This is used by modules that need a position offset for things
        like rotary embeddings or convolution state tracking.
        """
        if not self.slot_ids:
            return 0
        slot_lens = self.cache.kv_lens[self._slot_array]
        return int(mx.max(slot_lens).item())

    @property
    def block_size(self) -> int:
        """Return the block size (tokens per block) of the underlying cache."""
        return int(self.cache.block_size)

    def concatenate_to_cache(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        cache_metadata: PageMetadata,
        rope: tp.Any = None,
    ) -> tuple[mx.array, mx.array, mx.array, PageMetadata]:
        """Append K/V to the paged cache and optionally apply RoPE.

        All operations are pure MLX and compatible with ``mx.compile``.

        Args:
            queries: Query tensor, shape ``[total_tokens, num_heads, head_dim]``.
            keys: Key tensor, shape ``[total_tokens, num_kv_heads, head_dim]``.
            values: Value tensor, shape ``[total_tokens, num_kv_heads, head_dim]``.
            cache_metadata: A :class:`PageMetadata` with query start locations
                and (optionally) block tables / kv lengths.
            rope: Optional rotary position embedding callable. When provided,
                it is applied to query and key chunks using the per-token
                offset from the cache.

        Returns:
            A four-tuple of ``(prepared_queries, key_cache, value_cache,
            resolved_metadata)`` where ``key_cache`` and ``value_cache`` are the
            full block-level cache arrays, and ``resolved_metadata`` has updated
            block tables and KV lengths for the selected slots.
        """
        block_size = self.cache.block_size
        base_kv_lens = self.cache.kv_lens[self._slot_array]

        if self._seq_ranges:
            seq_ranges = self._seq_ranges
        else:
            qsl = cache_metadata.query_start_loc
            if isinstance(qsl, mx.array):
                qsl_list = qsl.tolist()
            else:
                qsl_list = list(qsl)
            seq_ranges = [(int(qsl_list[i]), int(qsl_list[i + 1])) for i in range(len(self.slot_ids))]

        query_lens = [max(0, seq_end - seq_start) for seq_start, seq_end in seq_ranges]
        is_single_token_decode = (
            bool(query_lens)
            and queries.shape[0] == len(query_lens)
            and all(query_len == 1 for query_len in query_lens)
        )

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

            block_ids = self.cache.block_tables[self._slot_array, token_positions // block_size]
            in_block_offsets = token_positions % block_size
            self.cache.key_cache[block_ids, in_block_offsets] = prepared_keys
            self.cache.value_cache[block_ids, in_block_offsets] = prepared_values
            self.cache.kv_lens[self._slot_array] = token_positions + 1

            resolved_metadata = cache_metadata.with_cache_state(
                block_tables=self.cache.block_tables[self._slot_array],
                kv_lens=self.cache.kv_lens[self._slot_array],
                block_size=block_size,
            )
            return prepared_queries, self.cache.key_cache, self.cache.value_cache, resolved_metadata

        query_chunks: list[mx.array] = []
        key_chunks: list[mx.array] = []
        value_chunks: list[mx.array] = []
        slot_parts: list[mx.array] = []
        position_parts: list[mx.array] = []

        for seq_idx, slot in enumerate(self.slot_ids):
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
            block_ids = self.cache.block_tables[token_slots, token_positions // block_size]
            in_block_offsets = token_positions % block_size
            self.cache.key_cache[block_ids, in_block_offsets] = prepared_keys
            self.cache.value_cache[block_ids, in_block_offsets] = prepared_values

        self.cache.kv_lens[self._slot_array] = base_kv_lens + mx.array(query_lens, dtype=mx.int32)
        resolved_metadata = cache_metadata.with_cache_state(
            block_tables=self.cache.block_tables[self._slot_array],
            kv_lens=self.cache.kv_lens[self._slot_array],
            block_size=block_size,
        )
        return prepared_queries, self.cache.key_cache, self.cache.value_cache, resolved_metadata


__all__ = "PageCache"
