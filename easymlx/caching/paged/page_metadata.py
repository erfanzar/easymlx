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

"""Paged-attention metadata wrappers.

Provides the :class:`PageMetadata` dataclass that carries kernel-agnostic
metadata (query start locations, block tables, KV lengths) needed by
paged-attention implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


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
    """

    query_start_loc: mx.ArrayLike
    block_tables: mx.ArrayLike | None = None
    kv_lens: mx.ArrayLike | None = None
    block_size: int | None = None
    sliding_window: int | None = None

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
        )


__all__ = "PageMetadata"
