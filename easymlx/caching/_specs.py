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

"""Cache specification types.

Each spec describes the *kind* of cache a layer needs (attention, sliding
window, recurrent/SSM) and provides helpers for memory budgeting.

Mirrors EasyDeL's ``_specs.py`` hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class KVCacheSpec:
    """Root of the cache specification hierarchy."""

    dtype: mx.Dtype = mx.float16

    def page_size_tokens(self, page_size: int) -> int:
        """Return the number of tokens that fit in a single page.

        Args:
            page_size: The page size in tokens.

        Returns:
            The effective number of tokens per page (identity for the base class).
        """
        return page_size

    def element_byte_size(self) -> int:
        """Return the number of bytes per element for the configured dtype.

        Returns:
            Byte count for a single scalar of ``self.dtype``.
        """
        return mx.zeros(1, dtype=self.dtype).nbytes


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """Spec for standard attention caches (key + value per head)."""

    num_kv_heads: int = 1
    head_size: int = 64

    def per_token_bytes(self) -> int:
        """Return the memory footprint for one token across both key and value.

        Returns:
            Total bytes consumed by one token's key and value projections
            across all KV heads.
        """
        return 2 * self.num_kv_heads * self.head_size * self.element_byte_size()

    def max_memory_usage_bytes(self, max_length: int) -> int:
        """Return the maximum memory usage for a given sequence length.

        Args:
            max_length: Maximum number of tokens in the sequence.

        Returns:
            Total bytes required to cache *max_length* tokens.
        """
        return max_length * self.per_token_bytes()


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """Standard full attention — cache scales O(max_length)."""

    pass


@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    """Sliding window attention specification.

    Cache memory scales with ``O(min(max_length, window_size))`` since only
    the most recent *window_size* tokens are retained.

    Attributes:
        window_size: Number of tokens in the sliding window.
    """

    window_size: int = 4096

    def max_memory_usage_bytes(self, max_length: int) -> int:
        """Return the maximum memory usage, capped by the sliding window.

        Args:
            max_length: Maximum number of tokens in the sequence.

        Returns:
            Total bytes required, using ``min(max_length, window_size)`` as
            the effective cache length.
        """
        effective = min(max_length, self.window_size)
        return effective * self.per_token_bytes()


@dataclass(frozen=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    """Chunked local attention specification.

    Cache memory scales with ``O(min(max_length, chunk_size))`` since
    attention is computed locally within non-overlapping chunks.

    Attributes:
        chunk_size: Number of tokens in each local attention chunk.
    """

    chunk_size: int = 1024

    def max_memory_usage_bytes(self, max_length: int) -> int:
        """Return the maximum memory usage, capped by the chunk size.

        Args:
            max_length: Maximum number of tokens in the sequence.

        Returns:
            Total bytes required, using ``min(max_length, chunk_size)`` as
            the effective cache length.
        """
        effective = min(max_length, self.chunk_size)
        return effective * self.per_token_bytes()


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    """Spec for Mamba / SSM caches (conv state + recurrent state).

    *conv_dim* × *conv_kernel_size* for the rolling convolution buffer,
    plus a fixed-size recurrent state with *state_elements* total scalars.
    """

    conv_dim: int = 0
    conv_kernel_size: int = 4
    state_elements: int = 0

    def max_memory_usage_bytes(self, max_length: int) -> int:
        """Return the memory usage for Mamba/SSM caches.

        Memory is O(1) with respect to sequence length since Mamba caches
        consist of a fixed-size convolution buffer and recurrent state.

        Args:
            max_length: Maximum number of tokens (unused; included for
                interface compatibility).

        Returns:
            Total bytes for convolution buffer plus recurrent state.
        """
        conv_bytes = self.conv_dim * self.conv_kernel_size * self.element_byte_size()
        state_bytes = self.state_elements * self.element_byte_size()
        return conv_bytes + state_bytes


CacheSpec = KVCacheSpec
"""Type alias for :class:`KVCacheSpec`, the root cache specification."""

__all__ = (
    "AttentionSpec",
    "CacheSpec",
    "ChunkedLocalAttentionSpec",
    "FullAttentionSpec",
    "KVCacheSpec",
    "MambaSpec",
    "SlidingWindowSpec",
)
