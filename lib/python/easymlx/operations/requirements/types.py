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

"""Type definitions for operation requirements.

Defines the core enum types used throughout the operations framework:
execution modes, metadata field flags, and cache type declarations.
"""

from __future__ import annotations

from enum import Enum, Flag, auto


class ExecutionMode(Enum):
    """Execution mode for attention operations.

    Attributes:
        PREFILL: Processing a full prompt sequence.
        DECODE: Autoregressive single-token generation.
        MIXED: Either mode or both (used for requirement declarations).
    """

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class MetadataField(Flag):
    """Flag enum for metadata fields that operations may require or accept.

    Each flag represents a specific piece of metadata (e.g., an attention
    mask, block tables for paged attention). Multiple fields can be combined
    with bitwise OR.

    Attributes:
        NONE: No metadata fields.
        MASK: An attention mask tensor.
        SINKS: Attention sink token logits.
        QUERY_START_LOC: Cumulative query start positions for batched sequences.
        BLOCK_TABLES: Block index tables for paged KV caches.
        KV_LENS: Per-sequence KV lengths.
        BLOCK_SIZE: Number of slots per cache block.
        SLIDING_WINDOW: Sliding window size for local attention.
    """

    NONE = 0
    MASK = auto()
    SINKS = auto()
    QUERY_START_LOC = auto()
    BLOCK_TABLES = auto()
    KV_LENS = auto()
    BLOCK_SIZE = auto()
    SLIDING_WINDOW = auto()

    @classmethod
    def basic(cls) -> MetadataField:
        """Return the basic metadata fields (mask and sinks).

        Returns:
            Combined ``MASK | SINKS`` flags.
        """
        return cls.MASK | cls.SINKS

    @classmethod
    def paged(cls) -> MetadataField:
        """Return all metadata fields needed for paged attention.

        Returns:
            Combined flags for query start locations, block tables,
            KV lengths, block size, and sliding window.
        """
        return cls.QUERY_START_LOC | cls.BLOCK_TABLES | cls.KV_LENS | cls.BLOCK_SIZE | cls.SLIDING_WINDOW


class CacheType(Flag):
    """Flag enum for cache compatibility declarations.

    Operations declare which cache types they support.  The inference engine
    validates compatibility before dispatch.
    """

    NONE = 0
    TRANSFORMER = auto()
    PAGED = auto()
    RECURRENT = auto()
    HYBRID = auto()

    @classmethod
    def any(cls) -> CacheType:
        """All cache types."""
        return cls.TRANSFORMER | cls.PAGED | cls.RECURRENT | cls.HYBRID

    @classmethod
    def attention(cls) -> CacheType:
        """Attention-based caches (transformer + paged)."""
        return cls.TRANSFORMER | cls.PAGED

    def is_compatible_with(self, other: CacheType) -> bool:
        """Check if *self* has any overlap with *other*."""
        return bool(self & other)


__all__ = ("CacheType", "ExecutionMode", "MetadataField")
