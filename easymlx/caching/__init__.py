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

"""Caching subsystem for easymlx.

Mirrors EasyDeL's three-tier cache hierarchy:
  BaseCacheConfig / BaseCacheView / BaseCache

Four cache families:
  - **Transformer**: Standard KV cache for auto-regressive generation.
  - **Paged**: Block-table KV cache for serving (unified / paged attention).
  - **Recurrent**: Conv + recurrent state for Mamba / SSM models.
  - **Hybrid**: Mixed attention + recurrent per-layer routing.
"""

from __future__ import annotations

from ._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, OperationsMetadata
from ._metadatabuilder import CacheMetadata
from ._specs import (
    AttentionSpec,
    CacheSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from .hybrid import HybridCache, HybridCacheConfig, HybridCacheView
from .paged import (
    PageCache,
    PageCacheConfig,
    PageCacheView,
    PageMetadata,
    build_query_start_loc,
)
from .recurrent import RecurrentCache, RecurrentCacheConfig, RecurrentCacheView
from .transformer import TransformerCache, TransformerCacheConfig, TransformerCacheView

__all__ = (
    "AttentionSpec",
    "BaseCache",
    "BaseCacheConfig",
    "BaseCacheView",
    "CacheMetadata",
    "CacheSpec",
    "ChunkedLocalAttentionSpec",
    "FullAttentionSpec",
    "HybridCache",
    "HybridCacheConfig",
    "HybridCacheView",
    "KVCacheSpec",
    "MambaSpec",
    "OperationsMetadata",
    "PageCache",
    "PageCacheConfig",
    "PageCacheView",
    "PageCacheView",
    "PageMetadata",
    "PageMetadata",
    "RecurrentCache",
    "RecurrentCacheConfig",
    "RecurrentCacheView",
    "SlidingWindowSpec",
    "TransformerCache",
    "TransformerCacheConfig",
    "TransformerCacheView",
    "build_query_start_loc",
)
