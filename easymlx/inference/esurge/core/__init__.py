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

"""Core runtime building blocks for eSurge.

Re-exports coordination state, data-parallel sharding, cache management
interfaces and implementations, page pools, samplers, and sampling metadata.
"""

from .coordinator import ExecutionCoordinatorState
from .dp_sharding import DataParallelSharding, dp_shard_for_page_id, pages_per_dp_shard
from .interface import (
    CacheAcquireResult,
    CacheGroupPolicy,
    CacheGroupsConfig,
    CacheManagerProtocol,
    InferenceEngineProtocol,
)
from .manager import CacheManager, EngineManagerState, PageRecord
from .page_pool import PagePool, PageState
from .sampler import argmax_token, sampling_params_to_kwargs
from .sampling_metadata import SamplingMetadata
from .single_type_cache_manager import CacheItem, SingleTypeCacheManager

__all__ = (
    "CacheAcquireResult",
    "CacheGroupPolicy",
    "CacheGroupsConfig",
    "CacheItem",
    "CacheManager",
    "CacheManagerProtocol",
    "DataParallelSharding",
    "EngineManagerState",
    "ExecutionCoordinatorState",
    "InferenceEngineProtocol",
    "PagePool",
    "PageRecord",
    "PageState",
    "SamplingMetadata",
    "SingleTypeCacheManager",
    "argmax_token",
    "dp_shard_for_page_id",
    "pages_per_dp_shard",
    "sampling_params_to_kwargs",
)
