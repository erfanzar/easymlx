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

"""eSurge inference engine for easymlx.

Provides the core eSurge engine, API server, configuration objects,
request/output types, page table management, sampling parameters,
metrics collection, monitoring, and dashboard utilities.
"""

from .config import CacheConfig, Config, SchedulerConfig, SpeculativeConfig
from .dashboard import build_dashboard_state
from .esurge_engine import eSurge
from .metrics import MetricsCollector, get_metrics_collector, initialize_metrics, log_metrics_summary
from .monitoring import get_monitoring_snapshot
from .outputs import CompletionOutput, RequestOutput
from .page_table import PageTable, PageTableEntry
from .request import EngineRequest, EngineRequestStatus
from .sampling_params import SamplingParams
from .server import (
    AdminKeyCreateRequest,
    AdminKeyResponse,
    AdminState,
    AuthEndpointsMixin,
    ChatCompletionRequest,
    CompletionRequest,
    MetricsResponse,
    eSurgeApiServer,
)

__all__ = (
    "AdminKeyCreateRequest",
    "AdminKeyResponse",
    "AdminState",
    "AuthEndpointsMixin",
    "CacheConfig",
    "ChatCompletionRequest",
    "CompletionOutput",
    "CompletionRequest",
    "Config",
    "EngineRequest",
    "EngineRequestStatus",
    "MetricsCollector",
    "MetricsResponse",
    "PageTable",
    "PageTableEntry",
    "RequestOutput",
    "SamplingParams",
    "SchedulerConfig",
    "SpeculativeConfig",
    "build_dashboard_state",
    "eSurge",
    "eSurgeApiServer",
    "get_metrics_collector",
    "get_monitoring_snapshot",
    "initialize_metrics",
    "log_metrics_summary",
)
