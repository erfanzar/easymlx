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

"""Server helpers for easymlx eSurge.

This package provides the FastAPI-based OpenAI-compatible API server for the
eSurge inference engine, including request/response models, authentication,
admin key management, and tool execution support.

Lazy imports are used for ``AdminState``, ``AuthEndpointsMixin``, and
``eSurgeApiServer`` to avoid circular dependencies during startup.
"""

from __future__ import annotations

from .api_models import (
    AdminKeyCreateRequest,
    AdminKeyResponse,
    ChatCompletionRequest,
    CompletionRequest,
    MetricsResponse,
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from .tool_registry import ToolRegistry

__all__ = (
    "AdminKeyCreateRequest",
    "AdminKeyResponse",
    "AdminState",
    "AuthEndpointsMixin",
    "ChatCompletionRequest",
    "CompletionRequest",
    "MetricsResponse",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
    "ToolRegistry",
    "eSurgeApiServer",
)


def __getattr__(name: str):
    """Lazily import heavy or circular-dependent symbols on first access.

    Supports lazy loading of ``AdminState``, ``AuthEndpointsMixin``, and
    ``eSurgeApiServer`` to break import cycles between the server and
    worker packages.

    Args:
        name: The attribute name being accessed on this module.

    Returns:
        The requested class or object.

    Raises:
        AttributeError: If *name* is not a recognised lazy attribute.
    """
    if name == "AdminState":
        from easymlx.workers.esurge.admin_state import AdminState

        return AdminState
    if name == "AuthEndpointsMixin":
        from easymlx.workers.esurge.auth_endpoints import AuthEndpointsMixin

        return AuthEndpointsMixin
    if name == "eSurgeApiServer":
        from .api_server import eSurgeApiServer

        return eSurgeApiServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
