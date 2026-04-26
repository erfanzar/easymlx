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

"""Minimal inference engine interface and API-server contract.

This module defines the core abstractions for inference engines in easymlx:

- ``InferenceEngineAdapter``: A Protocol that any inference engine must satisfy
  to be served over the API.
- ``EndpointConfig``: A simple dataclass describing an HTTP endpoint.
- ``create_error_response``: A helper for building standardized error payloads.

It also re-exports ``eSurge`` and ``eSurgeApiServer`` for convenience, and
provides ``BaseInferenceApiServer`` as an alias for the default server class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .esurge import eSurge, eSurgeApiServer


class InferenceEngineAdapter(Protocol):
    """Protocol for engine adapters exposed over APIs.

    Any object that implements this protocol can be used as a backend
    engine with the eSurge API server.
    """

    def generate(self, prompts: str | list["str"], sampling_params: Any, stream: bool = False) -> Any:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single prompt string or a list of prompt strings.
            sampling_params: Sampling parameters controlling generation
                (temperature, top_p, top_k, max_tokens, etc.).
            stream: If ``True``, return a streaming iterator of partial
                outputs instead of waiting for the full completion.

        Returns:
            A list of ``RequestOutput`` objects (non-streaming) or an
            iterator of ``RequestOutput`` objects (streaming).
        """
        ...


@dataclass
class EndpointConfig:
    """Configuration for a single HTTP API endpoint.

    Attributes:
        path: The URL path for the endpoint (e.g. ``"/v1/completions"``).
        handler: The callable that handles requests to this endpoint.
        methods: HTTP methods this endpoint responds to (e.g. ``["POST"]``).
    """

    path: str
    handler: Any
    methods: list["str"]


BaseInferenceApiServer = eSurgeApiServer
ServerStatus = str


def create_error_response(message: str, *, error_type: str = "error") -> dict[str, Any]:
    """Build a standardized OpenAI-style error response payload.

    Args:
        message: Human-readable error description.
        error_type: Error category string (e.g. ``"error"``,
            ``"invalid_request_error"``).

    Returns:
        A dictionary with an ``"error"`` key containing ``"message"``
        and ``"type"`` fields.
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
        }
    }


__all__ = (
    "BaseInferenceApiServer",
    "EndpointConfig",
    "InferenceEngineAdapter",
    "ServerStatus",
    "create_error_response",
    "eSurge",
    "eSurgeApiServer",
)
