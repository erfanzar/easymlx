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

"""Shared engine-adjacent types matching the EasyDeL eSurge layout.

Re-exports the core public types used across the eSurge subsystem so that
downstream code can import them from a single location.
"""

from __future__ import annotations

from .outputs import CompletionOutput, RequestOutput
from .request import EngineRequest, EngineRequestStatus
from .sampling_params import SamplingParams


class FinishReason:
    """Lightweight string constants for generation finish reasons.

    These replace hardcoded string literals across the eSurge engine,
    scheduler, and related modules.
    """

    EOS = "eos"
    LENGTH = "length"
    STOP = "stop"
    ERROR = "error"
    CANCELED = "canceled"
    TOOL_CALLS = "tool_calls"


__all__ = (
    "CompletionOutput",
    "EngineRequest",
    "EngineRequestStatus",
    "FinishReason",
    "RequestOutput",
    "SamplingParams",
)
