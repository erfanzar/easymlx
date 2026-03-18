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

"""Mixins and helper entrypoints mirroring EasyDeL's eSurge layout.

This package aggregates commonly used helper functions and type aliases so
that higher-level code (e.g. the server layer) can import them from a
single ``mixins`` namespace without reaching into individual sub-modules.
"""

from .io import build_chat_prompt
from .lifecycle import close_engine, pause_engine, resume_engine, terminate_engine
from .monitoring import get_engine_metrics_snapshot
from .parsing import extract_reasoning_and_tools
from .requests import EngineRequest, EngineRequestStatus

__all__ = (
    "EngineRequest",
    "EngineRequestStatus",
    "build_chat_prompt",
    "close_engine",
    "extract_reasoning_and_tools",
    "get_engine_metrics_snapshot",
    "pause_engine",
    "resume_engine",
    "terminate_engine",
)
