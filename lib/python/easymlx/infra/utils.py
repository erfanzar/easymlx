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

"""Infra-level utility re-exports.

Convenience module that re-exports commonly used constants and exception
classes from :mod:`easymlx.infra.errors` and :mod:`easymlx.infra.etils`
so that downstream code can import them from a single location.
"""

from __future__ import annotations

from .errors import EasyMLXConfigError, EasyMLXError, EasyMLXNotImplementedFeatureError, EasyMLXRuntimeError
from .etils import DEFAULT_ATTENTION_MECHANISM, MODEL_CONFIG_NAME, MODEL_WEIGHTS_GLOB, MODEL_WEIGHTS_NAME

__all__ = (
    "DEFAULT_ATTENTION_MECHANISM",
    "MODEL_CONFIG_NAME",
    "MODEL_WEIGHTS_GLOB",
    "MODEL_WEIGHTS_NAME",
    "EasyMLXConfigError",
    "EasyMLXError",
    "EasyMLXNotImplementedFeatureError",
    "EasyMLXRuntimeError",
)
