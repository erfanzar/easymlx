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

"""GLM-4 configuration for dense text models.

GLM-4 shares the same architecture as GLM but uses a different model type
identifier for registration. This module provides the ``Glm4Config`` class
which inherits all parameters from ``GlmConfig``.
"""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..glm.glm_configuration import GlmConfig


@register_config("glm4")
class Glm4Config(GlmConfig):
    """Configuration class for GLM-4 dense text models.

    Inherits all parameters from ``GlmConfig``. The only difference is
    the ``model_type`` identifier used for model registry lookup.

    Attributes:
        model_type: Identifier string for the model type, always ``"glm4"``.
    """

    model_type = "glm4"


__all__ = "Glm4Config"
