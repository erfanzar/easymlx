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

"""GLM-4V model family.

This module provides the MLX implementation of the GLM-4V vision-language
model for serving and inference. It exports configuration and model classes
along with lowercase aliases for backward compatibility.

Classes:
    Glm4VConfig: Configuration for the GLM-4V model.
    Glm4VForConditionalGeneration: GLM-4V causal language model wrapper.
    Glm4VModel: Base GLM-4V vision-language model.
    Glm4vConfig: Alias for ``Glm4VConfig``.
    Glm4vForConditionalGeneration: Alias for ``Glm4VForConditionalGeneration``.
    Glm4vModel: Alias for ``Glm4VModel``.
"""

from .glm4v_configuration import Glm4VConfig, Glm4vConfig
from .modeling_glm4v import Glm4VForConditionalGeneration, Glm4VModel

Glm4vModel = Glm4VModel
Glm4vForConditionalGeneration = Glm4VForConditionalGeneration

__all__ = (
    "Glm4VConfig",
    "Glm4VForConditionalGeneration",
    "Glm4VModel",
    "Glm4vConfig",
    "Glm4vForConditionalGeneration",
    "Glm4vModel",
)
