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

"""GLM-4V MoE model family.

This module provides the MLX implementation of the GLM-4V MoE
vision-language model for serving and inference. It combines a vision
encoder with a Mixture-of-Experts language backbone.

Classes:
    Glm4VMoeConfig: Configuration for the GLM-4V MoE model.
    Glm4VMoeForConditionalGeneration: GLM-4V MoE causal LM wrapper.
    Glm4VMoeModel: Base GLM-4V MoE vision-language model.
    Glm4vMoeConfig: Alias for ``Glm4VMoeConfig``.
    Glm4vMoeForConditionalGeneration: Alias for ``Glm4VMoeForConditionalGeneration``.
    Glm4vMoeModel: Alias for ``Glm4VMoeModel``.
"""

from .glm4v_moe_configuration import Glm4VMoeConfig, Glm4vMoeConfig
from .modeling_glm4v_moe import Glm4VMoeForConditionalGeneration, Glm4VMoeModel

Glm4vMoeModel = Glm4VMoeModel
Glm4vMoeForConditionalGeneration = Glm4VMoeForConditionalGeneration

__all__ = (
    "Glm4VMoeConfig",
    "Glm4VMoeForConditionalGeneration",
    "Glm4VMoeModel",
    "Glm4vMoeConfig",
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",
)
