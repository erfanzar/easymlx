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

"""GLM-4 MoE model family.

This module provides the MLX implementation of the GLM-4 Mixture-of-Experts
model for serving and inference. It exports the configuration class and
model classes needed to load and run GLM-4 MoE models.

Classes:
    Glm4MoeConfig: Configuration for the GLM-4 MoE model.
    Glm4MoeForCausalLM: GLM-4 MoE causal language model with LM head.
    Glm4MoeModel: Base GLM-4 MoE transformer model.
"""

from .glm4_moe_configuration import Glm4MoeConfig
from .modeling_glm4_moe import Glm4MoeForCausalLM, Glm4MoeModel

__all__ = ("Glm4MoeConfig", "Glm4MoeForCausalLM", "Glm4MoeModel")
