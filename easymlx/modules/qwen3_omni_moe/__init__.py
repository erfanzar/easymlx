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

"""Qwen3 Omni MoE model family.

This package provides the MLX implementation of the Qwen3 Omni Mixture-of-Experts
(MoE) multimodal model, including configuration, vision encoder, text decoder
with sparse MoE layers, and conditional generation wrapper.

Classes:
    Qwen3OmniMoeConfig: Top-level configuration for the Qwen3 Omni MoE model.
    Qwen3OmniMoeModel: Base multimodal model combining vision and language.
    Qwen3OmniMoeForConditionalGeneration: Causal language model head on top
        of the base model.
"""

from .modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeModel
from .qwen3_omni_moe_configuration import Qwen3OmniMoeConfig

__all__ = ("Qwen3OmniMoeConfig", "Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeModel")
