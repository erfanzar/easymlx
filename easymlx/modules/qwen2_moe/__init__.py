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

"""Qwen2 Mixture-of-Experts model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen2-MoE architecture,
featuring sparse expert routing alongside optional sliding-window attention,
optimized for Apple Silicon via MLX.

Classes:
    Qwen2MoeConfig: Configuration class for the Qwen2-MoE model.
    Qwen2MoeForCausalLM: Qwen2-MoE model with a causal language modeling head.
    Qwen2MoeModel: Base Qwen2-MoE transformer model.
"""

from .modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeModel
from .qwen2_moe_configuration import Qwen2MoeConfig

__all__ = ("Qwen2MoeConfig", "Qwen2MoeForCausalLM", "Qwen2MoeModel")
