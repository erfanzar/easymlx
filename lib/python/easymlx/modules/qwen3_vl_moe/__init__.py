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

"""Qwen3-VL-MoE model family.

This package provides the MLX implementation of the Qwen3 Vision-Language
Mixture-of-Experts model, including configuration, text decoder with sparse
MoE layers, and conditional generation wrapper. The vision encoder is
reused from the ``qwen3_vl`` package.

Classes:
    Qwen3VLMoeConfig: Top-level configuration for the Qwen3-VL-MoE model.
    Qwen3VLMoeModel: Base vision-language model with MoE text decoder.
    Qwen3VLMoeForConditionalGeneration: Causal LM head on top of the base model.
"""

from .modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeModel
from .qwen3_vl_moe_configuration import Qwen3VLMoeConfig

__all__ = ("Qwen3VLMoeConfig", "Qwen3VLMoeForConditionalGeneration", "Qwen3VLMoeModel")
