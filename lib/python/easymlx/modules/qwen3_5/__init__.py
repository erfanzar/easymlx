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

"""Qwen3.5 model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen3.5
vision-language architecture, featuring a hybrid full/linear attention
text backbone (from Qwen3-Next) combined with a Qwen3-VL vision encoder,
optimized for Apple Silicon via MLX.

Classes:
    Qwen3_5Config: Top-level configuration for the Qwen3.5 multimodal model.
    Qwen3_5TextConfig: Configuration for the Qwen3.5 text backbone.
    Qwen3_5VisionConfig: Configuration for the Qwen3.5 vision encoder.
    Qwen3_5ForCausalLM: Text-only causal language model.
    Qwen3_5ForConditionalGeneration: Multimodal conditional generation model.
    Qwen3_5Model: Base multimodal (vision-language) model.
    Qwen3_5TextModel: Base text-only model.
"""

from .modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5TextModel,
)
from .qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

__all__ = (
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
    "Qwen3_5VisionConfig",
)
