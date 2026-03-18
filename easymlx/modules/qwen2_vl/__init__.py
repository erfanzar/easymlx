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

"""Qwen2-VL (Vision-Language) model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen2-VL multimodal
architecture, combining a vision encoder with a text decoder for image and
video understanding tasks, optimized for Apple Silicon via MLX.

Classes:
    Qwen2VLConfig: Configuration class for the Qwen2-VL model.
    Qwen2VLForConditionalGeneration: Qwen2-VL model with a language modeling
        head for conditional generation.
    Qwen2VLModel: Base Qwen2-VL multimodal model.
"""

from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel
from .qwen2_vl_configuration import Qwen2VLConfig

__all__ = ("Qwen2VLConfig", "Qwen2VLForConditionalGeneration", "Qwen2VLModel")
