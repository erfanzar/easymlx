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

"""Qwen3-VL model family.

This package provides the MLX implementation of the Qwen3 Vision-Language
model, including configuration, vision encoder, text decoder, and
conditional generation wrapper.

Classes:
    Qwen3VLConfig: Top-level configuration for the Qwen3-VL model.
    Qwen3VLModel: Base vision-language model combining vision and text.
    Qwen3VLForConditionalGeneration: Causal language model head on top
        of the base model.
"""

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
from .qwen3_vl_configuration import Qwen3VLConfig

__all__ = ("Qwen3VLConfig", "Qwen3VLForConditionalGeneration", "Qwen3VLModel")
