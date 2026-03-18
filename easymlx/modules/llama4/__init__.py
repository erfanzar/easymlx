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

"""Llama4 model family for MLX inference.

This package provides the EasyMLX implementation of the Llama4 multimodal
model architecture, including configuration, vision encoder, text decoder,
and conditional generation modules optimized for Apple Silicon via MLX.

Classes:
    Llama4Config: Configuration dataclass for the Llama4 model.
    Llama4ForConditionalGeneration: Llama4 model with a language modeling head
        for conditional (vision+language) generation.
    Llama4Model: Base Llama4 multimodal model combining vision and language towers.
"""

from .llama4_configuration import Llama4Config
from .modeling_llama4 import Llama4ForConditionalGeneration, Llama4Model

__all__ = ("Llama4Config", "Llama4ForConditionalGeneration", "Llama4Model")
