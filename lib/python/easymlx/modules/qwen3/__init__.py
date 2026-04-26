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

"""Qwen3 model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen3 architecture,
featuring QK normalization and optional sliding-window attention, optimized
for Apple Silicon via MLX.

Classes:
    Qwen3Config: Configuration class for the Qwen3 model.
    Qwen3ForCausalLM: Qwen3 model with a causal language modeling head.
    Qwen3Model: Base Qwen3 transformer model.
"""

from .modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from .qwen3_configuration import Qwen3Config

__all__ = ("Qwen3Config", "Qwen3ForCausalLM", "Qwen3Model")
