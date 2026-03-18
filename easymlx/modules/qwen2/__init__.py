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

"""Qwen2 model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen2 architecture,
including configuration, base model with optional sliding-window attention,
and causal language model wrapper optimized for Apple Silicon via MLX.

Classes:
    Qwen2Config: Configuration class for the Qwen2 model.
    Qwen2ForCausalLM: Qwen2 model with a causal language modeling head.
    Qwen2Model: Base Qwen2 transformer model.
"""

from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from .qwen2_configuration import Qwen2Config

__all__ = ("Qwen2Config", "Qwen2ForCausalLM", "Qwen2Model")
