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

"""Qwen3-Next model family for MLX inference.

This package provides the EasyMLX implementation of the Qwen3-Next
architecture, featuring a hybrid full/linear attention mechanism with MoE
routing, gated normalization, and partial rotary embeddings, optimized for
Apple Silicon via MLX.

Classes:
    Qwen3NextConfig: Configuration class for the Qwen3-Next model.
    Qwen3NextForCausalLM: Qwen3-Next model with a causal language modeling head.
    Qwen3NextModel: Base Qwen3-Next transformer model.
"""

from .modeling_qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from .qwen3_next_configuration import Qwen3NextConfig

__all__ = ("Qwen3NextConfig", "Qwen3NextForCausalLM", "Qwen3NextModel")
