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

"""Llama4 text-only model family for MLX inference.

This package provides a text-only EasyMLX wrapper around the Llama4 text
decoder stack, exposed under the HuggingFace ``llama4_text`` model type.
It supports MoE routing, chunked attention, and no-RoPE layer intervals
inherited from the full Llama4 architecture.

Classes:
    Llama4TextConfig: Text-only configuration for the Llama4 decoder stack.
    Llama4TextForCausalLM: Llama4 text-only model with a causal language
        modeling head.
    Llama4TextModel: Base Llama4 text-only transformer model.
"""

from .llama4_text_configuration import Llama4TextConfig
from .modeling_llama4_text import Llama4TextForCausalLM, Llama4TextModel

__all__ = ("Llama4TextConfig", "Llama4TextForCausalLM", "Llama4TextModel")
