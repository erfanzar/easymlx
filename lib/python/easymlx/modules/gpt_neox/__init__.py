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

"""GPT-NeoX model family for MLX inference.

This package provides the EasyMLX implementation of the GPT-NeoX
architecture with partial rotary embeddings and parallel residual connections.

Classes:
    GPTNeoXConfig: Configuration for the GPT-NeoX model.
    GPTNeoXModel: Base GPT-NeoX transformer model.
    GPTNeoXForCausalLM: GPT-NeoX causal language model with LM head.
"""

from .gpt_neox_configuration import GPTNeoXConfig
from .modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXModel

__all__ = (
    "GPTNeoXConfig",
    "GPTNeoXForCausalLM",
    "GPTNeoXModel",
)
