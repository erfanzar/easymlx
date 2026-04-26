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

"""GPT-BigCode model family for MLX inference.

This package provides the EasyMLX implementation of the GPT-BigCode
architecture with multi-query attention and absolute position embeddings.

Classes:
    GPTBigCodeConfig: Configuration for the GPT-BigCode model.
    GPTBigCodeModel: Base GPT-BigCode transformer model.
    GPTBigCodeForCausalLM: GPT-BigCode causal language model with LM head.
"""

from .gpt_bigcode_configuration import GPTBigCodeConfig
from .modeling_gpt_bigcode import GPTBigCodeForCausalLM, GPTBigCodeModel

__all__ = (
    "GPTBigCodeConfig",
    "GPTBigCodeForCausalLM",
    "GPTBigCodeModel",
)
