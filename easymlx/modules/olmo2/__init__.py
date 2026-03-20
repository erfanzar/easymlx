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

"""OLMo2 model family for MLX inference.

This package provides the EasyMLX implementation of the OLMo2
architecture, which is Llama-like but with Q/K RMSNorm and a distinct
post-attention and post-feedforward normalization pattern.

Classes:
    Olmo2Config: Configuration for the OLMo2 model.
    Olmo2Model: Base OLMo2 transformer model.
    Olmo2ForCausalLM: OLMo2 causal language model with LM head.
"""

from .modeling_olmo2 import Olmo2ForCausalLM, Olmo2Model
from .olmo2_configuration import Olmo2Config

__all__ = (
    "Olmo2Config",
    "Olmo2ForCausalLM",
    "Olmo2Model",
)
