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

"""Kimi K2.5 model family for MLX inference.

This package provides the EasyMLX implementation of the Kimi K2.5 architecture,
which wraps a DeepSeek V3 backbone, optimized for Apple Silicon via MLX.

Classes:
    KimiK25Config: Configuration class for the Kimi K2.5 model.
    KimiK25ForCausalLM: Kimi K2.5 model with a causal language modeling head.
    KimiK25Model: Base Kimi K2.5 transformer model.
"""

from .kimi_k25_configuration import KimiK25Config
from .modeling_kimi_k25 import KimiK25ForCausalLM, KimiK25Model

__all__ = ("KimiK25Config", "KimiK25ForCausalLM", "KimiK25Model")
