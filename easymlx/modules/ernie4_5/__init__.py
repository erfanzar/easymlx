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

"""ERNIE 4.5 dense model family for MLX inference.

This package provides the EasyMLX implementation of the ERNIE 4.5 dense
architecture, optimized for Apple Silicon via MLX.

Classes:
    Ernie45Config: Configuration class for the ERNIE 4.5 model.
    Ernie45ForCausalLM: ERNIE 4.5 model with a causal language modeling head.
    Ernie45Model: Base ERNIE 4.5 transformer model.
"""

from .ernie4_5_configuration import Ernie45Config
from .modeling_ernie4_5 import Ernie45ForCausalLM, Ernie45Model

__all__ = ("Ernie45Config", "Ernie45ForCausalLM", "Ernie45Model")
