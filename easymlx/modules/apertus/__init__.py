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

"""Apertus model family for EasyMLX.

This package provides the EasyMLX implementation of the Apertus architecture,
featuring XieLU activation, QK-norm attention, and non-gated MLP,
optimized for Apple Silicon via MLX.
"""

from .apertus_configuration import ApertusConfig
from .modeling_apertus import ApertusForCausalLM, ApertusModel

__all__ = ("ApertusConfig", "ApertusForCausalLM", "ApertusModel")
