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

"""Nemotron model family.

This module provides the MLX implementation of the Nemotron transformer
model family for serving and inference.

Classes:
    NemotronConfig: Configuration for the Nemotron model.
    NemotronForCausalLM: Nemotron causal language model with LM head.
    NemotronModel: Base Nemotron transformer model.
"""

from .modeling_nemotron import NemotronForCausalLM, NemotronModel
from .nemotron_configuration import NemotronConfig

__all__ = ("NemotronConfig", "NemotronForCausalLM", "NemotronModel")
