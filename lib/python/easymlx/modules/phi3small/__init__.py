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

"""Phi3Small model family.

This module provides the MLX implementation of the Phi3Small transformer
model family for serving and inference.

Classes:
    Phi3SmallConfig: Configuration for the Phi3Small model.
    Phi3SmallForCausalLM: Phi3Small causal language model with LM head.
    Phi3SmallModel: Base Phi3Small transformer model.
"""

from .modeling_phi3small import Phi3SmallForCausalLM, Phi3SmallModel
from .phi3small_configuration import Phi3SmallConfig

__all__ = ("Phi3SmallConfig", "Phi3SmallForCausalLM", "Phi3SmallModel")
