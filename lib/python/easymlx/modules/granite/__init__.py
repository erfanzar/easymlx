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

"""Granite model family.

This module provides the MLX implementation of the Granite transformer
model family for serving and inference.

Classes:
    GraniteConfig: Configuration for the Granite model.
    GraniteForCausalLM: Granite causal language model with LM head.
    GraniteModel: Base Granite transformer model.
"""

from .granite_configuration import GraniteConfig
from .modeling_granite import GraniteForCausalLM, GraniteModel

__all__ = ("GraniteConfig", "GraniteForCausalLM", "GraniteModel")
