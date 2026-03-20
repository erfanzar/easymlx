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

"""Bailing MoE Linear model family for MLX inference.

This package provides the EasyMLX implementation of the Bailing MoE Linear
architecture with hybrid linear + full attention layers and MoE feed-forward
blocks, optimized for Apple Silicon via MLX.

Classes:
    BailingMoeLinearConfig: Configuration class for the Bailing MoE Linear model.
    BailingMoeLinearForCausalLM: Bailing MoE Linear model with a causal language modeling head.
    BailingMoeLinearModel: Base Bailing MoE Linear transformer model.
"""

from .bailing_moe_linear_configuration import BailingMoeLinearConfig
from .modeling_bailing_moe_linear import BailingMoeLinearForCausalLM, BailingMoeLinearModel

__all__ = ("BailingMoeLinearConfig", "BailingMoeLinearForCausalLM", "BailingMoeLinearModel")
