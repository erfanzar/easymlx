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

"""Nemotron-H model family for MLX inference.

This package provides the EasyMLX implementation of the Nemotron-H hybrid
architecture, featuring interleaved Mamba2 SSM, attention, MLP, and MoE blocks
controlled by a configurable hybrid override pattern.
"""

from .modeling_nemotron_h import NemotronHForCausalLM, NemotronHModel
from .nemotron_h_configuration import NemotronHConfig

__all__ = ("NemotronHConfig", "NemotronHForCausalLM", "NemotronHModel")
