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

"""AFM7 (Apple Foundation Model 7) for MLX inference.

This package provides the EasyMLX implementation of the AFM7 architecture,
featuring KV-reuse layers where later decoder layers share cached KV states
from the last standard transformer layer.
"""

from .afm7_configuration import Afm7Config
from .modeling_afm7 import Afm7ForCausalLM, Afm7Model

__all__ = ("Afm7Config", "Afm7ForCausalLM", "Afm7Model")
