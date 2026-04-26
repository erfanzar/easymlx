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

"""Cohere2 model family for EasyMLX.

This package provides the EasyMLX implementation of the Cohere2 architecture,
extending Cohere with a sliding window attention pattern where every Nth layer
uses full attention.
"""

from .cohere2_configuration import Cohere2Config
from .modeling_cohere2 import Cohere2ForCausalLM, Cohere2Model

__all__ = ("Cohere2Config", "Cohere2ForCausalLM", "Cohere2Model")
