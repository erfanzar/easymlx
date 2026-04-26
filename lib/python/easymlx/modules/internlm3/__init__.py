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

"""InternLM3 model family.

This module provides the MLX implementation of the InternLM3 transformer
model family for serving and inference.

Classes:
    InternLM3Config: Configuration for the InternLM3 model.
    InternLM3ForCausalLM: InternLM3 causal language model with LM head.
    InternLM3Model: Base InternLM3 transformer model.
"""

from .internlm3_configuration import InternLM3Config
from .modeling_internlm3 import InternLM3ForCausalLM, InternLM3Model

__all__ = ("InternLM3Config", "InternLM3ForCausalLM", "InternLM3Model")
