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

"""GLM-4 MLX implementation for dense text models.

This module provides thin aliases over the GLM implementation, registered
under the ``"glm4"`` model type. The GLM-4 dense architecture is identical
to GLM; only the model type and configuration class differ.
"""

from __future__ import annotations

from easymlx.infra import TaskType
from easymlx.infra.factory import register_module

from ..glm.modeling_glm import GlmForCausalLM as _GlmForCausalLM
from ..glm.modeling_glm import GlmModel as _GlmModel
from .glm4_configuration import Glm4Config


@register_module(task_type=TaskType.BASE_MODULE, config=Glm4Config, model_type="glm4")
class Glm4Model(_GlmModel):
    """GLM-4 base transformer model (decoder-only).

    Thin alias over ``GlmModel`` registered under the ``"glm4"`` model type.

    Attributes:
        config_class: The configuration class (``Glm4Config``).
    """

    config_class = Glm4Config


@register_module(task_type=TaskType.CAUSAL_LM, config=Glm4Config, model_type="glm4")
class Glm4ForCausalLM(_GlmForCausalLM):
    """GLM-4 model with a causal language modeling head.

    Thin alias over ``GlmForCausalLM`` registered under the ``"glm4"`` model type.

    Attributes:
        config_class: The configuration class (``Glm4Config``).
    """

    config_class = Glm4Config


__all__ = ("Glm4ForCausalLM", "Glm4Model")
