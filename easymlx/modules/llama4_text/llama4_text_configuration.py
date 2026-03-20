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

"""Llama4 text-only configuration for serving and inference.

This module defines the text-only configuration class for the Llama4 text
decoder stack, registered with the EasyMLX factory under ``"llama4_text"``.
It inherits all parameters from the multimodal ``Llama4TextConfig`` and
automatically calls ``finalize()`` on construction.
"""

from __future__ import annotations

from easymlx.infra.factory import register_config
from easymlx.modules.llama4.llama4_configuration import Llama4TextConfig as _Llama4TextConfig


@register_config("llama4_text")
class Llama4TextConfig(_Llama4TextConfig):
    """Text-only configuration for the Llama4 decoder stack.

    Inherits all parameters from the multimodal ``Llama4TextConfig``
    and automatically resolves derived attributes (``no_rope_layers``,
    ``layer_types``, ``moe_layers``, etc.) during construction.
    Registered with the EasyMLX factory under ``"llama4_text"``.

    Attributes:
        model_type: The model type identifier (``"llama4_text"``).

    Example::

        config = Llama4TextConfig(hidden_size=2048, num_hidden_layers=16)
    """

    model_type = "llama4_text"

    def __init__(self, *args, **kwargs):
        """Initialize the text-only Llama4 configuration.

        Args:
            *args: Positional arguments forwarded to ``Llama4TextConfig``.
            **kwargs: Keyword arguments forwarded to ``Llama4TextConfig``.
                See ``Llama4TextConfig`` for the full parameter list.
        """
        super().__init__(*args, **kwargs)
        self.finalize()


__all__ = ("Llama4TextConfig",)
