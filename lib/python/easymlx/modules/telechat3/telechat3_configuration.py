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

"""Telechat3 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


@register_config("telechat3")
class Telechat3Config(LlamaConfig):
    """Configuration for the Telechat3 transformer model.

    Telechat3 is a standard dense transformer that reuses the Llama
    architecture via ``LlamaConfig``. It defaults to untied word
    embeddings (``tie_word_embeddings=False``). All Llama configuration
    attributes (hidden_size, num_hidden_layers, RoPE, GQA, etc.) are
    inherited.

    Attributes:
        model_type: The model type identifier (``"telechat3"``).

    Example:
        >>> config = Telechat3Config(hidden_size=4096, num_hidden_layers=32)
        >>> config.tie_word_embeddings
        False
    """

    model_type = "telechat3"

    def __init__(self, **kwargs):
        """Initialize a Telechat3 configuration.

        Args:
            **kwargs: All arguments are forwarded to ``LlamaConfig``.
                Defaults ``tie_word_embeddings`` to ``False`` if not
                explicitly provided.
        """
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


__all__ = ("Telechat3Config",)
