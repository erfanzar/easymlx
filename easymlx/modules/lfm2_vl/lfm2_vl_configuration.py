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

"""LFM2-VL configuration for EasyMLX inference.

LFM2-VL is a vision-language wrapper around LFM2 that strips the vision
tower at load time and only runs the text (LFM2) backbone.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.lfm2 import Lfm2Config


@register_config("lfm2-vl")
class Lfm2VlConfig(EasyMLXBaseConfig):
    """Configuration for the LFM2-VL vision-language model (text-only path).

    LFM2-VL is a vision-language wrapper around LFM2 that strips the vision
    tower at load time and only runs the text backbone. This config holds a
    nested ``Lfm2Config`` for the text model.

    Attributes:
        model_type: Identifier string (``"lfm2-vl"``).
        text_config: Nested ``Lfm2Config`` for the text backbone.
        vocab_size: Vocabulary size (exposed from ``text_config``).
        hidden_size: Hidden dimension (exposed from ``text_config``).
        num_hidden_layers: Number of decoder layers (exposed from ``text_config``).

    Example:
        >>> config = Lfm2VlConfig(text_config={"vocab_size": 1000, "hidden_size": 64})
    """

    model_type = "lfm2-vl"

    def __init__(
        self,
        *,
        text_config: dict[str, tp.Any] | Lfm2Config | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize LFM2-VL configuration.

        Args:
            text_config: Configuration for the text backbone, either as a dict
                or a ``Lfm2Config`` instance. Defaults to an empty ``Lfm2Config``.
            tie_word_embeddings: Whether to tie input/output embeddings.
                Defaults to ``False``.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config.setdefault("tie_word_embeddings", False)
            text_config = Lfm2Config(**text_config)
        self.text_config = text_config

        # Expose text model dimensions for BaseCausalLMModule
        self.vocab_size = int(text_config.vocab_size)
        self.hidden_size = int(text_config.hidden_size)
        self.num_hidden_layers = int(text_config.num_hidden_layers)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("Lfm2VlConfig",)
