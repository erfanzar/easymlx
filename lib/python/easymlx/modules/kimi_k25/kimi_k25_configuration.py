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

"""Kimi K2.5 configuration for serving and inference.

This module defines the configuration class for the Kimi K2.5 model,
which wraps a DeepSeek V3 text backbone. Registered with the EasyMLX
factory under the ``"kimi_k25"`` model type.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.deepseek_v3.deepseek_v3_configuration import DeepseekV3Config


@register_config("kimi_k25")
class KimiK25Config(EasyMLXBaseConfig):
    """Configuration for the Kimi K2.5 language model.

    Registered with the EasyMLX factory under the ``"kimi_k25"`` model type.
    Wraps a ``DeepseekV3Config`` as the text backbone configuration.

    Attributes:
        model_type: The model type identifier (``"kimi_k25"``).
        text_config: The DeepSeek V3 configuration for the language backbone.
    """

    model_type = "kimi_k25"

    def __init__(
        self,
        *,
        text_config: dict[str, tp.Any] | DeepseekV3Config | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize KimiK25Config.

        Args:
            text_config: Configuration for the DeepSeek V3 text backbone.
                May be a dict, a ``DeepseekV3Config`` instance, or ``None``
                (uses default DeepSeek V3 config).
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
        """
        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)

        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.num_hidden_layers = text_config.num_hidden_layers
        self.num_attention_heads = text_config.num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "KimiK25Config"
