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

"""OLMo2 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


@register_config("olmo2")
class Olmo2Config(LlamaConfig):
    """Configuration for the OLMo2 transformer model.

    OLMo2 is Llama-like but with Q/K RMSNorm on each attention layer and
    a post-attention norm + post-feedforward norm pattern (4 norms per layer).

    Inherits all Llama parameters and defaults:
        - attention_bias=False
        - mlp_bias=False
        - tie_word_embeddings=True
    """

    model_type = "olmo2"

    def __init__(self, **kwargs):
        """Initialize OLMo2 configuration.

        Sets defaults specific to OLMo2: ``attention_bias=False``,
        ``mlp_bias=False``, ``tie_word_embeddings=True``.

        Args:
            **kwargs: All keyword arguments are forwarded to
                ``LlamaConfig.__init__``. See ``LlamaConfig`` for the full
                parameter list.
        """
        kwargs.setdefault("attention_bias", False)
        kwargs.setdefault("mlp_bias", False)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)


__all__ = ("Olmo2Config",)
