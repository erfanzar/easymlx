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

"""Helium configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


@register_config("helium")
class HeliumConfig(LlamaConfig):
    """Configuration for the Helium transformer model.

    Helium extends the Llama architecture with traditional (interleaved) RoPE
    and a bias-free output projection in the attention layer. All other
    hyperparameters are inherited from ``LlamaConfig``.

    Attributes:
        model_type: Identifier string (``"helium"``).
        rope_traditional: Always ``True`` for Helium (traditional RoPE layout).
        tie_word_embeddings: Defaults to ``False`` (separate LM head).

    Example:
        >>> config = HeliumConfig(
        ...     vocab_size=32000,
        ...     hidden_size=2048,
        ...     num_hidden_layers=16,
        ...     num_attention_heads=16,
        ... )
        >>> config.model_type
        'helium'
        >>> config.rope_traditional
        True
    """

    model_type = "helium"

    def __init__(self, **kwargs):
        """Initialize HeliumConfig.

        Args:
            **kwargs: All keyword arguments are forwarded to ``LlamaConfig``.
                Defaults ``rope_traditional`` to ``True`` and
                ``tie_word_embeddings`` to ``False`` if not provided.
        """
        kwargs.setdefault("rope_traditional", True)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


__all__ = ("HeliumConfig",)
