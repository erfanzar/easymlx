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

"""Pixtral configuration for EasyMLX."""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


def _as_text_config(text_config: LlamaConfig | dict[str, Any] | None, *, tie_word_embeddings: bool) -> LlamaConfig:
    """Coerce a text config argument into a resolved ``LlamaConfig``.

    Args:
        text_config: A ``LlamaConfig`` instance, a plain dict of keyword
            arguments, or None for a default config.
        tie_word_embeddings: Default tie-word-embeddings flag applied when
            creating from None or when the dict does not specify it.

    Returns:
        Resolved ``LlamaConfig`` instance.
    """
    if text_config is None:
        return LlamaConfig(tie_word_embeddings=tie_word_embeddings)
    if isinstance(text_config, LlamaConfig):
        return text_config
    text_cfg = dict(text_config)
    text_cfg.setdefault("tie_word_embeddings", tie_word_embeddings)
    return LlamaConfig(**text_cfg)


@register_config("pixtral")
class PixtralConfig(EasyMLXBaseConfig):
    """Compatibility-first Pixtral config that wraps a Llama text stack.

    Pixtral is a vision-language model. This config handles the text
    backbone by delegating to a ``LlamaConfig``. Vision-specific
    parameters are not included because this EasyMLX implementation
    only supports the text generation path.

    Attributes:
        model_type: Identifier string (``"pixtral"``).
        text_config: Serialized text configuration dict.
        vocab_size: Size of the token vocabulary (from text config).
        hidden_size: Dimensionality of hidden states (from text config).

    Example:
        >>> config = PixtralConfig(text_config={"hidden_size": 4096})
    """

    model_type = "pixtral"

    def __init__(
        self,
        *,
        text_config: LlamaConfig | dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Pixtral configuration.

        Args:
            text_config: Text backbone configuration as a ``LlamaConfig``,
                a dict of its parameters, or None for defaults.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            bos_token_id: Token ID for beginning of sequence.
            eos_token_id: Token ID for end of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        text_cfg = _as_text_config(text_config, tie_word_embeddings=tie_word_embeddings)

        self.text_config = text_cfg.to_dict()
        self.vocab_size = int(text_cfg.vocab_size)
        self.hidden_size = int(text_cfg.hidden_size)
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.num_hidden_layers = int(text_cfg.num_hidden_layers)
        self.num_attention_heads = int(text_cfg.num_attention_heads)
        self.num_key_value_heads = int(text_cfg.num_key_value_heads)
        self.head_dim = int(text_cfg.head_dim or (text_cfg.hidden_size // text_cfg.num_attention_heads))
        self.max_position_embeddings = int(text_cfg.max_position_embeddings)
        self.rms_norm_eps = float(text_cfg.rms_norm_eps)
        self.rope_theta = float(text_cfg.rope_theta)
        self.rope_traditional = bool(text_cfg.rope_traditional)
        self.rope_scaling = text_cfg.rope_scaling
        self.attention_bias = bool(text_cfg.attention_bias)
        self.mlp_bias = bool(text_cfg.mlp_bias)
        self.tie_word_embeddings = bool(getattr(text_cfg, "tie_word_embeddings", tie_word_embeddings))

        if pad_token_id is None:
            pad_token_id = text_cfg.pad_token_id
        if bos_token_id is None:
            bos_token_id = text_cfg.bos_token_id
        if eos_token_id is None:
            eos_token_id = text_cfg.eos_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> LlamaConfig:
        """Return the resolved Llama text configuration.

        The extra arguments are accepted for API compatibility with the
        nested text-config helpers used by other multimodal families.
        """
        del decoder, encoder
        return LlamaConfig(**self.text_config)


__all__ = ("PixtralConfig",)
