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

"""Gemma3 configuration (serving/inference only).

Gemma3 is a VLM wrapper that delegates text generation to a Gemma3Text backbone.
Only the text portion is supported for inference; vision components are stripped.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config

from ..gemma3_text import Gemma3TextConfig


def _as_text_config(
    text_config: Gemma3TextConfig | dict[str, Any] | None,
    *,
    vocab_size: int,
) -> Gemma3TextConfig:
    """Coerce a text config value into a ``Gemma3TextConfig`` instance.

    Args:
        text_config: A ``Gemma3TextConfig`` instance, a dict of config
            parameters, or None for defaults.
        vocab_size: Vocabulary size to propagate if not present in
            ``text_config``.

    Returns:
        A fully initialized ``Gemma3TextConfig``.
    """
    if text_config is None:
        return Gemma3TextConfig(vocab_size=vocab_size)
    if isinstance(text_config, Gemma3TextConfig):
        return text_config
    text_cfg = dict(text_config)
    text_cfg.setdefault("vocab_size", vocab_size)
    text_cfg.setdefault("num_attention_heads", text_cfg.get("num_attention_heads", 8))
    text_cfg.setdefault("num_key_value_heads", text_cfg.get("num_key_value_heads", 4))
    text_cfg.pop("model_type", None)
    return Gemma3TextConfig(**text_cfg)


@register_config("gemma3")
class Gemma3Config(EasyMLXBaseConfig):
    """Configuration for the Gemma3 VLM wrapper model.

    Wraps a ``Gemma3TextConfig`` for the text backbone. Vision components
    are stripped at load time; this configuration delegates all text
    parameters to the inner text config, including sliding window
    attention patterns and ``query_pre_attn_scalar``.
    Registered as model type ``"gemma3"``.

    Attributes:
        model_type: Identifier string (``"gemma3"``).
        text_config: Serialized dictionary of the text backbone config.
        vocab_size: Vocabulary size (propagated from text config).
        sliding_window: Sliding window size for local attention layers.
        sliding_window_pattern: Period of full-attention layers (every Nth
            layer is full attention, the rest use sliding window).
        query_pre_attn_scalar: Pre-attention scalar for query scaling.

    Args:
        text_config: Text backbone configuration as a ``Gemma3TextConfig``
            instance, dict, or None for defaults.
        vocab_size: Vocabulary size. Defaults to 262208.
        tie_word_embeddings: Whether to tie embeddings. Defaults to True.
        pad_token_id: Padding token id. Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.

    Example::

        >>> config = Gemma3Config(vocab_size=262208)
        >>> config.model_type
        'gemma3'
        >>> config.get_text_config().sliding_window
        512
    """

    model_type = "gemma3"

    def __init__(
        self,
        *,
        text_config: Gemma3TextConfig | dict[str, Any] | None = None,
        vocab_size: int = 262208,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        text_cfg = _as_text_config(text_config, vocab_size=vocab_size)
        self.text_config = text_cfg.to_dict()
        self.vocab_size = int(text_cfg.vocab_size)
        self.hidden_size = int(text_cfg.hidden_size)
        self.intermediate_size = int(text_cfg.intermediate_size)
        self.num_hidden_layers = int(text_cfg.num_hidden_layers)
        self.num_attention_heads = int(text_cfg.num_attention_heads)
        self.num_key_value_heads = int(text_cfg.num_key_value_heads)
        self.head_dim = text_cfg.head_dim
        self.max_position_embeddings = int(text_cfg.max_position_embeddings)
        self.rms_norm_eps = float(text_cfg.rms_norm_eps)
        self.rope_theta = float(text_cfg.rope_theta)
        self.rope_traditional = bool(text_cfg.rope_traditional)
        self.rope_scaling = text_cfg.rope_scaling
        self.attention_bias = bool(text_cfg.attention_bias)
        self.mlp_bias = bool(text_cfg.mlp_bias)
        self.query_pre_attn_scalar = float(text_cfg.query_pre_attn_scalar)
        self.sliding_window = int(text_cfg.sliding_window)
        self.sliding_window_pattern = int(text_cfg.sliding_window_pattern)
        self.tie_word_embeddings = bool(getattr(text_cfg, "tie_word_embeddings", tie_word_embeddings))

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> Gemma3TextConfig:
        """Return the text backbone configuration as a ``Gemma3TextConfig``.

        Args:
            decoder: Unused, kept for API compatibility.
            encoder: Unused, kept for API compatibility.

        Returns:
            A ``Gemma3TextConfig`` reconstructed from the stored dict.
        """
        del decoder, encoder
        cfg = dict(self.text_config)
        cfg.pop("model_type", None)
        return Gemma3TextConfig(**cfg)


__all__ = ("Gemma3Config",)
