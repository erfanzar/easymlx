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

"""Mistral3 configuration for EasyMLX."""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config

from ..llama import LlamaConfig
from ..ministral3 import Ministral3Config


def _as_text_config(
    text_config: Ministral3Config | LlamaConfig | dict[str, Any] | None,
    *,
    tie_word_embeddings: bool,
) -> Ministral3Config | LlamaConfig:
    """Resolve a text config to a ``Ministral3Config`` or ``LlamaConfig``.

    Handles three input forms: an already-instantiated config object, a
    dict (from JSON deserialization), or ``None`` (uses defaults).

    Args:
        text_config: Text model configuration. Can be a config object,
            a dict with a ``model_type`` key, or ``None``.
        tie_word_embeddings: Default value for ``tie_word_embeddings``
            when constructing from a dict.

    Returns:
        A ``Ministral3Config`` or ``LlamaConfig`` instance.

    Raises:
        ValueError: If the dict's ``model_type`` is not ``"ministral3"``
            or ``"llama"``.
    """
    if text_config is None:
        return Ministral3Config(tie_word_embeddings=tie_word_embeddings)
    if isinstance(text_config, (Ministral3Config, LlamaConfig)):
        return text_config

    text_cfg = dict(text_config)
    text_cfg.setdefault("tie_word_embeddings", tie_word_embeddings)
    model_type = str(text_cfg.pop("model_type", "ministral3"))
    if model_type == "ministral3":
        return Ministral3Config(**text_cfg)
    if model_type == "llama":
        return LlamaConfig(**text_cfg)
    raise ValueError(f"Unsupported mistral3 text_config model_type: {model_type!r}")


@register_config("mistral3")
class Mistral3Config(EasyMLXBaseConfig):
    """Configuration for the Mistral3 vision-language model wrapper.

    Mistral3 is a VLM architecture that wraps a Ministral3 (or Llama)
    text backbone. This configuration flattens key text-model attributes
    to the top level for compatibility with the EasyMLX module registry,
    while preserving the full ``text_config`` dict for reconstruction.

    Attributes:
        model_type: The model type identifier (``"mistral3"``).
        text_config: Serialized text model configuration dict.
        vocab_size: Vocabulary size (from text config).
        hidden_size: Hidden dimensionality (from text config).
        intermediate_size: MLP intermediate size (from text config).
        num_hidden_layers: Number of decoder layers (from text config).
        num_attention_heads: Number of query heads (from text config).
        num_key_value_heads: Number of KV heads (from text config).
        head_dim: Per-head dimensionality (from text config).
        max_position_embeddings: Maximum sequence length (from text config).
        rms_norm_eps: RMSNorm epsilon (from text config).
        rope_theta: RoPE base frequency (from text config).
        rope_traditional: RoPE layout (from text config).
        rope_scaling: RoPE scaling config (from text config).
        attention_bias: Attention bias (from text config).
        mlp_bias: MLP bias (from text config).
        layer_types: Per-layer attention types (from text config).
        sliding_window: Sliding window size (from text config).
        tie_word_embeddings: Whether to tie embeddings (from text config).

    Example:
        >>> config = Mistral3Config(text_config={"model_type": "ministral3"})
        >>> config.model_type
        'mistral3'
    """

    model_type = "mistral3"

    def __init__(
        self,
        *,
        text_config: Ministral3Config | LlamaConfig | dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Mistral3 configuration.

        Args:
            text_config: Text backbone configuration. Accepts a config
                object, a dict, or ``None`` (defaults to Ministral3).
            tie_word_embeddings: Default for weight tying when building
                from a dict text config.
            pad_token_id: Padding token ID override.
            bos_token_id: Beginning-of-sequence token ID override.
            eos_token_id: End-of-sequence token ID override.
            **kwargs: Additional arguments forwarded to the base config.
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
        self.layer_types = list(text_cfg.layer_types)
        self.sliding_window = text_cfg.sliding_window
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

    def get_text_config(
        self, decoder: bool | None = None, encoder: bool | None = None
    ) -> Ministral3Config | LlamaConfig:
        """Reconstruct the text backbone configuration from the stored dict.

        Args:
            decoder: Ignored (present for API compatibility).
            encoder: Ignored (present for API compatibility).

        Returns:
            A ``Ministral3Config`` or ``LlamaConfig`` instance.

        Raises:
            ValueError: If the stored ``model_type`` is unsupported.
        """
        del decoder, encoder
        model_type = str(self.text_config.get("model_type", "ministral3"))
        if model_type == "ministral3":
            return Ministral3Config(**self.text_config)
        if model_type == "llama":
            return LlamaConfig(**self.text_config)
        raise ValueError(f"Unsupported mistral3 text_config model_type: {model_type!r}")


__all__ = ("Mistral3Config",)
