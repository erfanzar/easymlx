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

"""Exaone configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("exaone")
class ExaoneConfig(EasyMLXBaseConfig):
    """Configuration for the Exaone transformer model.

    Note: Exaone uses ``num_layers`` instead of ``num_hidden_layers`` and
    ``layer_norm_epsilon`` instead of ``rms_norm_eps``, but internally we
    normalize to ``num_hidden_layers`` for consistency with the framework.

    Attributes:
        model_type: Identifier string (``"exaone"``).
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_hidden_layers: Number of decoder layers.
        num_layers: Alias for ``num_hidden_layers``.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        head_dim: Per-head dimension, or ``None``.
        max_position_embeddings: Maximum sequence length.
        layer_norm_epsilon: RMSNorm epsilon (named for upstream compat).
        rope_theta: RoPE base frequency.
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention has bias.
        mlp_bias: Whether MLP has bias.
    """

    model_type = "exaone"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_layers: int | None = None,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        layer_norm_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Exaone configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``102400``.
            hidden_size: Hidden dimension. Defaults to ``2048``.
            intermediate_size: MLP intermediate dimension. Defaults to ``5632``.
            num_layers: Number of layers (upstream naming). Takes priority
                over ``num_hidden_layers`` if provided.
            num_hidden_layers: Number of decoder layers. Defaults to ``32``.
            num_attention_heads: Number of query heads. Defaults to ``32``.
            num_key_value_heads: Number of KV heads, or ``None``.
            head_dim: Per-head dimension, or ``None``.
            max_position_embeddings: Max sequence length. Defaults to ``2048``.
            layer_norm_epsilon: RMSNorm epsilon. Defaults to ``1e-5``.
            rope_theta: RoPE base frequency. Defaults to ``10000.0``.
            rope_traditional: Whether to use traditional RoPE. Defaults to ``False``.
            rope_scaling: Optional RoPE scaling configuration.
            attention_bias: Whether attention has bias. Defaults to ``False``.
            mlp_bias: Whether MLP has bias. Defaults to ``False``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``True``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if num_layers is not None:
            num_hidden_layers = num_layers

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_layers = self.num_hidden_layers
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("ExaoneConfig",)
