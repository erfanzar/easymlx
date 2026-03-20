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

"""Nemotron configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("nemotron")
class NemotronConfig(EasyMLXBaseConfig):
    """Configuration for the Nemotron transformer model.

    Nemotron uses a dense architecture with ReLU-squared activation,
    NemotronLayerNorm1P (weights offset by +1), and partial RoPE
    (only applied to first portion of head dimensions).

    Attributes:
        model_type: Identifier string (``"nemotron"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        hidden_act: Activation function name.
        partial_rotary_factor: Fraction of head_dim with RoPE applied.
        norm_eps: LayerNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        max_position_embeddings: Maximum sequence length.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
    """

    model_type = "nemotron"

    def __init__(
        self,
        *,
        vocab_size: int = 256000,
        hidden_size: int = 6144,
        intermediate_size: int = 24576,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 48,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "relu_squared",
        partial_rotary_factor: float = 0.5,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 4096,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Nemotron configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads for GQA.
                Defaults to ``num_attention_heads``.
            head_dim (int | None): Per-head dimensionality. Auto-derived when None.
            hidden_act (str): Activation function name (``"relu_squared"``).
            partial_rotary_factor (float): Fraction of head_dim with RoPE
                applied (e.g. 0.5 means only first half of dims get RoPE).
            norm_eps (float): LayerNorm epsilon.
            rope_theta (float): RoPE base frequency.
            rope_scaling (dict[str, Any] | None): Optional RoPE scaling config.
            max_position_embeddings (int): Maximum sequence length.
            attention_bias (bool): Whether attention projections include bias.
            mlp_bias (bool): Whether MLP projections include bias.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.hidden_act = str(hidden_act)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.norm_eps = float(norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "NemotronConfig"
