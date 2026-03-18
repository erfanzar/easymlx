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

"""GLM configuration for serving and inference on MLX.

This module defines the ``GlmConfig`` class that holds all hyperparameters
for GLM models, including attention dimensions, RoPE settings, and
layer configuration.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("glm")
class GlmConfig(EasyMLXBaseConfig):
    """Configuration class for GLM (General Language Model) models.

    Stores all hyperparameters needed to construct a GLM model, including
    architecture dimensions, attention settings, RoPE parameters, and
    layer type specifications.

    Attributes:
        model_type: Identifier string for the model type, always ``"glm"``.
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden representations.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads for queries.
        num_key_value_heads: Number of key-value heads (for GQA).
        partial_rotary_factor: Fraction of head dimensions to apply RoPE to.
        head_dim: Dimensionality of each attention head.
        hidden_act: Activation function name (e.g., ``"silu"``).
        attention_dropout: Dropout rate for attention weights.
        max_position_embeddings: Maximum supported sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether to enable KV caching for generation.
        rope_theta: Base frequency for rotary embeddings.
        attention_bias: Whether attention projections include bias terms.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        layer_types: List specifying the type of each layer.
    """

    model_type = "glm"

    def __init__(
        self,
        *,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        partial_rotary_factor: float = 0.5,
        head_dim: int = 128,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 0.00000015625,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        pad_token_id: int = 151329,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        attention_bias: bool = True,
        layer_types: list["str"] | None = None,
        rope_scaling: dict[str, tp.Any] | None = None,
        **kwargs,
    ):
        """Initialize a GLM configuration.

        Args:
            vocab_size: Size of the vocabulary. Defaults to 151552.
            hidden_size: Dimensionality of hidden representations.
                Defaults to 4096.
            intermediate_size: Dimensionality of the MLP intermediate layer.
                Defaults to 13696.
            num_hidden_layers: Number of transformer decoder layers.
                Defaults to 40.
            num_attention_heads: Number of attention heads. Defaults to 32.
            num_key_value_heads: Number of key-value heads for GQA.
                Defaults to 2.
            partial_rotary_factor: Fraction of head dim to apply RoPE to.
                Defaults to 0.5.
            head_dim: Size of each attention head. Defaults to 128.
            hidden_act: Activation function name. Defaults to ``"silu"``.
            attention_dropout: Dropout rate for attention. Defaults to 0.0.
            max_position_embeddings: Maximum sequence length.
                Defaults to 131072.
            rms_norm_eps: RMS normalization epsilon. Defaults to 1.5625e-7.
            use_cache: Whether to enable KV caching. Defaults to True.
            tie_word_embeddings: Whether to tie input/output embeddings.
                Defaults to False.
            rope_theta: Base frequency for RoPE. Defaults to 10000.0.
            pad_token_id: Padding token ID. Defaults to 151329.
            eos_token_id: End-of-sequence token ID(s). Defaults to
                ``[151329, 151336, 151338]``.
            bos_token_id: Beginning-of-sequence token ID. Defaults to None.
            attention_bias: Whether to use bias in attention projections.
                Defaults to True.
            layer_types: List of layer type strings. Defaults to
                ``["full_attention"]`` for all layers.
            rope_scaling: Optional RoPE scaling configuration.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        if eos_token_id is None:
            eos_token_id = [151329, 151336, 151338]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.head_dim = int(head_dim)
        self.hidden_act = str(hidden_act)
        self.attention_dropout = float(attention_dropout)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.rope_theta = float(rope_theta)
        self.attention_bias = bool(attention_bias)
        self.rope_scaling = rope_scaling

        self.layer_types = layer_types or ["full_attention"] * self.num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rotary_dim(self) -> int:
        """Compute the number of dimensions to apply rotary embeddings to.

        Calculated from ``head_dim * partial_rotary_factor``, rounded down
        to the nearest even number.

        Returns:
            Number of rotary embedding dimensions, guaranteed to be even.
        """
        rotary_dim = int(self.head_dim * float(self.partial_rotary_factor))
        return max(0, min(self.head_dim, rotary_dim - (rotary_dim % 2)))


__all__ = "GlmConfig"
