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

"""Phi configuration (serving/inference only).

This module defines the configuration class for the Phi model family.
Phi uses parallel residuals, partial RoPE, LayerNorm, and GELU approx
activation, which makes it architecturally distinct from Llama.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("phi")
class PhiConfig(EasyMLXBaseConfig):
    """Configuration for the Phi transformer model.

    Inherits from EasyMLXBaseConfig since Phi has a unique architecture
    with parallel residuals, partial RoPE, LayerNorm, and biased projections.

    Attributes:
        model_type: Identifier string (``"phi"``).
        partial_rotary_factor: Fraction of head_dim to apply RoPE to.
        layer_norm_eps: LayerNorm epsilon value.
    """

    model_type = "phi"

    def __init__(
        self,
        *,
        vocab_size: int = 51200,
        hidden_size: int = 2560,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 2048,
        partial_rotary_factor: float = 0.4,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Phi configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA. Defaults to
                ``num_attention_heads`` (MHA) if None.
            max_position_embeddings: Maximum sequence length.
            partial_rotary_factor: Fraction of ``head_dim`` to which RoPE
                is applied (e.g. 0.4 means 40% of dims get rotation).
            layer_norm_eps: Epsilon for LayerNorm.
            rope_theta: Base frequency for rotary positional embeddings.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.layer_norm_eps = float(layer_norm_eps)
        self.rope_theta = float(rope_theta)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "PhiConfig"
