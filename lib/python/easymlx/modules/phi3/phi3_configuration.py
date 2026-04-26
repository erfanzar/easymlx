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

"""Phi3 configuration (serving/inference only).

This module defines the configuration class for the Phi3 model family.
Phi3 is architecturally very similar to Llama with optional partial RoPE
and rope_scaling support.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra.factory import register_config
from easymlx.modules.llama import LlamaConfig


@register_config("phi3")
class Phi3Config(LlamaConfig):
    """Configuration for the Phi3 transformer model.

    Inherits from LlamaConfig since Phi3 is very similar to Llama
    (SwiGLU MLP, RMSNorm, RoPE). Adds partial_rotary_factor support.

    Attributes:
        model_type: Identifier string (``"phi3"``).
        partial_rotary_factor: Fraction of head_dim to apply RoPE to.
        original_max_position_embeddings: Original max position embeddings
            for SuScaled RoPE.
    """

    model_type = "phi3"

    def __init__(
        self,
        *,
        vocab_size: int = 32064,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        partial_rotary_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Phi3 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: SwiGLU MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA. Defaults to
                ``num_attention_heads`` if None.
            head_dim: Per-head dimensionality. If None, computed as
                ``hidden_size // num_attention_heads``.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: Base frequency for RoPE.
            rope_traditional: Whether to use the traditional RoPE layout.
            rope_scaling: Optional RoPE scaling configuration dict for
                SuScaledRoPE or other scaling methods.
            attention_bias: Whether attention projections use bias.
            mlp_bias: Whether MLP projections use bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            partial_rotary_factor: Fraction of ``head_dim`` to apply RoPE
                to. 1.0 means full rotation.
            original_max_position_embeddings: Original max position
                embeddings for SuScaledRoPE computation.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``LlamaConfig``.
        """
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.original_max_position_embeddings = int(original_max_position_embeddings)

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )


__all__ = "Phi3Config"
