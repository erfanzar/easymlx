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

"""Jamba configuration for EasyMLX inference."""

from __future__ import annotations

import math

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("jamba")
class JambaConfig(EasyMLXBaseConfig):
    """Configuration for the Jamba hybrid attention/SSM model.

    Jamba interleaves standard transformer attention layers with Mamba
    SSM layers, and optionally uses Mixture-of-Experts (MoE) in select
    feed-forward layers.

    Attributes:
        model_type: Identifier string (``"jamba"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP/MoE intermediate dimensionality.
        num_hidden_layers: Total number of decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        num_experts: Total number of MoE experts.
        num_experts_per_tok: Number of experts activated per token.
        attn_layer_offset: Layer index offset for attention layers.
        attn_layer_period: Period for attention layer placement.
        expert_layer_offset: Layer index offset for MoE layers.
        expert_layer_period: Period for MoE layer placement.
        mamba_d_state: SSM state dimensionality.
        mamba_d_conv: SSM convolution kernel size.
        mamba_expand: SSM expansion factor (intermediate = expand * hidden).
        mamba_dt_rank: Rank for delta projection, or ``"auto"``.
        mamba_proj_bias: Whether Mamba projections use bias.
        mamba_conv_bias: Whether Mamba conv1d uses bias.
        rms_norm_eps: RMSNorm epsilon.
        tie_word_embeddings: Whether to tie input/output embeddings.
        layers_block_type: Pre-computed list of layer types.
    """

    model_type = "jamba"

    def __init__(
        self,
        *,
        vocab_size: int = 65536,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        attn_layer_offset: int = 4,
        attn_layer_period: int = 8,
        expert_layer_offset: int = 1,
        expert_layer_period: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: int | str = "auto",
        mamba_proj_bias: bool = False,
        mamba_conv_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 262144,
        tie_word_embeddings: bool = True,
        layers_block_type: list[str] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize JambaConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: MLP/MoE intermediate dimensionality.
            num_hidden_layers: Total number of decoder layers.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            num_experts: Total number of MoE experts.
            num_experts_per_tok: Experts activated per token.
            attn_layer_offset: Layer index offset for attention placement.
            attn_layer_period: Period for attention layer placement.
            expert_layer_offset: Layer index offset for MoE placement.
            expert_layer_period: Period for MoE layer placement.
            mamba_d_state: SSM state dimensionality.
            mamba_d_conv: SSM convolution kernel size.
            mamba_expand: SSM expansion factor.
            mamba_dt_rank: Rank for delta projection (``"auto"`` = ceil(hidden/16)).
            mamba_proj_bias: Whether Mamba projections use bias.
            mamba_conv_bias: Whether Mamba conv1d uses bias.
            rms_norm_eps: RMSNorm epsilon.
            max_position_embeddings: Maximum sequence length.
            tie_word_embeddings: Whether to tie input/output embeddings.
            layers_block_type: Pre-computed list of layer types. Auto-derived
                if ``None``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.attn_layer_offset = int(attn_layer_offset)
        self.attn_layer_period = int(attn_layer_period)
        self.expert_layer_offset = int(expert_layer_offset)
        self.expert_layer_period = int(expert_layer_period)
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)
        self.mamba_proj_bias = bool(mamba_proj_bias)
        self.mamba_conv_bias = bool(mamba_conv_bias)
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = int(max_position_embeddings)

        if mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.hidden_size / 16)
        else:
            self.mamba_dt_rank = int(mamba_dt_rank)

        if layers_block_type is not None:
            self.layers_block_type = list(layers_block_type)
        else:
            self.layers_block_type = [
                ("attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba")
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("JambaConfig",)
