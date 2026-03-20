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

"""Falcon-H1 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("falcon_h1")
class FalconH1Config(EasyMLXBaseConfig):
    """Configuration for the Falcon-H1 hybrid Attention+Mamba+MLP model.

    Attributes:
        model_type: Identifier string (``"falcon_h1"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        num_hidden_layers: Number of decoder layers.
        intermediate_size: MLP intermediate dimensionality.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: Optional RoPE scaling configuration.
        max_position_embeddings: Maximum sequence length.
        mamba_d_state: Mamba SSM state dimension.
        mamba_d_conv: Mamba convolution kernel size.
        mamba_expand: Mamba expansion factor.
        mamba_n_groups: Mamba number of groups.
        mamba_n_heads: Mamba number of heads.
        mamba_d_ssm: Mamba SSM hidden dimension.
        mamba_d_head: Mamba per-head dimension.
        mamba_chunk_size: Mamba chunk size for processing.
        mamba_conv_bias: Whether Mamba conv1d has bias.
        mamba_proj_bias: Whether Mamba projections have bias.
        mamba_rms_norm: Whether Mamba uses RMSNorm gating.
        mamba_norm_before_gate: Whether to apply norm before gate.
        mamba_use_mlp: Whether to include MLP block.
        attention_bias: Whether attention includes bias.
        attention_in_multiplier: MuP scaling for attention input.
        attention_out_multiplier: MuP scaling for attention output.
        key_multiplier: MuP scaling for key projection.
        ssm_in_multiplier: MuP scaling for SSM input.
        ssm_out_multiplier: MuP scaling for SSM output.
        ssm_multipliers: Per-component SSM scaling factors.
        embedding_multiplier: MuP scaling for embeddings.
        lm_head_multiplier: MuP scaling for LM head.
        mlp_multipliers: MuP scaling for MLP gate/down projections.
        mlp_bias: Whether MLP includes bias.
        mlp_expansion_factor: MLP expansion factor.
        projectors_bias: Whether Mamba output projector has bias.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "falcon_h1"

    def __init__(
        self,
        *,
        vocab_size: int = 32784,
        hidden_size: int = 1024,
        num_hidden_layers: int = 36,
        intermediate_size: int = 2048,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = None,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000000000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 131072,
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_groups: int = 1,
        mamba_n_heads: int = 24,
        mamba_d_ssm: int = 1536,
        mamba_d_head: int = 64,
        mamba_chunk_size: int = 128,
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mamba_rms_norm: bool = False,
        mamba_norm_before_gate: bool = False,
        mamba_use_mlp: bool = True,
        attention_bias: bool = False,
        attention_in_multiplier: float = 1.0,
        attention_out_multiplier: float = 0.9375,
        key_multiplier: float = 0.390625,
        ssm_in_multiplier: float = 1.25,
        ssm_out_multiplier: float = 0.23570226039551587,
        ssm_multipliers: list[float] | None = None,
        embedding_multiplier: float = 5.656854249492381,
        lm_head_multiplier: float = 0.0390625,
        mlp_multipliers: list[float] | None = None,
        mlp_bias: bool = False,
        mlp_expansion_factor: int = 8,
        projectors_bias: bool = False,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize Falcon-H1 configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``32784``.
            hidden_size: Hidden dimension. Defaults to ``1024``.
            num_hidden_layers: Number of hybrid decoder layers. Defaults to ``36``.
            intermediate_size: MLP intermediate dimension. Defaults to ``2048``.
            num_attention_heads: Number of attention heads. Defaults to ``8``.
            num_key_value_heads: Number of KV heads, or ``None``.
            head_dim: Per-head dimension. Defaults to ``64``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-5``.
            rope_theta: RoPE base frequency. Defaults to ``100000000000.0``.
            rope_traditional: Whether to use traditional RoPE. Defaults to ``False``.
            rope_scaling: Optional RoPE scaling configuration.
            max_position_embeddings: Max sequence length. Defaults to ``131072``.
            mamba_d_state: Mamba SSM state dimension. Defaults to ``128``.
            mamba_d_conv: Mamba convolution kernel size. Defaults to ``4``.
            mamba_expand: Mamba expansion factor. Defaults to ``2``.
            mamba_n_groups: Mamba number of groups. Defaults to ``1``.
            mamba_n_heads: Mamba number of heads. Defaults to ``24``.
            mamba_d_ssm: Mamba SSM hidden dimension. Defaults to ``1536``.
            mamba_d_head: Mamba per-head dimension. Defaults to ``64``.
            mamba_chunk_size: Mamba chunk size for processing. Defaults to ``128``.
            mamba_conv_bias: Whether Mamba conv1d has bias. Defaults to ``True``.
            mamba_proj_bias: Whether Mamba projections have bias. Defaults to ``False``.
            mamba_rms_norm: Whether Mamba uses RMSNorm gating. Defaults to ``False``.
            mamba_norm_before_gate: Whether to norm before gating. Defaults to ``False``.
            mamba_use_mlp: Whether to include MLP block. Defaults to ``True``.
            attention_bias: Whether attention has bias. Defaults to ``False``.
            attention_in_multiplier: MuP scaling for attention input projections.
                Defaults to ``1.0``.
            attention_out_multiplier: MuP scaling for attention output projection.
                Defaults to ``0.9375``.
            key_multiplier: MuP scaling for key projection. Defaults to ``0.390625``.
            ssm_in_multiplier: MuP scaling for SSM input projection.
                Defaults to ``1.25``.
            ssm_out_multiplier: MuP scaling for SSM output projection.
                Defaults to ``0.2357...``.
            ssm_multipliers: Per-component SSM scaling factors (5 values for
                x, z, B, C, dt). Uses defaults if ``None``.
            embedding_multiplier: MuP scaling for embeddings. Defaults to ``5.657...``.
            lm_head_multiplier: MuP scaling for LM head. Defaults to ``0.0390625``.
            mlp_multipliers: MuP scaling for MLP [gate, down] projections.
                Uses defaults if ``None``.
            mlp_bias: Whether MLP has bias. Defaults to ``False``.
            mlp_expansion_factor: MLP expansion factor. Defaults to ``8``.
            projectors_bias: Whether Mamba output projector has bias. Defaults to ``False``.
            initializer_range: Weight initializer range. Defaults to ``0.02``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``True``.
            **kwargs: Forwarded to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.intermediate_size = int(intermediate_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)

        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)
        self.mamba_n_groups = int(mamba_n_groups)
        self.mamba_n_heads = int(mamba_n_heads)
        self.mamba_d_ssm = int(mamba_d_ssm)
        self.mamba_d_head = int(mamba_d_head)
        self.mamba_chunk_size = int(mamba_chunk_size)
        self.mamba_conv_bias = bool(mamba_conv_bias)
        self.mamba_proj_bias = bool(mamba_proj_bias)
        self.mamba_rms_norm = bool(mamba_rms_norm)
        self.mamba_norm_before_gate = bool(mamba_norm_before_gate)
        self.mamba_use_mlp = bool(mamba_use_mlp)

        self.attention_bias = bool(attention_bias)
        self.attention_in_multiplier = float(attention_in_multiplier)
        self.attention_out_multiplier = float(attention_out_multiplier)
        self.key_multiplier = float(key_multiplier)
        self.ssm_in_multiplier = float(ssm_in_multiplier)
        self.ssm_out_multiplier = float(ssm_out_multiplier)
        self.ssm_multipliers = (
            list(ssm_multipliers)
            if ssm_multipliers is not None
            else [
                0.3535533905932738,
                0.25,
                0.3535533905932738,
                0.5,
                0.3535533905932738,
            ]
        )
        self.embedding_multiplier = float(embedding_multiplier)
        self.lm_head_multiplier = float(lm_head_multiplier)
        self.mlp_multipliers = (
            list(mlp_multipliers)
            if mlp_multipliers is not None
            else [
                0.8838834764831844,
                0.5859375,
            ]
        )
        self.mlp_bias = bool(mlp_bias)
        self.mlp_expansion_factor = int(mlp_expansion_factor)
        self.projectors_bias = bool(projectors_bias)
        self.initializer_range = float(initializer_range)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("FalconH1Config",)
