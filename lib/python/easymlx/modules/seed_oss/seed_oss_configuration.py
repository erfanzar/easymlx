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

"""Seed OSS configuration for EasyMLX.

Seed OSS is a standard dense transformer (Llama-style) with optional
attention and MLP biases.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("seed_oss")
class SeedOssConfig(EasyMLXBaseConfig):
    """Configuration for the Seed OSS transformer model.

    Seed OSS is a standard dense transformer architecture following the Llama
    design pattern. It uses Grouped Query Attention (GQA), Rotary Positional
    Embeddings (RoPE), SwiGLU MLP, and RMSNorm. It supports optional biases
    on attention projections (input and output separately) and MLP projections.

    Attributes:
        model_type: The model type identifier (``"seed_oss"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality (SwiGLU gate/up size).
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA. Defaults to
            ``num_attention_heads`` (standard MHA) when not specified.
        head_dim: Per-head dimensionality. When ``None``, computed as
            ``hidden_size // num_attention_heads``.
        max_position_embeddings: Maximum sequence length supported by RoPE.
        rms_norm_eps: Epsilon for RMSNorm layers.
        rope_theta: RoPE base frequency for positional encoding.
        rope_traditional: Whether to use the traditional (interleaved) RoPE layout.
        rope_scaling: Optional RoPE scaling configuration dict (e.g., for YaRN).
        attention_bias: Whether Q/K/V projections include bias terms.
        attention_out_bias: Whether the output projection includes a bias term.
        mlp_bias: Whether MLP gate/up/down projections include bias terms.

    Example:
        >>> config = SeedOssConfig(hidden_size=2048, num_hidden_layers=24)
        >>> config.hidden_size
        2048
    """

    model_type = "seed_oss"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        attention_out_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize a Seed OSS configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Inner dimensionality of the SwiGLU MLP.
            num_hidden_layers: Number of stacked decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA. Defaults
                to ``num_attention_heads`` when ``None``.
            head_dim: Per-head dimensionality. Inferred from
                ``hidden_size // num_attention_heads`` when ``None``.
            max_position_embeddings: Maximum sequence length supported by RoPE.
            rms_norm_eps: Epsilon for RMSNorm numerical stability.
            rope_theta: Base frequency for RoPE.
            rope_traditional: If ``True``, uses interleaved RoPE layout.
            rope_scaling: Optional dict for RoPE scaling (e.g., YaRN config).
            attention_bias: Whether Q/K/V projections have bias.
            attention_out_bias: Whether the output projection has bias.
            mlp_bias: Whether MLP projections have bias.
            tie_word_embeddings: Whether to tie input and output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end-of-sequence.
            bos_token_id: Token ID for beginning-of-sequence.
            **kwargs: Additional keyword arguments forwarded to the base config.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_out_bias = bool(attention_out_bias)
        self.mlp_bias = bool(mlp_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("SeedOssConfig",)
