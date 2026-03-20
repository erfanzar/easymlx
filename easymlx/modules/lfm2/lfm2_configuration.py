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

"""LFM2 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("lfm2")
class Lfm2Config(EasyMLXBaseConfig):
    """Configuration for the LFM2 hybrid conv-attention model.

    LFM2 interleaves short depthwise convolution layers with full
    RoPE-based multi-head attention layers. Convolution layers provide
    efficient local context while attention layers handle long-range
    dependencies.

    Attributes:
        model_type: Identifier string (``"lfm2"``).
        vocab_size: Size of the token vocabulary. Defaults to ``32000``.
        hidden_size: Dimensionality of the hidden representations. Defaults to ``4096``.
        num_hidden_layers: Number of decoder layers. Defaults to ``32``.
        num_attention_heads: Number of attention heads. Defaults to ``32``.
        num_key_value_heads: Number of key/value heads for GQA. Defaults to
            ``num_attention_heads`` (MHA).
        max_position_embeddings: Maximum sequence length for RoPE. Defaults to ``8192``.
        norm_eps: Epsilon for RMSNorm layers. Defaults to ``1e-5``.
        conv_bias: Whether convolution layers use bias. Defaults to ``True``.
        conv_L_cache: Kernel size for the depthwise conv1d. Defaults to ``4``.
        block_dim: Input dimension for the MLP block. Defaults to ``4096``.
        block_ff_dim: Base feed-forward intermediate dimension. Defaults to ``11008``.
        block_multiple_of: Alignment constraint for the adjusted FF dim. Defaults to ``256``.
        block_ffn_dim_multiplier: Multiplier applied to the FF dim. Defaults to ``1.0``.
        block_auto_adjust_ff_dim: Whether to auto-adjust the FF dim using the
            2/3 heuristic and alignment. Defaults to ``True``.
        rope_theta: RoPE base frequency. Defaults to ``1000000.0``.
        rope_parameters: Optional dict with additional RoPE parameters
            (may override ``rope_theta``).
        full_attn_idxs: Indices of layers that use full attention. Derived
            from ``layer_types`` when not provided.
        layer_types: Per-layer type list (``"full_attention"`` or ``"conv"``).
            Defaults to all ``"full_attention"``.
        tie_word_embeddings: Whether to tie input and output embedding weights.
            Defaults to ``True``.

    Example:
        >>> config = Lfm2Config(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
    """

    model_type = "lfm2"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 8192,
        norm_eps: float = 1e-5,
        conv_bias: bool = True,
        conv_L_cache: int = 4,
        block_dim: int = 4096,
        block_ff_dim: int = 11008,
        block_multiple_of: int = 256,
        block_ffn_dim_multiplier: float = 1.0,
        block_auto_adjust_ff_dim: bool = True,
        rope_theta: float = 1000000.0,
        rope_parameters: dict[str, tp.Any] | None = None,
        full_attn_idxs: list[int] | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize LFM2 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of key/value heads for GQA. If ``None``,
                defaults to ``num_attention_heads``.
            max_position_embeddings: Maximum sequence length for positional encoding.
            norm_eps: Epsilon for RMSNorm layers.
            conv_bias: Whether convolution and projection layers use bias.
            conv_L_cache: Kernel size for the depthwise convolution.
            block_dim: Input dimension for the MLP block.
            block_ff_dim: Base feed-forward intermediate dimension.
            block_multiple_of: Alignment constraint for the adjusted FF dim.
            block_ffn_dim_multiplier: Multiplier applied to the FF dim.
            block_auto_adjust_ff_dim: Whether to auto-adjust the FF dim.
            rope_theta: RoPE base frequency.
            rope_parameters: Optional dict with additional RoPE parameters.
            full_attn_idxs: Explicit list of attention layer indices.
            layer_types: Per-layer type list. If ``None``, defaults to all
                ``"full_attention"``.
            tie_word_embeddings: Whether to tie input/output embedding weights.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.norm_eps = float(norm_eps)
        self.conv_bias = bool(conv_bias)
        self.conv_L_cache = int(conv_L_cache)
        self.block_dim = int(block_dim)
        self.block_ff_dim = int(block_ff_dim)
        self.block_multiple_of = int(block_multiple_of)
        self.block_ffn_dim_multiplier = float(block_ffn_dim_multiplier)
        self.block_auto_adjust_ff_dim = bool(block_auto_adjust_ff_dim)
        self.rope_theta = float(rope_theta)
        self.rope_parameters = rope_parameters

        if layer_types is not None:
            self.layer_types = layer_types
        else:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if full_attn_idxs is not None:
            self.full_attn_idxs = full_attn_idxs
        else:
            self.full_attn_idxs = [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("Lfm2Config",)
