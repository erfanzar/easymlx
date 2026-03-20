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

"""LFM2-MoE configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("lfm2_moe")
class Lfm2MoeConfig(EasyMLXBaseConfig):
    """Configuration for the LFM2-MoE hybrid conv-attention MoE model.

    LFM2-MoE extends LFM2 with Mixture-of-Experts feed-forward layers.
    Layers below ``num_dense_layers`` use dense MLPs; layers at or above
    use a sparse MoE block with SwitchGLU experts and softmax gating.

    Attributes:
        model_type: Identifier string (``"lfm2_moe"``).
        vocab_size: Size of the token vocabulary. Defaults to ``32000``.
        hidden_size: Dimensionality of hidden representations. Defaults to ``4096``.
        intermediate_size: Dense MLP intermediate dimension. Defaults to ``11008``.
        moe_intermediate_size: Per-expert MLP intermediate dimension. Defaults to ``4096``.
        num_hidden_layers: Number of decoder layers. Defaults to ``32``.
        num_experts: Total number of MoE experts. Defaults to ``8``.
        num_experts_per_tok: Experts activated per token. Defaults to ``2``.
        norm_topk_prob: Whether to normalize top-k probabilities. Defaults to ``True``.
        num_attention_heads: Number of attention heads. Defaults to ``32``.
        num_key_value_heads: Number of KV heads for GQA. Defaults to ``num_attention_heads``.
        max_position_embeddings: Maximum sequence length. Defaults to ``8192``.
        use_expert_bias: Whether to add a learned bias to expert scores. Defaults to ``False``.
        num_dense_layers: Number of initial layers that use dense MLPs instead
            of MoE. Defaults to ``0``.
        norm_eps: Epsilon for RMSNorm. Defaults to ``1e-5``.
        conv_bias: Whether conv projections use bias. Defaults to ``True``.
        conv_L_cache: Convolution kernel size. Defaults to ``4``.
        rope_theta: RoPE base frequency. Defaults to ``1000000.0``.
        rope_parameters: Optional dict with additional RoPE parameters.
        full_attn_idxs: Indices of layers using full attention.
        layer_types: Per-layer type list.
        tie_word_embeddings: Whether to tie embeddings. Defaults to ``True``.

    Example:
        >>> config = Lfm2MoeConfig(
        ...     vocab_size=1000, hidden_size=64, num_hidden_layers=2,
        ...     num_experts=4, num_experts_per_tok=2,
        ... )
    """

    model_type = "lfm2_moe"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 4096,
        num_hidden_layers: int = 32,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        norm_topk_prob: bool = True,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 8192,
        use_expert_bias: bool = False,
        num_dense_layers: int = 0,
        norm_eps: float = 1e-5,
        conv_bias: bool = True,
        conv_L_cache: int = 4,
        rope_theta: float = 1000000.0,
        rope_parameters: dict[str, tp.Any] | None = None,
        full_attn_idxs: list[int] | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize LFM2-MoE configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            intermediate_size: Dense MLP intermediate dimension.
            moe_intermediate_size: Per-expert MLP intermediate dimension.
            num_hidden_layers: Number of decoder layers.
            num_experts: Total number of MoE experts.
            num_experts_per_tok: Number of experts activated per token.
            norm_topk_prob: Whether to normalize top-k probabilities.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            max_position_embeddings: Maximum sequence length.
            use_expert_bias: Whether to add a learned bias to expert gating.
            num_dense_layers: Number of initial dense (non-MoE) layers.
            norm_eps: Epsilon for RMSNorm.
            conv_bias: Whether conv and projection layers use bias.
            conv_L_cache: Convolution kernel size.
            rope_theta: RoPE base frequency.
            rope_parameters: Optional dict with additional RoPE parameters.
            full_attn_idxs: Explicit list of attention layer indices.
            layer_types: Per-layer type list.
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.use_expert_bias = bool(use_expert_bias)
        self.num_dense_layers = int(num_dense_layers)
        self.norm_eps = float(norm_eps)
        self.conv_bias = bool(conv_bias)
        self.conv_L_cache = int(conv_L_cache)
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


__all__ = ("Lfm2MoeConfig",)
