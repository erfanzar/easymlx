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

"""Dots1 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("dots1")
class Dots1Config(EasyMLXBaseConfig):
    """Configuration for the Dots1 MoE transformer model.

    Dots1 is a MoE architecture with QK-norm, grouped expert selection
    using sigmoid scoring, and score correction bias routing.

    Attributes:
        model_type: Identifier string (``"dots1"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        first_k_dense_replace: Number of initial dense layers before MoE.
        n_routed_experts: Total number of routed experts.
        n_shared_experts: Number of shared (always-active) experts.
        num_experts_per_tok: Number of experts activated per token.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        routed_scaling_factor: Scaling factor for routed expert weights.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to keep during routing.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
    """

    model_type = "dots1"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 1407,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        first_k_dense_replace: int = 0,
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        scoring_func: str = "noaux_tc",
        n_group: int = 1,
        topk_group: int = 1,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Dots1 configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``102400``.
            hidden_size: Hidden dimension. Defaults to ``4096``.
            intermediate_size: Dense MLP intermediate dimension. Defaults to ``11008``.
            moe_intermediate_size: Per-expert intermediate dimension. Defaults to ``1407``.
            num_hidden_layers: Number of decoder layers. Defaults to ``30``.
            num_attention_heads: Number of query heads. Defaults to ``32``.
            num_key_value_heads: Number of KV heads, or ``None`` (defaults to
                ``num_attention_heads``).
            head_dim: Per-head dimension, or ``None`` to derive.
            first_k_dense_replace: Number of initial dense layers before MoE.
                Defaults to ``0``.
            n_routed_experts: Total number of routed experts. Defaults to ``64``.
            n_shared_experts: Number of shared experts. Defaults to ``1``.
            num_experts_per_tok: Experts activated per token. Defaults to ``6``.
            norm_topk_prob: Whether to normalize top-k probs. Defaults to ``True``.
            routed_scaling_factor: Scaling for routed weights. Defaults to ``1.0``.
            scoring_func: Scoring function for routing. Defaults to ``"noaux_tc"``.
            n_group: Number of expert groups. Defaults to ``1``.
            topk_group: Groups to keep per token. Defaults to ``1``.
            max_position_embeddings: Max sequence length. Defaults to ``2048``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-6``.
            rope_theta: RoPE base frequency. Defaults to ``10000.0``.
            rope_scaling: Optional RoPE scaling configuration.
            attention_bias: Whether attention has bias. Defaults to ``False``.
            mlp_bias: Whether MLP has bias. Defaults to ``False``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``False``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.scoring_func = str(scoring_func)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
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


__all__ = ("Dots1Config",)
