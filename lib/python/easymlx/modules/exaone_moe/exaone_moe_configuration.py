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

"""ExaoneMoE configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("exaone_moe")
class ExaoneMoeConfig(EasyMLXBaseConfig):
    """Configuration for the ExaoneMoE transformer model.

    ExaoneMoE combines per-layer dense/MoE selection (via is_moe_layer),
    sliding window attention, Q/K RMSNorm, shared experts, and group-based
    expert routing with sigmoid scoring.

    Attributes:
        model_type: Identifier string (``"exaone_moe"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        moe_intermediate_size: MoE expert intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        num_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts activated per token.
        num_shared_experts: Number of shared experts.
        is_moe_layer: Per-layer list of booleans for MoE vs dense.
        layer_types: Per-layer attention type (sliding_attention or attention).
        sliding_window: Sliding window size for local attention.
        rms_norm_eps: RMSNorm epsilon.
        n_group: Number of groups for group expert selection.
        topk_group: Number of top groups to keep.
        routed_scaling_factor: Scaling factor for routed expert scores.
        norm_topk_prob: Whether to normalize top-k probabilities.
        scoring_func: Scoring function for expert routing.
        topk_method: Top-k method for routing.
    """

    model_type = "exaone_moe"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int = 128,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 2,
        is_moe_layer: list[bool] | None = None,
        layer_types: list[str] | None = None,
        sliding_window: int = 4096,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        rope_parameters: dict[str, tp.Any] | None = None,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize ExaoneMoE configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``102400``.
            hidden_size: Hidden dimension. Defaults to ``4096``.
            intermediate_size: Dense MLP intermediate dimension. Defaults to ``14336``.
            moe_intermediate_size: Per-expert intermediate dimension. Defaults to ``2048``.
            num_hidden_layers: Number of decoder layers. Defaults to ``32``.
            num_attention_heads: Number of query heads. Defaults to ``32``.
            num_key_value_heads: Number of KV heads, or ``None``.
            head_dim: Per-head dimension. Defaults to ``128``.
            num_experts: Total number of routed experts. Defaults to ``64``.
            num_experts_per_tok: Experts per token. Defaults to ``8``.
            num_shared_experts: Shared experts. Defaults to ``2``.
            is_moe_layer: Per-layer list of booleans for MoE vs dense. If ``None``,
                all layers use MoE.
            layer_types: Per-layer attention type list (``"sliding_attention"`` or
                ``"attention"``). If ``None``, all layers use global attention.
            sliding_window: Window size for sliding attention. Defaults to ``4096``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-5``.
            max_position_embeddings: Max sequence length. Defaults to ``32768``.
            rope_theta: RoPE base frequency. Defaults to ``1000000000.0``.
            rope_scaling: Optional RoPE scaling configuration.
            rope_parameters: Optional rope parameters dict.
            n_group: Number of expert groups. Defaults to ``1``.
            topk_group: Groups to keep per token. Defaults to ``1``.
            routed_scaling_factor: Scaling for routed weights. Defaults to ``2.5``.
            norm_topk_prob: Whether to normalize top-k probs. Defaults to ``True``.
            scoring_func: Scoring function. Defaults to ``"sigmoid"``.
            topk_method: Top-k method. Defaults to ``"noaux_tc"``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``False``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if is_moe_layer is None:
            is_moe_layer = [True] * num_hidden_layers
        if layer_types is None:
            layer_types = ["attention"] * num_hidden_layers

        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_shared_experts = int(num_shared_experts)
        self.is_moe_layer = list(is_moe_layer)
        self.layer_types = list(layer_types)
        self.sliding_window = int(sliding_window)
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.scoring_func = str(scoring_func)
        self.topk_method = str(topk_method)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "ExaoneMoeConfig"
