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

"""MiMo-V2-Flash configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("mimo_v2_flash")
class MiMoV2FlashConfig(EasyMLXBaseConfig):
    """Configuration for the MiMo-V2-Flash hybrid MoE transformer model.

    MiMo-V2-Flash uses a DeepSeekV3-style architecture with sliding
    window attention, group-based top-k routing, shared experts, and
    asymmetric head dimensions for queries/values.

    Attributes:
        model_type: Identifier string (``"mimo_v2_flash"``).
        vocab_size: Size of the token vocabulary. Defaults to ``32000``.
        hidden_size: Dimensionality of hidden representations. Defaults to ``4096``.
        intermediate_size: Dense MLP intermediate dimension. Defaults to ``11008``.
        moe_intermediate_size: Per-expert MLP intermediate dimension. Defaults to ``4096``.
        num_hidden_layers: Number of decoder layers. Defaults to ``32``.
        num_attention_heads: Number of full-attention heads. Defaults to ``32``.
        num_key_value_heads: Number of full-attention KV heads. Defaults to ``num_attention_heads``.
        num_experts_per_tok: Number of experts activated per token. Defaults to ``2``.
        hybrid_layer_pattern: Per-layer pattern (``1`` = SWA, ``0`` = full attention).
        moe_layer_freq: Per-layer pattern (``1`` = MoE, ``0`` = dense).
        add_swa_attention_sink_bias: Whether SWA layers add sink bias.
        add_full_attention_sink_bias: Whether full attention layers add sink bias.
        sliding_window_size: Sliding window size for SWA layers. Defaults to ``4096``.
        n_shared_experts: Number of shared (always-active) experts.
        n_routed_experts: Number of routed experts.
        routed_scaling_factor: Scaling factor for routed expert weights.
        topk_method: Routing method (e.g., ``"noaux_tc"``).
        scoring_func: Scoring function (e.g., ``"sigmoid"``).
        norm_topk_prob: Whether to normalize top-k probabilities.
        n_group: Number of expert groups for group routing.
        topk_group: Number of groups to select in group routing.
        layernorm_epsilon: Epsilon for RMSNorm. Defaults to ``1e-5``.
        rope_theta: RoPE base frequency for full attention. Defaults to ``10000.0``.
        swa_rope_theta: RoPE base frequency for SWA layers. Defaults to ``10000.0``.
        swa_num_attention_heads: Number of SWA attention heads. Defaults to ``32``.
        swa_num_key_value_heads: Number of SWA KV heads.
        head_dim: Full-attention query/key head dimension. Defaults to ``128``.
        v_head_dim: Full-attention value head dimension. Defaults to ``128``.
        swa_head_dim: SWA query/key head dimension. Defaults to ``128``.
        swa_v_head_dim: SWA value head dimension. Defaults to ``128``.
        partial_rotary_factor: Fraction of head_dim to apply RoPE to. Defaults to ``1``.

    Example:
        >>> config = MiMoV2FlashConfig(
        ...     vocab_size=1000, hidden_size=64, num_hidden_layers=2,
        ... )
    """

    model_type = "mimo_v2_flash"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        num_experts_per_tok: int = 2,
        hybrid_layer_pattern: list[int] | None = None,
        moe_layer_freq: list[int] | None = None,
        add_swa_attention_sink_bias: bool = False,
        add_full_attention_sink_bias: bool = False,
        sliding_window_size: int = 4096,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        routed_scaling_factor: float | None = None,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        max_position_embeddings: int = 32768,
        layernorm_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        swa_rope_theta: float = 10000.0,
        swa_num_attention_heads: int = 32,
        swa_num_key_value_heads: int | None = None,
        head_dim: int = 128,
        v_head_dim: int = 128,
        swa_head_dim: int = 128,
        swa_v_head_dim: int = 128,
        partial_rotary_factor: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if swa_num_key_value_heads is None:
            swa_num_key_value_heads = swa_num_attention_heads
        if hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0] * num_hidden_layers
        if moe_layer_freq is None:
            moe_layer_freq = [0] * num_hidden_layers

        """Initialize MiMo-V2-Flash configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            intermediate_size: Dense MLP intermediate dimension.
            moe_intermediate_size: Per-expert MLP intermediate dimension.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of full-attention heads.
            num_key_value_heads: Number of full-attention KV heads.
            num_experts_per_tok: Experts activated per token.
            hybrid_layer_pattern: Per-layer SWA pattern.
            moe_layer_freq: Per-layer MoE pattern.
            add_swa_attention_sink_bias: SWA sink bias flag.
            add_full_attention_sink_bias: Full attention sink bias flag.
            sliding_window_size: Sliding window size.
            n_shared_experts: Number of shared experts.
            n_routed_experts: Number of routed experts.
            routed_scaling_factor: Routing weight scaling factor.
            topk_method: Routing method.
            scoring_func: Scoring function for routing.
            norm_topk_prob: Whether to normalize top-k probabilities.
            n_group: Number of expert groups.
            topk_group: Number of groups selected.
            max_position_embeddings: Maximum sequence length.
            layernorm_epsilon: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency for full attention.
            swa_rope_theta: RoPE base frequency for SWA.
            swa_num_attention_heads: Number of SWA attention heads.
            swa_num_key_value_heads: Number of SWA KV heads.
            head_dim: Full-attention Q/K head dimension.
            v_head_dim: Full-attention V head dimension.
            swa_head_dim: SWA Q/K head dimension.
            swa_v_head_dim: SWA V head dimension.
            partial_rotary_factor: Fraction of head_dim for RoPE.
            tie_word_embeddings: Whether to tie embeddings.
            **kwargs: Additional keyword arguments.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.moe_layer_freq = moe_layer_freq
        self.add_swa_attention_sink_bias = bool(add_swa_attention_sink_bias)
        self.add_full_attention_sink_bias = bool(add_full_attention_sink_bias)
        self.sliding_window_size = int(sliding_window_size)
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.norm_topk_prob = bool(norm_topk_prob)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.max_position_embeddings = int(max_position_embeddings)
        self.layernorm_epsilon = float(layernorm_epsilon)
        self.rope_theta = float(rope_theta)
        self.swa_rope_theta = float(swa_rope_theta)
        self.swa_num_attention_heads = int(swa_num_attention_heads)
        self.swa_num_key_value_heads = int(swa_num_key_value_heads)
        self.head_dim = int(head_dim)
        self.v_head_dim = int(v_head_dim)
        self.swa_head_dim = int(swa_head_dim)
        self.swa_v_head_dim = int(swa_v_head_dim)
        self.partial_rotary_factor = int(partial_rotary_factor)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("MiMoV2FlashConfig",)
