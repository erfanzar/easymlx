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

"""Step 3.5 configuration for EasyMLX.

Step 3.5 is a Mixture-of-Experts model with sliding window attention,
zero-centered RMSNorm, head-wise attention gating, and clamped SwiGLU
activations.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("step3p5")
class Step3p5Config(EasyMLXBaseConfig):
    """Configuration for the Step 3.5 Mixture-of-Experts transformer model.

    Step 3.5 is a large-scale MoE architecture featuring sliding window
    attention on designated layers, zero-centered RMSNorm, head-wise
    attention gating (learned sigmoid gates per head), clamped SwiGLU
    activations, sigmoid-scored MoE routing with router bias, and a
    shared expert alongside routed experts.

    Attributes:
        model_type: The model type identifier (``"step3p5"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality (for
            non-MoE layers and shared expert fallback).
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_attention_groups: Number of KV groups for GQA.
        head_dim: Per-head dimensionality.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency (can be a per-layer list).
        rope_scaling: Optional RoPE scaling configuration dict.
        max_position_embeddings: Maximum sequence length.
        sliding_window: Sliding window size for sliding-attention layers.
        layer_types: Per-layer attention type strings (e.g.,
            ``"sliding_attention"`` or ``"full_attention"``).
        yarn_only_types: Layer types that should use YaRN rope scaling.
        partial_rotary_factors: Per-layer partial rotary factors.
        attention_other_setting: Alternative head count config for sliding
            attention layers.
        use_head_wise_attn_gate: Whether to apply learned sigmoid gates
            per attention head.
        moe_num_experts: Total number of routed experts.
        moe_top_k: Number of experts activated per token.
        moe_intermediate_size: Per-expert SwiGLU intermediate dimensionality.
        share_expert_dim: Shared expert intermediate dimensionality.
        moe_layers_enum: Comma-separated layer indices that use MoE (when
            ``None``, layers 1..N-1 are MoE).
        moe_router_scaling_factor: Scaling factor applied to routed expert
            weights after top-k selection.
        norm_expert_weight: Whether to L1-normalize top-k expert weights.
        swiglu_limits: Per-layer clamping limits for routed expert SwiGLU.
        swiglu_limits_shared: Per-layer clamping limits for shared expert
            SwiGLU.

    Example:
        >>> config = Step3p5Config(moe_num_experts=288, moe_top_k=8)
        >>> config.moe_num_experts
        288
    """

    model_type = "step3p5"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_attention_groups: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float | list[float] = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 262144,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        yarn_only_types: list[str] | None = None,
        partial_rotary_factors: list[float] | None = None,
        attention_other_setting: dict[str, tp.Any] | None = None,
        use_head_wise_attn_gate: bool = True,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        moe_intermediate_size: int = 1280,
        share_expert_dim: int = 1280,
        moe_layers_enum: str | None = None,
        moe_router_scaling_factor: float = 3.0,
        norm_expert_weight: bool = True,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize a Step 3.5 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of query attention heads.
            num_attention_groups: Number of KV groups for GQA.
            head_dim: Per-head dimensionality.
            rms_norm_eps: RMSNorm epsilon.
            rope_theta: RoPE base frequency or per-layer list.
            rope_scaling: Optional RoPE scaling config dict.
            max_position_embeddings: Maximum sequence length.
            sliding_window: Sliding window size.
            layer_types: Per-layer attention type strings.
            yarn_only_types: Layer types that receive YaRN scaling.
            partial_rotary_factors: Per-layer partial rotary factors.
            attention_other_setting: Alternative head counts for sliding layers.
            use_head_wise_attn_gate: Whether to use head-wise attention gating.
            moe_num_experts: Total number of routed experts.
            moe_top_k: Number of experts per token.
            moe_intermediate_size: Per-expert intermediate size.
            share_expert_dim: Shared expert intermediate size.
            moe_layers_enum: Comma-separated MoE layer indices.
            moe_router_scaling_factor: Router scaling factor.
            norm_expert_weight: Whether to normalize expert weights.
            swiglu_limits: Per-layer SwiGLU clamp limits for routed experts.
            swiglu_limits_shared: Per-layer SwiGLU clamp limits for shared expert.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional arguments forwarded to the base config.
        """
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            if "type" in rope_scaling and "rope_type" not in rope_scaling:
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_attention_groups = int(num_attention_groups)
        self.num_key_value_heads = self.num_attention_groups
        self.head_dim = int(head_dim)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)
        self.sliding_window = int(sliding_window)
        self.layer_types = layer_types
        self.yarn_only_types = yarn_only_types
        self.partial_rotary_factors = partial_rotary_factors
        self.attention_other_setting = attention_other_setting
        self.use_head_wise_attn_gate = bool(use_head_wise_attn_gate)
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.share_expert_dim = int(share_expert_dim)
        self.moe_layers_enum = moe_layers_enum
        self.moe_router_scaling_factor = float(moe_router_scaling_factor)
        self.norm_expert_weight = bool(norm_expert_weight)
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Step3p5Config",)
