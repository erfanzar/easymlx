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

"""Kimi Linear configuration for EasyMLX.

Kimi Linear is a hybrid model mixing MLA attention layers with
gated-delta-rule linear attention layers, plus an MoE feed-forward.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("kimi_linear")
class KimiLinearConfig(EasyMLXBaseConfig):
    """Configuration for the Kimi Linear transformer model.

    Attributes:
        model_type: The model type identifier (``"kimi_linear"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        rope_theta: RoPE base frequency.
        rms_norm_eps: RMSNorm epsilon.
        linear_attn_config: Configuration dict for linear attention layers.
        num_experts: Total number of routing experts.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        kv_lora_rank: Rank of the KV LoRA compression.
        qk_nope_head_dim: Dimension of the non-RoPE portion of each head.
        qk_rope_head_dim: Dimension of the RoPE portion of each head.
        v_head_dim: Dimension of the value head.
        num_experts_per_token: Number of experts per token.
        num_shared_experts: Number of shared experts.
        routed_scaling_factor: Scaling factor for routed expert weights.
        first_k_dense_replace: Number of initial dense layers.
        moe_layer_freq: Frequency of MoE layers.
    """

    model_type = "kimi_linear"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        rms_norm_eps: float = 1e-6,
        linear_attn_config: dict[str, tp.Any] | None = None,
        model_max_length: int = 131072,
        num_experts: int = 0,
        moe_intermediate_size: int = 1407,
        kv_lora_rank: int = 512,
        mla_use_nope: bool = False,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        num_experts_per_token: int = 1,
        num_shared_experts: int = 0,
        moe_router_activation_func: str = "sigmoid",
        moe_renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize KimiLinearConfig.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads.
            head_dim: Per-head dimensionality.
            rope_theta: RoPE base frequency.
            rope_scaling: Optional RoPE scaling configuration.
            rms_norm_eps: RMSNorm epsilon.
            linear_attn_config: Configuration dict for linear attention layers.
            model_max_length: Maximum model context length.
            num_experts: Total number of routing experts (0 = dense).
            moe_intermediate_size: Per-expert intermediate dimensionality.
            kv_lora_rank: Rank of the KV LoRA compression.
            mla_use_nope: Whether MLA uses non-positional encoding.
            qk_nope_head_dim: Dimension of the non-RoPE Q/K portion.
            qk_rope_head_dim: Dimension of the RoPE Q/K portion.
            v_head_dim: Dimension of the value head.
            num_experts_per_token: Experts activated per token.
            num_shared_experts: Number of always-active shared experts.
            moe_router_activation_func: Router activation (``"sigmoid"`` or ``"softmax"``).
            moe_renormalize: Whether to renormalize expert scores.
            routed_scaling_factor: Scaling factor for routed expert weights.
            first_k_dense_replace: Number of initial dense (non-MoE) layers.
            moe_layer_freq: Frequency of MoE layers.
            use_grouped_topk: Whether to use grouped top-k selection.
            num_expert_group: Number of expert groups.
            topk_group: Top-k within each group.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
        """
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = float(rms_norm_eps)
        self.linear_attn_config = linear_attn_config or {
            "kda_layers": [],
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        }
        self.model_max_length = int(model_max_length)
        self.num_experts = int(num_experts)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.kv_lora_rank = int(kv_lora_rank)
        self.mla_use_nope = bool(mla_use_nope)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_experts_per_token = int(num_experts_per_token)
        self.num_shared_experts = int(num_shared_experts)
        self.moe_router_activation_func = str(moe_router_activation_func)
        self.moe_renormalize = bool(moe_renormalize)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.moe_layer_freq = int(moe_layer_freq)
        self.use_grouped_topk = bool(use_grouped_topk)
        self.num_expert_group = int(num_expert_group)
        self.topk_group = int(topk_group)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("KimiLinearConfig",)
