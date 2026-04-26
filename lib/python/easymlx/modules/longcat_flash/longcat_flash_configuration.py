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

"""LongcatFlash configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("longcat_flash")
class LongcatFlashConfig(EasyMLXBaseConfig):
    """Configuration for the LongcatFlash transformer model.

    LongcatFlash uses Multi-head Latent Attention (MLA) with dual attention
    sub-layers per block, MoE feed-forward with identity zero experts,
    and SwiGLU dense MLPs.

    Attributes:
        model_type: Identifier string (``"longcat_flash"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        ffn_hidden_size: Dense MLP intermediate dimensionality.
        expert_ffn_hidden_size: Per-expert MLP intermediate dimensionality.
        moe_topk: Number of experts activated per token.
        n_routed_experts: Total number of routed experts.
        zero_expert_num: Number of identity (zero) experts.
        zero_expert_type: Type of zero expert (``"identity"``).
        num_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        kv_lora_rank: Rank of the KV LoRA compression.
        q_lora_rank: Rank of the query LoRA compression.
        qk_rope_head_dim: Dimension of the RoPE portion of each head.
        qk_nope_head_dim: Dimension of the non-RoPE portion of each head.
        v_head_dim: Dimension of the value head.
        routed_scaling_factor: Scaling factor for routed expert weights.
        attention_bias: Whether attention includes bias.
        mla_scale_q_lora: Whether to scale Q LoRA output.
        mla_scale_kv_lora: Whether to scale KV LoRA output.
    """

    model_type = "longcat_flash"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        ffn_hidden_size: int = 11008,
        expert_ffn_hidden_size: int = 1407,
        moe_topk: int = 6,
        n_routed_experts: int = 64,
        zero_expert_num: int = 1,
        zero_expert_type: str = "identity",
        num_layers: int = 30,
        max_position_embeddings: int = 4096,
        num_attention_heads: int = 32,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        routed_scaling_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        mla_scale_q_lora: bool = False,
        mla_scale_kv_lora: bool = False,
        norm_topk_prob: bool = False,
        router_bias: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_method: str = "mla",
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self.intermediate_size = self.ffn_hidden_size
        self.expert_ffn_hidden_size = int(expert_ffn_hidden_size)
        self.moe_topk = int(moe_topk)
        self.n_routed_experts = int(n_routed_experts)
        self.zero_expert_num = int(zero_expert_num)
        self.zero_expert_type = str(zero_expert_type)
        self.num_layers = int(num_layers)
        self.num_hidden_layers = self.num_layers
        self.max_position_embeddings = int(max_position_embeddings)
        self.num_attention_heads = int(num_attention_heads)
        self.kv_lora_rank = int(kv_lora_rank)
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.attention_bias = bool(attention_bias)
        self.mla_scale_q_lora = bool(mla_scale_q_lora)
        self.mla_scale_kv_lora = bool(mla_scale_kv_lora)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.router_bias = bool(router_bias)
        self.rope_scaling = rope_scaling
        self.attention_method = str(attention_method)

        """Initialize LongcatFlash configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            ffn_hidden_size: Dense MLP intermediate dimension.
            expert_ffn_hidden_size: Per-expert MLP intermediate dimension.
            moe_topk: Number of experts activated per token.
            n_routed_experts: Total number of routed experts.
            zero_expert_num: Number of identity (zero) experts.
            zero_expert_type: Type of zero expert (e.g., ``"identity"``).
            num_layers: Number of decoder layers.
            max_position_embeddings: Maximum sequence length.
            num_attention_heads: Number of attention heads.
            kv_lora_rank: Rank of the KV LoRA compression.
            q_lora_rank: Rank of the query LoRA compression. ``None`` for
                direct projection.
            qk_rope_head_dim: Dimension of the RoPE portion of each head.
            qk_nope_head_dim: Dimension of the non-RoPE portion of each head.
            v_head_dim: Dimension of the value head.
            routed_scaling_factor: Scaling factor for routed expert weights.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            attention_bias: Whether attention includes bias.
            mla_scale_q_lora: Whether to scale Q LoRA output.
            mla_scale_kv_lora: Whether to scale KV LoRA output.
            norm_topk_prob: Whether to normalize top-k probabilities.
            router_bias: Whether the router uses bias.
            rope_scaling: Optional RoPE scaling configuration dict.
            attention_method: Attention method identifier.
            tie_word_embeddings: Whether to tie embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("LongcatFlashConfig",)
