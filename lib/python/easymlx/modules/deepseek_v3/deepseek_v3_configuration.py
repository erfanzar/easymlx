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

"""DeepSeek V3 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("deepseek_v3")
class DeepseekV3Config(EasyMLXBaseConfig):
    """Configuration for the DeepSeek V3 transformer model with MLA and MoE.

    DeepSeek V3 extends V2 with MultiLinear layers for embed_q/unembed_out,
    FP8 dequantization support, compiled group expert selection, and
    noaux_tc routing method with sigmoid scoring and score correction bias.

    Attributes:
        model_type: The model type identifier (``"deepseek_v3"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads.
        kv_lora_rank: Rank of the KV LoRA compression.
        q_lora_rank: Rank of the query LoRA compression.
        qk_rope_head_dim: Dimension of the RoPE portion of each head.
        qk_nope_head_dim: Dimension of the non-RoPE portion of each head.
        v_head_dim: Dimension of the value head.
        scoring_func: Scoring function for expert routing.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        n_routed_experts: Total number of routing experts.
        n_shared_experts: Number of shared experts.
        num_experts_per_tok: Number of experts per token.
        routed_scaling_factor: Scaling factor for routed expert weights.
        topk_method: Method for top-k expert selection.
        n_group: Number of expert groups.
        topk_group: Number of groups to keep per token.
    """

    model_type = "deepseek_v3"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 1407,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        routed_scaling_factor: float = 1.0,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 1,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 0,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        head_dim: int | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize DeepSeek V3 configuration.

        Args:
            vocab_size: Size of the token vocabulary. Defaults to ``102400``.
            hidden_size: Dimensionality of the transformer hidden states. Defaults to ``4096``.
            intermediate_size: Dense MLP intermediate dimensionality. Defaults to ``11008``.
            moe_intermediate_size: Per-expert intermediate dimensionality. Defaults to ``1407``.
            num_hidden_layers: Number of transformer decoder layers. Defaults to ``30``.
            num_attention_heads: Number of query attention heads. Defaults to ``32``.
            num_key_value_heads: Number of key/value heads. Defaults to ``32``.
            n_shared_experts: Number of shared (always-active) experts, or ``None``.
            n_routed_experts: Total number of routing experts, or ``None``.
            routed_scaling_factor: Scaling factor for routed expert weights. Defaults to ``1.0``.
            kv_lora_rank: Rank of the KV LoRA compression in MLA. Defaults to ``512``.
            q_lora_rank: Rank of the query LoRA compression, or ``None``.
            qk_rope_head_dim: RoPE portion dimension per head. Defaults to ``64``.
            v_head_dim: Value head dimension. Defaults to ``128``.
            qk_nope_head_dim: Non-RoPE portion dimension per head. Defaults to ``128``.
            topk_method: Expert selection method (``"noaux_tc"``). Defaults to ``"noaux_tc"``.
            scoring_func: Scoring function for routing (``"sigmoid"``). Defaults to ``"sigmoid"``.
            norm_topk_prob: Whether to normalize top-k routing probabilities. Defaults to ``True``.
            n_group: Number of expert groups. Defaults to ``1``.
            topk_group: Number of groups to keep per token. Defaults to ``1``.
            num_experts_per_tok: Number of experts activated per token. Defaults to ``1``.
            moe_layer_freq: Frequency of MoE layers. Defaults to ``1``.
            first_k_dense_replace: Initial dense layers before MoE. Defaults to ``0``.
            max_position_embeddings: Maximum sequence length. Defaults to ``2048``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-6``.
            rope_theta: RoPE base frequency. Defaults to ``10000.0``.
            rope_scaling: Optional RoPE scaling configuration dict.
            attention_bias: Whether attention includes bias. Defaults to ``False``.
            head_dim: Per-head dimension override, or ``None``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``False``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to ``EasyMLXBaseConfig``.
        """
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
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.kv_lora_rank = int(kv_lora_rank)
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.topk_method = str(topk_method)
        self.scoring_func = str(scoring_func)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.moe_layer_freq = int(moe_layer_freq)
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("DeepseekV3Config",)
