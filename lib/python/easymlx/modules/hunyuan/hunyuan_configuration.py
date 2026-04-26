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

"""Hunyuan configuration for EasyMLX inference.

Hunyuan is a Chinese LLM with MoE architecture featuring:
  - Dynamic NTK-Alpha RoPE scaling
  - QK normalization
  - Cross-Layer Attention (CLA) for KV sharing
  - Mixed MLP/MoE layers with shared experts
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("hunyuan")
class HunyuanConfig(EasyMLXBaseConfig):
    """Configuration for the Hunyuan MoE language model.

    Attributes:
        model_type: Identifier string (``"hunyuan"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        moe_topk: Number of experts activated per token.
        num_experts: Total number of routing experts.
        num_shared_expert: Number of shared (always-active) experts.
        use_mixed_mlp_moe: Whether to use mixed MLP/MoE.
        use_qk_norm: Whether to apply QK normalization.
        use_cla: Whether to use Cross-Layer Attention.
        cla_share_factor: CLA sharing factor.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: RoPE scaling configuration.
        attention_bias: Whether attention projections have bias.
        moe_intermediate_size: Per-expert intermediate size.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "hunyuan"

    def __init__(
        self,
        *,
        vocab_size: int = 128256,
        hidden_size: int = 6400,
        intermediate_size: int = 18304,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 80,
        num_key_value_heads: int | None = None,
        moe_topk: int = 1,
        num_experts: int = 1,
        num_shared_expert: int | tp.Any = 1,
        use_mixed_mlp_moe: bool = False,
        use_qk_norm: bool = True,
        use_cla: bool = False,
        cla_share_factor: int = 2,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = True,
        moe_intermediate_size: int | list[int] | None = None,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize HunyuanConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA. Defaults to
                ``num_attention_heads`` if ``None``.
            moe_topk: Number of experts activated per token.
            num_experts: Total number of routing experts (1 = dense).
            num_shared_expert: Number of shared (always-active) experts.
            use_mixed_mlp_moe: Whether to use mixed MLP/MoE layers.
            use_qk_norm: Whether to apply per-head QK normalization.
            use_cla: Whether to use Cross-Layer Attention.
            cla_share_factor: KV sharing period for CLA.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            rope_scaling: RoPE scaling configuration dict (may contain ``"alpha"``).
            attention_bias: Whether attention projections have bias terms.
            moe_intermediate_size: Per-expert intermediate size. May be a
                list for per-layer configuration.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base config.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.moe_topk = moe_topk
        self.num_experts = num_experts
        self.num_shared_expert = num_shared_expert
        self.use_mixed_mlp_moe = bool(use_mixed_mlp_moe)
        self.use_qk_norm = bool(use_qk_norm)
        self.use_cla = bool(use_cla)
        self.cla_share_factor = int(cla_share_factor)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.moe_intermediate_size = moe_intermediate_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "HunyuanConfig"
