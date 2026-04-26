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

"""Klear configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("Klear")
class KlearConfig(EasyMLXBaseConfig):
    """Configuration for the Klear MoE transformer model.

    Klear uses sigmoid-gated MoE routing with shared experts,
    a coefficient-based mixing layer, QK-norm, and per-layer
    sparse/dense MLP selection.

    Attributes:
        model_type: Identifier string (``"Klear"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality (auto-derived if ``None``).
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        attention_bias: Whether attention projections have bias.
        mlp_only_layers: Layer indices that use dense MLP instead of MoE.
        num_experts: Total number of routing experts.
        num_experts_per_tok: Experts activated per token.
        decoder_sparse_step: Period for MoE layer placement.
        n_shared_experts: Number of always-active shared experts.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        norm_topk_prob: Whether to normalize top-k routing probabilities.

    Example:
        >>> config = KlearConfig(
        ...     vocab_size=32000, hidden_size=2048,
        ...     num_hidden_layers=16, num_attention_heads=16,
        ...     num_experts=8, num_experts_per_tok=2,
        ... )
        >>> config.num_experts
        8
    """

    model_type = "Klear"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        mlp_only_layers: list[int] | None = None,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        decoder_sparse_step: int = 1,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 1408,
        norm_topk_prob: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize KlearConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            head_dim: Per-head dimensionality (auto-derived if ``None``).
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            attention_bias: Whether attention projections have bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            mlp_only_layers: Indices of layers using dense MLP instead of MoE.
            num_experts: Total number of routing experts.
            num_experts_per_tok: Experts activated per token.
            decoder_sparse_step: Period for MoE layer placement.
            n_shared_experts: Number of always-active shared experts.
            moe_intermediate_size: Per-expert intermediate dimensionality.
            norm_topk_prob: Whether to normalize top-k routing probabilities.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.attention_bias = bool(attention_bias)
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.decoder_sparse_step = int(decoder_sparse_step)
        self.n_shared_experts = int(n_shared_experts)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.norm_topk_prob = bool(norm_topk_prob)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("KlearConfig",)
