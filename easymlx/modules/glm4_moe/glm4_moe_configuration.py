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

"""GLM-4 MoE configuration (serving/inference only).

This module defines the configuration class for the GLM-4 Mixture-of-Experts
model, specifying architecture hyperparameters for attention, MLP, and MoE
routing.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("glm4_moe")
class Glm4MoeConfig(EasyMLXBaseConfig):
    """Configuration for the GLM-4 Mixture-of-Experts model.

    This configuration defines all hyperparameters for the GLM-4 MoE
    architecture including attention, feed-forward, and expert routing
    parameters. It is registered under the model type ``"glm4_moe"``.

    Attributes:
        model_type: Identifier string for this model type (``"glm4_moe"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the hidden states.
        intermediate_size: Dimensionality of the dense MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        partial_rotary_factor: Fraction of head dimensions using rotary
            embeddings.
        num_key_value_heads: Number of key-value heads for GQA.
        head_dim: Per-head dimensionality, or None to derive from hidden_size.
        hidden_act: Activation function name for the MLP.
        max_position_embeddings: Maximum sequence length supported.
        rms_norm_eps: Epsilon for RMS normalization layers.
        use_cache: Whether to use KV caching during generation.
        tie_word_embeddings: Whether input and output embeddings are tied.
        rope_theta: Base frequency for rotary position embeddings.
        rope_scaling: Optional dictionary for RoPE scaling configuration.
        attention_bias: Whether attention projections include bias terms.
        attention_dropout: Dropout rate for attention weights.
        moe_intermediate_size: Intermediate size per routed expert.
        num_experts_per_tok: Number of experts activated per token.
        n_shared_experts: Number of shared (always-active) experts.
        n_routed_experts: Total number of routed experts.
        routed_scaling_factor: Scaling factor for routed expert scores.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to keep during routing.
        first_k_dense_replace: Number of initial layers using dense MLP
            instead of MoE.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        use_qk_norm: Whether to apply QK normalization.
        layer_types: Per-layer attention type specification.
    """

    model_type = "glm4_moe"

    def __init__(
        self,
        *,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        intermediate_size: int = 10944,
        num_hidden_layers: int = 46,
        num_attention_heads: int = 96,
        partial_rotary_factor: float = 0.5,
        num_key_value_heads: int = 8,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 1408,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 128,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = False,
        layer_types: list["str"] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initializes a GLM-4 MoE configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to 151552.
            hidden_size: Hidden state dimensionality. Defaults to 4096.
            intermediate_size: Dense MLP intermediate size. Defaults to 10944.
            num_hidden_layers: Number of decoder layers. Defaults to 46.
            num_attention_heads: Number of attention heads. Defaults to 96.
            partial_rotary_factor: Fraction of head dim for rotary embeddings.
                Defaults to 0.5.
            num_key_value_heads: Number of KV heads for GQA. Defaults to 8.
            head_dim: Per-head dimension. Defaults to None (auto-computed).
            hidden_act: Activation function name. Defaults to ``"silu"``.
            max_position_embeddings: Maximum sequence length. Defaults to 131072.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            use_cache: Enable KV caching. Defaults to True.
            tie_word_embeddings: Tie input/output embeddings. Defaults to False.
            rope_theta: RoPE base frequency. Defaults to 10000.0.
            rope_scaling: RoPE scaling config dict. Defaults to None.
            attention_bias: Include bias in attention projections.
                Defaults to False.
            attention_dropout: Attention dropout rate. Defaults to 0.0.
            moe_intermediate_size: Expert intermediate size. Defaults to 1408.
            num_experts_per_tok: Experts per token. Defaults to 8.
            n_shared_experts: Shared expert count. Defaults to 1.
            n_routed_experts: Total routed experts. Defaults to 128.
            routed_scaling_factor: Expert score scaling. Defaults to 1.0.
            n_group: Expert group count. Defaults to 1.
            topk_group: Top groups to keep. Defaults to 1.
            first_k_dense_replace: Dense layers before MoE. Defaults to 1.
            norm_topk_prob: Normalize routing probs. Defaults to True.
            use_qk_norm: Apply QK normalization. Defaults to False.
            layer_types: Per-layer attention types. Defaults to all
                ``"full_attention"``.
            pad_token_id: Padding token ID. Defaults to None.
            eos_token_id: End-of-sequence token ID(s). Defaults to None.
            bos_token_id: Beginning-of-sequence token ID. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.n_shared_experts = int(n_shared_experts)
        self.n_routed_experts = int(n_routed_experts)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.use_qk_norm = bool(use_qk_norm)
        self.layer_types = layer_types or ["full_attention"] * self.num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rotary_dim(self) -> int:
        """Computes the effective rotary embedding dimension.

        The dimension is derived from the head dimension and partial rotary
        factor, rounded down to the nearest even number.

        Returns:
            The number of dimensions to apply rotary embeddings to.
        """
        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)
        rotary_dim = int(head_dim * float(self.partial_rotary_factor))
        return max(0, min(head_dim, rotary_dim - (rotary_dim % 2)))


__all__ = "Glm4MoeConfig"
