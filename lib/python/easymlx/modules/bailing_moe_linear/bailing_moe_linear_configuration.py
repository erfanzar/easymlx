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

"""Bailing MoE Linear configuration for serving and inference.

This module defines the configuration class for the Bailing MoE Linear model
(hybrid linear + full attention), registered with the EasyMLX factory under
the ``"bailing_moe_linear"`` model type.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("bailing_moe_linear")
class BailingMoeLinearConfig(EasyMLXBaseConfig):
    """Configuration for the Bailing MoE Linear language model.

    Registered with the EasyMLX factory under the ``"bailing_moe_linear"``
    model type. This architecture uses a hybrid of full attention and
    GLA-style linear attention layers, interleaved based on
    ``layer_group_size``. Every Nth layer (and trailing layers) uses
    full attention; others use gated linear attention.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 102400.
        hidden_size: Dimensionality of hidden representations. Defaults to 4096.
        intermediate_size: Dense MLP intermediate dimensionality. Defaults to 11008.
        moe_intermediate_size: Per-expert intermediate dimensionality. Defaults to 1407.
        num_hidden_layers: Number of decoder layers. Defaults to 30.
        num_attention_heads: Number of query attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 32.
        num_experts: Total number of routing experts. Defaults to 8.
        num_shared_experts: Number of shared experts. Defaults to 2.
        num_experts_per_tok: Experts activated per token. Defaults to 2.
        first_k_dense_replace: Layers before this index use dense MLP. Defaults to 0.
        norm_topk_prob: Whether to normalize top-k routing probabilities. Defaults to ``True``.
        head_dim: Per-head dimensionality. If ``None``, auto-derived. Defaults to ``None``.
        layer_group_size: Every Nth layer uses full attention; others use
            linear attention. Defaults to 4.
        group_norm_size: Number of groups for GroupRMSNorm in linear
            attention output. Defaults to 1.
        rope_traditional: Whether to use traditional RoPE layout. Defaults to ``False``.

    Attributes:
        model_type: The model type identifier (``"bailing_moe_linear"``).

    Example::

        >>> config = BailingMoeLinearConfig(hidden_size=2048, layer_group_size=4)
        >>> config.model_type
        'bailing_moe_linear'
    """

    model_type = "bailing_moe_linear"

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
        num_experts: int = 8,
        num_shared_experts: int = 2,
        num_experts_per_tok: int = 2,
        first_k_dense_replace: int = 0,
        norm_topk_prob: bool = True,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        rope_traditional: bool = False,
        use_bias: bool = False,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = False,
        norm_head: bool = False,
        norm_softmax: bool = False,
        tie_word_embeddings: bool = False,
        partial_rotary_factor: float = 1.0,
        moe_router_enable_expert_bias: bool = False,
        moe_router_enable_routed_scaling: bool = True,
        routed_scaling_factor: float = 1.0,
        score_function: str = "softmax",
        n_group: int = 1,
        topk_group: int = 4,
        use_rmsnorm: bool = True,
        moe_shared_expert_intermediate_size: int | None = None,
        moe_router_enable_shared_expert: bool = True,
        head_dim: int | None = None,
        layer_group_size: int = 4,
        group_norm_size: int = 1,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Bailing MoE Linear configuration.

        See class docstring for full parameter documentation.
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
        self.num_experts = int(num_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.first_k_dense_replace = int(first_k_dense_replace)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.rope_traditional = bool(rope_traditional)
        self.use_bias = bool(use_bias)
        self.use_qkv_bias = bool(use_qkv_bias)
        self.use_qk_norm = bool(use_qk_norm)
        self.norm_head = bool(norm_head)
        self.norm_softmax = bool(norm_softmax)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.moe_router_enable_expert_bias = bool(moe_router_enable_expert_bias)
        self.moe_router_enable_routed_scaling = bool(moe_router_enable_routed_scaling)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.score_function = str(score_function)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_router_enable_shared_expert = bool(moe_router_enable_shared_expert)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_group_size = int(layer_group_size)
        self.group_norm_size = int(group_norm_size)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "BailingMoeLinearConfig"
