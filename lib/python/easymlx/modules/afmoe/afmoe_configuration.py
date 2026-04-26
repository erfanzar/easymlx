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

"""AFMoE configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("afmoe")
class AfmoeConfig(EasyMLXBaseConfig):
    """Configuration for the AFMoE transformer model.

    AFMoE features sliding and full attention patterns, gated attention,
    QK-norm, pre/post MLP layer norms, MuP input scaling, and a
    mixture-of-experts feed-forward with grouped expert routing.

    Attributes:
        model_type: Identifier string (``"afmoe"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        layer_types: Per-layer attention type specification.
        sliding_window: Sliding window size for local attention.
        num_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts activated per token.
        num_shared_experts: Number of shared (always-active) experts.
        num_dense_layers: Number of initial dense MLP layers.
        route_norm: Whether to normalize routing probabilities.
        route_scale: Scaling factor for routing scores.
        score_func: Scoring function for expert routing.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to keep during routing.
        mup_enabled: Whether MuP input scaling is enabled.
    """

    model_type = "afmoe"

    def __init__(
        self,
        *,
        vocab_size: int = 200192,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        moe_intermediate_size: int = 1024,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int = 64,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        tie_word_embeddings: bool = False,
        layer_types: list[str] | None = None,
        sliding_window: int = 2048,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        num_dense_layers: int = 2,
        route_norm: bool = True,
        route_scale: float = 2.826,
        score_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        mup_enabled: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the AFMoE configuration.

        Args:
            vocab_size: Number of tokens in the vocabulary. Defaults to 200192.
            hidden_size: Dimensionality of hidden representations. Defaults to 2048.
            intermediate_size: Dense MLP intermediate dimensionality. Defaults to 6144.
            moe_intermediate_size: Per-expert intermediate dimensionality. Defaults to 1024.
            num_hidden_layers: Number of decoder layers. Defaults to 32.
            num_attention_heads: Number of query attention heads. Defaults to 32.
            num_key_value_heads: Number of KV heads for GQA. Defaults to ``num_attention_heads``.
            head_dim: Per-head dimensionality. Defaults to 64.
            max_position_embeddings: Maximum sequence length. Defaults to 131072.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            rope_theta: RoPE base frequency. Defaults to 10000.0.
            rope_scaling: Optional RoPE scaling configuration dict. Defaults to ``None``.
            tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``False``.
            layer_types: Per-layer attention type (``"full_attention"`` or
                ``"sliding_attention"``). Defaults to all ``"full_attention"``.
            sliding_window: Sliding window size for local attention. Defaults to 2048.
            num_experts: Total number of routed experts. Defaults to 128.
            num_experts_per_tok: Experts activated per token. Defaults to 8.
            num_shared_experts: Always-active shared experts. Defaults to 1.
            num_dense_layers: Initial layers using dense MLP. Defaults to 2.
            route_norm: Whether to normalize routing probabilities. Defaults to ``True``.
            route_scale: Scaling factor for routing scores. Defaults to 2.826.
            score_func: Scoring function (``"sigmoid"`` or ``"softmax"``). Defaults to ``"sigmoid"``.
            n_group: Number of expert groups for grouped routing. Defaults to 1.
            topk_group: Number of top groups to keep. Defaults to 1.
            mup_enabled: Whether MuP input scaling is enabled. Defaults to ``True``.
            pad_token_id: Padding token ID. Defaults to ``None``.
            eos_token_id: End-of-sequence token ID(s). Defaults to ``None``.
            bos_token_id: Beginning-of-sequence token ID. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
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
        self.head_dim = int(head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.layer_types = layer_types or ["full_attention"] * self.num_hidden_layers
        self.sliding_window = int(sliding_window)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_shared_experts = int(num_shared_experts)
        self.num_dense_layers = int(num_dense_layers)
        self.route_norm = bool(route_norm)
        self.route_scale = float(route_scale)
        self.score_func = str(score_func)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.mup_enabled = bool(mup_enabled)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("AfmoeConfig",)
