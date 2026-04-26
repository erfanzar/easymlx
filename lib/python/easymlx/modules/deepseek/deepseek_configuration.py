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

"""DeepSeek configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("deepseek")
class DeepseekConfig(EasyMLXBaseConfig):
    """Configuration for the DeepSeek transformer model with MoE support.

    DeepSeek uses standard GQA attention with RoPE and an optional
    Mixture-of-Experts feed-forward with shared experts. MoE layers
    are placed at every ``moe_layer_freq``-th layer starting from
    ``first_k_dense_replace``. Registered under model type ``"deepseek"``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 102400.
        hidden_size: Dimensionality of hidden representations. Defaults to 4096.
        intermediate_size: Dense MLP intermediate dimensionality. Defaults to 11008.
        moe_intermediate_size: Per-expert intermediate dimensionality. Defaults to 1407.
        num_hidden_layers: Number of decoder layers. Defaults to 30.
        num_attention_heads: Number of query attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 32.
        n_shared_experts: Number of shared (always-active) experts.
            Defaults to ``None`` (no shared experts).
        n_routed_experts: Total number of routing experts. Defaults to ``None``
            (pure dense model).
        num_experts_per_tok: Experts activated per token. Defaults to ``None``.
        moe_layer_freq: Frequency of MoE layers (every Nth layer). Defaults to 1.
        first_k_dense_replace: Layers before this index always use dense MLP.
            Defaults to 0.
        max_position_embeddings: Maximum sequence length. Defaults to 2048.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        rope_scaling: Optional RoPE scaling config dict. Defaults to ``None``.
        attention_bias: Whether attention projections use bias. Defaults to ``False``.
        head_dim: Per-head dimensionality. If ``None``, auto-derived.
            Defaults to ``None``.
        tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``False``.

    Attributes:
        model_type: The model type identifier (``"deepseek"``).

    Example::

        >>> config = DeepseekConfig(hidden_size=2048, n_routed_experts=64)
        >>> config.model_type
        'deepseek'
    """

    model_type = "deepseek"

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
        num_experts_per_tok: int | None = None,
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
        """Initialize the DeepSeek configuration.

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
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
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


__all__ = ("DeepseekConfig",)
