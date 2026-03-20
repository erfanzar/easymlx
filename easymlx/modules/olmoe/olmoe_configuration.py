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

"""OLMoE configuration for serving and inference.

This module defines the configuration class for the OLMoE model,
registered with the EasyMLX factory under ``"olmoe"``.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("olmoe")
class OlmoeConfig(EasyMLXBaseConfig):
    """Configuration for the OLMoE language model.

    Registered with the EasyMLX factory under the ``"olmoe"`` model type.
    Features Llama-like attention with Q/K RMSNorm and SwitchGLU MoE.

    Attributes:
        model_type: The model type identifier (``"olmoe"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        num_experts: Total number of routing experts.
        num_experts_per_tok: Number of experts activated per token.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        max_position_embeddings: Maximum sequence length.
        attention_bias: Whether attention projections include bias.
        mlp_bias: Whether MLP projections include bias.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "olmoe"

    def __init__(
        self,
        *,
        vocab_size: int = 50304,
        hidden_size: int = 2048,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 4096,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        norm_topk_prob: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize OLMoE configuration.

        Args:
            vocab_size (int): Size of the token vocabulary.
            hidden_size (int): Dimensionality of the transformer hidden states.
            intermediate_size (int): Per-expert MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of query attention heads.
            num_key_value_heads (int | None): Number of key/value heads for GQA.
            head_dim (int | None): Per-head dimensionality, auto-derived when None.
            num_experts (int): Total number of routing experts.
            num_experts_per_tok (int): Number of experts activated per token.
            rms_norm_eps (float): Epsilon for RMS normalization.
            rope_theta (float): Base frequency for rotary positional embeddings.
            rope_traditional (bool): Whether to use traditional RoPE layout.
            rope_scaling (dict[str, Any] | None): RoPE scaling configuration.
            max_position_embeddings (int): Maximum sequence length.
            attention_bias (bool): Whether attention projections include bias.
            mlp_bias (bool): Whether MLP projections include bias.
            norm_topk_prob (bool): Whether to normalize top-k routing probs
                so they sum to 1.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = head_dim
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.norm_topk_prob = bool(norm_topk_prob)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "OlmoeConfig"
