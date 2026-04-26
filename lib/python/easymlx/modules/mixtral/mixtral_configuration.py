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

"""Mixtral configuration for serving and inference.

This module defines the configuration class for the Mixtral Mixture-of-Experts
model, registered with the EasyMLX factory under ``"mixtral"``.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("mixtral")
class MixtralConfig(EasyMLXBaseConfig):
    """Configuration for the Mixtral Mixture-of-Experts language model.

    Registered with the EasyMLX factory under the ``"mixtral"`` model type.

    Attributes:
        model_type: The model type identifier (``"mixtral"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        num_local_experts: Total number of routing experts.
        num_experts_per_tok: Number of experts activated per token.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        max_position_embeddings: Maximum sequence length.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "mixtral"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1e6,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Mixtral configuration.

        Args:
            vocab_size (int): Size of the token vocabulary.
            hidden_size (int): Dimensionality of the transformer hidden states.
            intermediate_size (int): MLP intermediate dimensionality per expert.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of query attention heads.
            num_key_value_heads (int | None): Number of key/value heads for GQA.
            head_dim (int | None): Per-head dimensionality, auto-derived when None.
            num_local_experts (int): Total number of routing experts.
            num_experts_per_tok (int): Number of experts activated per token.
            rms_norm_eps (float): Epsilon for RMS normalization.
            rope_theta (float): Base frequency for rotary positional embeddings.
            rope_traditional (bool): Whether to use traditional RoPE layout.
            rope_scaling (dict[str, Any] | None): RoPE scaling configuration.
            max_position_embeddings (int): Maximum sequence length.
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
        self.num_local_experts = int(num_local_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "MixtralConfig"
