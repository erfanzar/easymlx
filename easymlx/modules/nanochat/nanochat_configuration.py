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

"""Nanochat configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("nanochat")
class NanochatConfig(EasyMLXBaseConfig):
    """Configuration for the Nanochat transformer model.

    Nanochat is a minimal architecture with functional (parameter-free)
    RMSNorm, Q/K RMSNorm after RoPE, ReLU^2 activation, and logit
    soft-capping.

    Attributes:
        model_type: Identifier string (``"nanochat"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        logits_soft_cap: Logit soft-capping value.
    """

    model_type = "nanochat"

    def __init__(
        self,
        *,
        vocab_size: int = 65536,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 10,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        logits_soft_cap: float = 15.0,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Nanochat configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads for GQA.
                Defaults to ``num_attention_heads``.
            head_dim (int | None): Per-head dimensionality. Auto-derived when None.
            max_position_embeddings (int): Maximum sequence length.
            rms_norm_eps (float): RMSNorm epsilon for functional norm.
            rope_theta (float): RoPE base frequency.
            logits_soft_cap (float): Logit soft-capping value applied as
                ``cap * tanh(logits / cap)``.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
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
        self.logits_soft_cap = float(logits_soft_cap)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "NanochatConfig"
