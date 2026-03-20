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

"""Gemma2 configuration (serving/inference only).

This module defines the configuration class for the Gemma2 model family.
Gemma2 extends Gemma with attention logit softcapping, final logit softcapping,
query pre-attention scalar, and four normalization layers per decoder block.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gemma2")
class Gemma2Config(EasyMLXBaseConfig):
    """Configuration for the Gemma2 transformer model.

    Gemma2 extends Gemma with four RMSNorm layers per decoder block
    (input, post-attention, pre-feedforward, post-feedforward),
    attention logit softcapping to stabilize training, final logit
    softcapping on output logits, and ``query_pre_attn_scalar`` for
    attention scale computation instead of ``1/sqrt(head_dim)``.
    Registered as model type ``"gemma2"``.

    Attributes:
        model_type: Identifier string (``"gemma2"``).
        attn_logit_softcapping: Softcapping value for attention logits.
        final_logit_softcapping: Softcapping value for final output logits.
        query_pre_attn_scalar: Scalar used to compute attention scale
            as ``1/sqrt(query_pre_attn_scalar)``.

    Args:
        vocab_size: Vocabulary size. Defaults to 256000.
        hidden_size: Hidden dimensionality. Defaults to 3072.
        intermediate_size: Feed-forward intermediate size. Defaults to 24576.
        num_hidden_layers: Number of decoder layers. Defaults to 28.
        num_attention_heads: Number of query attention heads. Defaults to 16.
        num_key_value_heads: Number of key/value heads for GQA. Defaults
            to ``num_attention_heads``.
        head_dim: Per-head dimensionality. Defaults to 256.
        max_position_embeddings: Maximum sequence length. Defaults to 8192.
        rms_norm_eps: Epsilon for RMSNorm. Defaults to 1e-6.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        rope_traditional: Whether to use traditional RoPE. Defaults to False.
        rope_scaling: Optional RoPE scaling config. Defaults to None.
        attention_bias: Whether attention has bias. Defaults to False.
        mlp_bias: Whether MLP has bias. Defaults to False.
        tie_word_embeddings: Whether to tie embeddings. Defaults to True.
        attn_logit_softcapping: Softcap for attention logits. Applied as
            ``tanh(logits/cap) * cap``. Defaults to 50.0.
        final_logit_softcapping: Softcap for output logits. Defaults to 30.0.
        query_pre_attn_scalar: Pre-attention scalar for query scaling.
            Defaults to 144.0.
        pad_token_id: Padding token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.

    Example::

        >>> config = Gemma2Config(hidden_size=2048, attn_logit_softcapping=50.0)
        >>> config.model_type
        'gemma2'
    """

    model_type = "gemma2"

    def __init__(
        self,
        *,
        vocab_size: int = 256000,
        hidden_size: int = 3072,
        intermediate_size: int = 24576,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int | None = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        attn_logit_softcapping: float = 50.0,
        final_logit_softcapping: float = 30.0,
        query_pre_attn_scalar: float = 144.0,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
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
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.attn_logit_softcapping = float(attn_logit_softcapping)
        self.final_logit_softcapping = float(final_logit_softcapping)
        self.query_pre_attn_scalar = float(query_pre_attn_scalar)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Gemma2Config"
