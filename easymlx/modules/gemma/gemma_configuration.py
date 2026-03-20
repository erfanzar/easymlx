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

"""Gemma configuration (serving/inference only).

This module defines the configuration class for the Gemma model family.
Gemma is architecturally similar to Llama but uses GELU activation,
embedding scaling by sqrt(hidden_size), and ties word embeddings by default.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra.factory import register_config
from easymlx.modules.llama import LlamaConfig


@register_config("gemma")
class GemmaConfig(LlamaConfig):
    """Configuration for the Gemma transformer model.

    Extends ``LlamaConfig`` with Gemma-specific defaults. Embeddings
    are scaled by ``sqrt(hidden_size)`` and GELU activation is used
    instead of SiLU. Word embeddings are tied by default.
    Registered as model type ``"gemma"``.

    Attributes:
        model_type: Identifier string (``"gemma"``).

    Args:
        vocab_size: Vocabulary size. Defaults to 256000.
        hidden_size: Hidden dimensionality. Defaults to 3072.
        intermediate_size: Feed-forward intermediate size. Defaults to 24576.
        num_hidden_layers: Number of decoder layers. Defaults to 28.
        num_attention_heads: Number of query attention heads. Defaults to 16.
        num_key_value_heads: Number of key/value heads for GQA. Defaults
            to ``num_attention_heads`` if not set.
        head_dim: Per-head dimensionality. Defaults to 256.
        max_position_embeddings: Maximum sequence length for RoPE.
            Defaults to 8192.
        rms_norm_eps: Epsilon for RMSNorm layers. Defaults to 1e-6.
        rope_theta: Base frequency for rotary position embeddings.
            Defaults to 10000.0.
        rope_traditional: Whether to use the traditional RoPE layout.
            Defaults to False.
        rope_scaling: Optional RoPE scaling configuration dictionary.
            Defaults to None.
        attention_bias: Whether attention projections include bias terms.
            Defaults to False.
        mlp_bias: Whether MLP projections include bias terms.
            Defaults to False.
        tie_word_embeddings: Whether to tie input and output embeddings.
            Defaults to True.
        pad_token_id: Padding token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.

    Example::

        >>> config = GemmaConfig(hidden_size=2048)
        >>> config.model_type
        'gemma'
    """

    model_type = "gemma"

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
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )


__all__ = "GemmaConfig"
