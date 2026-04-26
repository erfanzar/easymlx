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

"""ERNIE 4.5 configuration for serving and inference.

This module defines the configuration class for the ERNIE 4.5 dense model,
registered with the EasyMLX factory under the ``"ernie4_5"`` model type.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("ernie4_5")
class Ernie45Config(EasyMLXBaseConfig):
    """Configuration for the ERNIE 4.5 dense language model.

    Registered with the EasyMLX factory under the ``"ernie4_5"`` model type.

    Attributes:
        model_type: The model type identifier (``"ernie4_5"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base frequency for rotary positional embeddings.
        use_bias: Whether to use bias in linear projections.
    """

    model_type = "ernie4_5"

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize ERNIE 4.5 configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``151936``.
            hidden_size: Hidden dimension. Defaults to ``4096``.
            intermediate_size: MLP intermediate dimension. Defaults to ``11008``.
            num_hidden_layers: Number of decoder layers. Defaults to ``32``.
            num_attention_heads: Number of query heads. Defaults to ``32``.
            num_key_value_heads: Number of KV heads, or ``None``.
            head_dim: Per-head dimension, or ``None``.
            max_position_embeddings: Max sequence length. Defaults to ``32768``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-6``.
            rope_theta: RoPE base frequency. Defaults to ``10000.0``.
            use_bias: Whether to use bias in projections. Defaults to ``False``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``False``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.use_bias = bool(use_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Ernie45Config"
