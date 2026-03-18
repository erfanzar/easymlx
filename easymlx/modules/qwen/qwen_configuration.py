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

"""Qwen (v1) configuration for serving and inference.

This module defines the configuration class for the original Qwen model,
registered with the EasyMLX factory under the ``"qwen"`` model type.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("qwen")
class QwenConfig(EasyMLXBaseConfig):
    """Configuration for the original Qwen language model.

    Registered with the EasyMLX factory under the ``"qwen"`` model type.

    Attributes:
        model_type: The model type identifier (``"qwen"``).
        hidden_size: Dimensionality of the transformer hidden states.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer decoder layers.
        kv_channels: Dimensionality of key/value projections per head.
        max_position_embeddings: Maximum sequence length.
        layer_norm_epsilon: Epsilon for RMS layer normalization.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        no_bias: Whether to disable bias in linear projections.
        vocab_size: Size of the token vocabulary.
        num_key_value_heads: Number of key/value heads (defaults to
            ``num_attention_heads`` for multi-head attention).
    """

    model_type = "qwen"

    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        kv_channels: int = 128,
        max_position_embeddings: int = 8192,
        layer_norm_epsilon: float = 1e-6,
        intermediate_size: int = 11008,
        no_bias: bool = True,
        vocab_size: int = 151936,
        num_key_value_heads: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Qwen configuration.

        Args:
            hidden_size: Dimensionality of the transformer hidden states.
            num_attention_heads: Number of attention heads.
            num_hidden_layers: Number of transformer decoder layers.
            kv_channels: Dimensionality of key/value projections per head.
            max_position_embeddings: Maximum sequence length.
            layer_norm_epsilon: Epsilon for RMS layer normalization.
            intermediate_size: Dimensionality of the MLP intermediate layer.
            no_bias: Whether to disable bias in linear projections.
            vocab_size: Size of the token vocabulary.
            num_key_value_heads: Number of key/value heads. Defaults to
                ``num_attention_heads`` when ``None``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        self.kv_channels = int(kv_channels)
        self.max_position_embeddings = int(max_position_embeddings)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.intermediate_size = int(intermediate_size)
        self.no_bias = bool(no_bias)
        self.vocab_size = int(vocab_size)
        self.num_key_value_heads = int(num_key_value_heads)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = "QwenConfig"
