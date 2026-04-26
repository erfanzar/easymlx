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

"""Qwen3 configuration for serving and inference.

This module defines the configuration class for the Qwen3 model, including
support for QK normalization, sliding-window attention, and RoPE scaling,
registered with the EasyMLX factory under the ``"qwen3"`` model type.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("qwen3")
class Qwen3Config(EasyMLXBaseConfig):
    """Configuration for the Qwen3 language model.

    Registered with the EasyMLX factory under the ``"qwen3"`` model type.
    Extends Qwen2 with explicit ``head_dim``, ``attention_bias``, and
    mandatory QK normalization.

    Attributes:
        model_type: The model type identifier (``"qwen3"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        attention_bias: Whether to use bias in attention projections.
        use_sliding_window: Whether sliding-window attention is enabled.
        sliding_window: Size of the sliding attention window.
        max_window_layers: Layer index threshold for sliding window.
        attention_dropout: Dropout rate for attention weights.
        layer_types: Per-layer attention type list.
    """

    model_type = "qwen3"

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        layer_types: list["str"] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Qwen3 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Dimensionality of the MLP intermediate layer.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA. Defaults
                to ``num_attention_heads`` when ``None``.
            head_dim: Dimensionality per head. Defaults to
                ``hidden_size // num_attention_heads`` when ``None``.
            hidden_act: Activation function name.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMS normalization.
            use_cache: Whether KV caching is enabled.
            tie_word_embeddings: Whether to tie input/output embeddings.
            rope_theta: Base frequency for RoPE.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            attention_bias: Whether to use bias in attention projections.
            use_sliding_window: Whether to enable sliding-window attention.
            sliding_window: Size of the sliding attention window.
            max_window_layers: Layer index threshold for sliding window.
            attention_dropout: Dropout rate for attention weights.
            layer_types: Per-layer attention type strings. Auto-generated when ``None``.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
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
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self.max_window_layers = int(max_window_layers)

        if layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and self.use_sliding_window and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Qwen3Config"
