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

"""Llama configuration (serving/inference only).

This module defines the configuration class for the Llama model family,
supporting standard dense Llama architectures with optional sliding
window attention and various RoPE scaling methods.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("llama")
class LlamaConfig(EasyMLXBaseConfig):
    """Configuration for the Llama transformer model.

    Defines all hyperparameters for the Llama architecture including
    attention, MLP, RoPE, and optional sliding window settings.
    Registered under model type ``"llama"``.

    Attributes:
        model_type: Identifier string (``"llama"``).
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
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
        layer_types: Per-layer attention types (``"full_attention"``
            or ``"sliding_attention"``).
        sliding_window: Sliding window size, or None for no sliding.
    """

    model_type = "llama"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        layer_types: list["str"] | None = None,
        sliding_window: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initializes a Llama configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to 32000.
            hidden_size: Hidden dim. Defaults to 4096.
            intermediate_size: MLP intermediate size. Defaults to 11008.
            num_hidden_layers: Decoder layers. Defaults to 32.
            num_attention_heads: Attention heads. Defaults to 32.
            num_key_value_heads: KV heads for GQA. Defaults to
                ``num_attention_heads`` (MHA).
            head_dim: Per-head dim. Defaults to None (auto-derived).
            max_position_embeddings: Max seq length. Defaults to 2048.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            rope_theta: RoPE base frequency. Defaults to 10000.0.
            rope_traditional: Traditional RoPE layout. Defaults to False.
            rope_scaling: RoPE scaling config. Defaults to None.
            attention_bias: Attention bias. Defaults to False.
            mlp_bias: MLP bias. Defaults to False.
            tie_word_embeddings: Tie embeddings. Defaults to True.
            layer_types: Per-layer attention types. Defaults to all
                ``"full_attention"``.
            sliding_window: Sliding window size. Defaults to None.
            pad_token_id: Padding token ID. Defaults to None.
            eos_token_id: EOS token ID(s). Defaults to None.
            bos_token_id: BOS token ID. Defaults to None.
            **kwargs: Additional keyword arguments for the base class.
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
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.layer_types = layer_types or ["full_attention"] * self.num_hidden_layers
        self.sliding_window = sliding_window

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "LlamaConfig"
