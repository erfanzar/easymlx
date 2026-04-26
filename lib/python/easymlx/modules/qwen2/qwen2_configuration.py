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

"""Qwen2 configuration for serving and inference.

This module defines the configuration class for the Qwen2 model, including
support for sliding-window attention and RoPE scaling, registered with the
EasyMLX factory under the ``"qwen2"`` model type.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("qwen2")
class Qwen2Config(EasyMLXBaseConfig):
    """Configuration for the Qwen2 language model.

    Registered with the EasyMLX factory under the ``"qwen2"`` model type.
    Supports sliding-window attention, RoPE scaling, and per-layer attention
    type configuration.

    Attributes:
        model_type: The model type identifier (``"qwen2"``).
        hidden_size: Dimensionality of the transformer hidden states.
        num_hidden_layers: Number of transformer decoder layers.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for grouped-query attention.
        hidden_act: Activation function name for MLP layers.
        rms_norm_eps: Epsilon for RMS normalization.
        vocab_size: Size of the token vocabulary.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        use_cache: Whether KV caching is enabled.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_traditional: Whether to use the traditional RoPE formulation.
        rope_scaling: Optional dictionary specifying RoPE scaling parameters.
        use_sliding_window: Whether sliding-window attention is enabled.
        sliding_window: Size of the sliding attention window.
        max_window_layers: Layer index threshold above which sliding window is used.
        attention_dropout: Dropout rate for attention weights.
        resid_pdrop: Residual connection dropout rate.
        embd_pdrop: Embedding dropout rate.
        gradient_checkpointing: Gradient checkpointing strategy.
        fcm_min_ratio: Minimum ratio for flash cross-attention masking.
        fcm_max_ratio: Maximum ratio for flash cross-attention masking.
        use_scan_mlp: Whether to use scan-based MLP.
        scan_mlp_chunk_size: Chunk size for scan-based MLP.
        number_rep_kv: Number of KV repetitions.
        bits: Quantization bit width, or ``None`` for full precision.
        scan_layers: Whether to scan layers.
        layer_types: Per-layer attention type list.
        head_dim: Computed head dimensionality.
    """

    model_type = "qwen2"

    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        intermediate_size: int = 11008,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        vocab_size: int = 151936,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        rope_theta: float = 1000000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        gradient_checkpointing: str | None = None,
        fcm_min_ratio: float = 0.0,
        fcm_max_ratio: float = 0.0,
        use_scan_mlp: bool = False,
        scan_mlp_chunk_size: int = 1024,
        number_rep_kv: int = 1,
        bits: int | None = None,
        scan_layers: bool = True,
        layer_types: list["str"] | None = None,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Qwen2 configuration.

        Args:
            hidden_size: Dimensionality of the transformer hidden states.
            num_hidden_layers: Number of transformer decoder layers.
            intermediate_size: Dimensionality of the MLP intermediate layer.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA.
            hidden_act: Activation function name.
            rms_norm_eps: Epsilon for RMS normalization.
            vocab_size: Size of the token vocabulary.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            use_cache: Whether KV caching is enabled.
            rope_theta: Base frequency for RoPE.
            rope_traditional: Whether to use traditional RoPE formulation.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            use_sliding_window: Whether to enable sliding-window attention.
            sliding_window: Size of the sliding attention window.
            max_window_layers: Layer index threshold for sliding window.
            attention_dropout: Dropout rate for attention weights.
            resid_pdrop: Residual connection dropout rate.
            embd_pdrop: Embedding dropout rate.
            gradient_checkpointing: Gradient checkpointing strategy name.
            fcm_min_ratio: Minimum ratio for flash cross-attention masking.
            fcm_max_ratio: Maximum ratio for flash cross-attention masking.
            use_scan_mlp: Whether to use scan-based MLP.
            scan_mlp_chunk_size: Chunk size for scan-based MLP.
            number_rep_kv: Number of KV repetitions.
            bits: Quantization bit width, or ``None`` for full precision.
            scan_layers: Whether to scan layers.
            layer_types: Per-layer attention type strings. Auto-generated
                from sliding window settings when ``None``.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.intermediate_size = int(intermediate_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.hidden_act = str(hidden_act)
        self.rms_norm_eps = float(rms_norm_eps)
        self.vocab_size = int(vocab_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window)
        self.max_window_layers = int(max_window_layers)
        self.attention_dropout = float(attention_dropout)
        self.resid_pdrop = float(resid_pdrop)
        self.embd_pdrop = float(embd_pdrop)
        self.gradient_checkpointing = gradient_checkpointing
        self.fcm_min_ratio = float(fcm_min_ratio)
        self.fcm_max_ratio = float(fcm_max_ratio)
        self.use_scan_mlp = bool(use_scan_mlp)
        self.scan_mlp_chunk_size = int(scan_mlp_chunk_size)
        self.number_rep_kv = int(number_rep_kv)
        self.bits = bits
        self.scan_layers = bool(scan_layers)
        self.head_dim = self.hidden_size // self.num_attention_heads

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
            use_scan_mlp=use_scan_mlp,
            scan_mlp_chunk_size=scan_mlp_chunk_size,
            bits=bits,
            **kwargs,
        )


__all__ = "Qwen2Config"
