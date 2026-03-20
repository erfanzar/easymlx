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

"""Gemma3Text configuration (serving/inference only).

Gemma3Text extends Gemma2 with sliding window + full attention alternation,
Q/K LayerNorm, query_pre_attn_scalar, GELU activation, and clip_residual.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gemma3_text")
class Gemma3TextConfig(EasyMLXBaseConfig):
    """Configuration for the Gemma3Text transformer model.

    Extends Gemma2 with sliding window and full attention alternation,
    Q/K LayerNorm (offset-based: ``1 + weight``), separate RoPE base
    frequencies for local vs. global layers, ``clip_residual`` for
    FP16 overflow prevention, and four normalization layers per block.
    Registered as model type ``"gemma3_text"``.

    Attributes:
        model_type: Identifier string (``"gemma3_text"``).
        sliding_window_pattern: Every Nth layer uses full attention;
            all other layers use sliding window attention.
        sliding_window: Sliding window size for local attention layers.
        query_pre_attn_scalar: Scalar for attention scale computation
            (``scale = 1/sqrt(query_pre_attn_scalar)``).
        rope_local_base_freq: RoPE base frequency used for sliding
            window (local) attention layers.

    Args:
        vocab_size: Vocabulary size. Defaults to 262144.
        hidden_size: Hidden dimensionality. Defaults to 1152.
        intermediate_size: Feed-forward intermediate size. Defaults to 6912.
        num_hidden_layers: Number of decoder layers. Defaults to 26.
        num_attention_heads: Number of query heads. Defaults to 4.
        num_key_value_heads: Number of KV heads. Defaults to
            ``num_attention_heads``.
        head_dim: Per-head dimensionality. Defaults to 256.
        max_position_embeddings: Maximum sequence length. Defaults to 32768.
        rms_norm_eps: Epsilon for RMSNorm. Defaults to 1e-6.
        rope_theta: RoPE base frequency for full attention layers.
            Defaults to 1000000.0.
        rope_local_base_freq: RoPE base frequency for sliding window
            layers. Defaults to 10000.0.
        rope_traditional: Whether to use traditional RoPE. Defaults to False.
        rope_scaling: Optional RoPE scaling config. Defaults to None.
        attention_bias: Whether attention has bias. Defaults to False.
        mlp_bias: Whether MLP has bias. Defaults to False.
        tie_word_embeddings: Whether to tie embeddings. Defaults to True.
        query_pre_attn_scalar: Pre-attention scalar. Defaults to 256.0.
        sliding_window: Sliding window size. Defaults to 512.
        sliding_window_pattern: Full attention every N layers.
            Defaults to 6 (i.e., layer indices 5, 11, 17, ... are full).
        pad_token_id: Padding token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.

    Example::

        >>> config = Gemma3TextConfig(sliding_window=1024)
        >>> config.sliding_window_pattern
        6
    """

    model_type = "gemma3_text"

    def __init__(
        self,
        *,
        vocab_size: int = 262144,
        hidden_size: int = 1152,
        intermediate_size: int = 6912,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        head_dim: int | None = 256,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rope_local_base_freq: float = 10_000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        query_pre_attn_scalar: float = 256.0,
        sliding_window: int = 512,
        sliding_window_pattern: int = 6,
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
        self.rope_local_base_freq = float(rope_local_base_freq)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.query_pre_attn_scalar = float(query_pre_attn_scalar)
        self.sliding_window = int(sliding_window)
        self.sliding_window_pattern = int(sliding_window_pattern)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Gemma3TextConfig"
