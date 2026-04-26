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

"""Exaone4 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("exaone4")
class Exaone4Config(EasyMLXBaseConfig):
    """Configuration for the Exaone4 transformer model with sliding window attention.

    Extends Exaone with sliding window attention patterns (alternating local
    and global attention via ``sliding_window_pattern``) and optional Q/K
    RMSNorm for query/key normalization.

    Attributes:
        model_type: Identifier string (``"exaone4"``).
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        head_dim: Per-head dimension.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention uses bias.
        sliding_window: Window size for local (sliding) attention layers.
        sliding_window_pattern: String pattern like ``"LLGLLG"`` where ``L``
            means local (sliding window) and ``G`` means global (full)
            attention. Repeats cyclically across layers.
        use_qk_norm: Whether to apply RMSNorm to Q and K projections.
    """

    model_type = "exaone4"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int = 128,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        sliding_window: int | None = None,
        sliding_window_pattern: str | None = None,
        use_qk_norm: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Exaone4 configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to ``102400``.
            hidden_size: Hidden dimension. Defaults to ``2048``.
            intermediate_size: MLP intermediate dimension. Defaults to ``5632``.
            num_hidden_layers: Number of decoder layers. Defaults to ``32``.
            num_attention_heads: Number of query heads. Defaults to ``32``.
            num_key_value_heads: Number of KV heads, or ``None``.
            head_dim: Per-head dimension. Defaults to ``128``.
            max_position_embeddings: Max sequence length. Defaults to ``2048``.
            rms_norm_eps: RMSNorm epsilon. Defaults to ``1e-5``.
            rope_theta: RoPE base frequency. Defaults to ``10000.0``.
            rope_scaling: Optional RoPE scaling configuration.
            attention_bias: Whether attention uses bias. Defaults to ``False``.
            sliding_window: Window size for local attention, or ``None``.
            sliding_window_pattern: Pattern string (e.g., ``"LLGLLG"``), or ``None``.
            use_qk_norm: Whether to apply Q/K RMSNorm. Defaults to ``False``.
            tie_word_embeddings: Whether to tie embeddings. Defaults to ``True``.
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
        self.head_dim = int(head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.use_qk_norm = bool(use_qk_norm)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Exaone4Config",)
