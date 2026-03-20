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

"""IQuestLoopCoder configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("iquestloopcoder")
class IQuestLoopCoderConfig(EasyMLXBaseConfig):
    """Configuration for the IQuestLoopCoder transformer model.

    IQuestLoopCoder uses a two-pass loop architecture where each
    transformer layer is applied twice: a global pass followed by
    a gated local/global mixed pass with sliding window attention.

    Attributes:
        model_type: Identifier string (``"iquestloopcoder"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention projections have bias.
        mlp_bias: Whether MLP projections have bias.
        loop_num: Number of loop passes (must be 2).
        loop_window_size: Sliding window size for local attention in pass 2.

    Example:
        >>> config = IQuestLoopCoderConfig(
        ...     vocab_size=32000, hidden_size=2048,
        ...     num_hidden_layers=16, num_attention_heads=16,
        ...     loop_num=2, loop_window_size=64,
        ... )
        >>> config.loop_window_size
        64
    """

    model_type = "iquestloopcoder"

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
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        loop_num: int = 2,
        loop_window_size: int = 64,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize IQuestLoopCoderConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            head_dim: Per-head dimensionality (auto-derived if ``None``).
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            rope_scaling: Optional RoPE scaling configuration.
            attention_bias: Whether attention projections have bias.
            mlp_bias: Whether MLP projections have bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            loop_num: Number of loop passes (must be 2).
            loop_window_size: Sliding window size for local attention.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
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
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.loop_num = int(loop_num)
        self.loop_window_size = int(loop_window_size)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("IQuestLoopCoderConfig",)
