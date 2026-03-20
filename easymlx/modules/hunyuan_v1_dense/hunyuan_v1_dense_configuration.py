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

"""Hunyuan V1 Dense configuration for EasyMLX inference.

Hunyuan V1 Dense is a dense (non-MoE) variant of the Hunyuan model family,
using Dynamic NTK-Alpha RoPE scaling, QK normalization, and a standard
SwiGLU MLP.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("hunyuan_v1_dense")
class HunyuanV1DenseConfig(EasyMLXBaseConfig):
    """Configuration for the Hunyuan V1 Dense transformer model.

    Attributes:
        model_type: Identifier string (``"hunyuan_v1_dense"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        max_position_embeddings: Maximum sequence length.
        attention_bias: Whether attention projections have bias.
        use_qk_norm: Whether to apply QK normalization.
        rope_scaling: RoPE scaling configuration.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "hunyuan_v1_dense"

    def __init__(
        self,
        *,
        vocab_size: int = 128256,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        attention_bias: bool = False,
        use_qk_norm: bool = True,
        rope_scaling: dict[str, tp.Any] | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize HunyuanV1DenseConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA. Defaults to
                ``num_attention_heads`` if ``None``.
            head_dim: Per-head dimensionality. Auto-derived if ``None``.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            max_position_embeddings: Maximum supported sequence length.
            attention_bias: Whether attention projections have bias terms.
            use_qk_norm: Whether to apply per-head QK normalization.
            rope_scaling: RoPE scaling configuration dict.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Forwarded to the base config.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_bias = bool(attention_bias)
        self.use_qk_norm = bool(use_qk_norm)
        self.rope_scaling = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "HunyuanV1DenseConfig"
