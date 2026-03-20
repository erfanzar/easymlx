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

"""MiniCPM configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("minicpm")
class MiniCPMConfig(EasyMLXBaseConfig):
    """Configuration for the MiniCPM transformer model.

    MiniCPM uses depth and embedding scaling to improve training stability
    and convergence for small models.

    Attributes:
        model_type: Identifier string (``"minicpm"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        dim_model_base: Base dimension for depth scaling.
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
        scale_depth: Depth scaling factor.
        scale_emb: Embedding scaling factor.
    """

    model_type = "minicpm"

    def __init__(
        self,
        *,
        vocab_size: int = 122753,
        hidden_size: int = 2304,
        dim_model_base: int = 256,
        intermediate_size: int = 5760,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 36,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int | None = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        scale_depth: float = 1.4,
        scale_emb: float = 12.0,
        tie_word_embeddings: bool = False,
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
        self.dim_model_base = int(dim_model_base)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings) if max_position_embeddings is not None else None
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.scale_depth = float(scale_depth)
        self.scale_emb = float(scale_emb)

        """Initialize MiniCPM configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            dim_model_base: Base dimension for depth scaling computation.
            intermediate_size: SwiGLU MLP intermediate dimension.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            head_dim: Per-head dimensionality. ``None`` to auto-derive.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            rope_traditional: Whether to use traditional RoPE layout.
            rope_scaling: Optional RoPE scaling configuration dict.
            scale_depth: Depth scaling factor. Each layer's residual is
                multiplied by ``scale_depth / sqrt(num_hidden_layers)``.
            scale_emb: Embedding scaling factor applied after token lookup.
            tie_word_embeddings: Whether to tie embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "MiniCPMConfig"
