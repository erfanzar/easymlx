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

"""MiMo configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("mimo")
class MiMoConfig(EasyMLXBaseConfig):
    """Configuration for the MiMo Llama-style transformer model.

    MiMo uses biased QKV projections and multi-token prediction (MTP)
    layers that are stripped at load time for single-token inference.

    Attributes:
        model_type: Identifier string (``"mimo"``).
        vocab_size: Size of the token vocabulary. Defaults to ``32000``.
        hidden_size: Dimensionality of hidden representations. Defaults to ``4096``.
        intermediate_size: SwiGLU MLP intermediate dimension. Defaults to ``11008``.
        num_hidden_layers: Number of decoder layers. Defaults to ``32``.
        num_attention_heads: Number of attention heads. Defaults to ``32``.
        num_key_value_heads: Number of KV heads for GQA. Defaults to ``num_attention_heads``.
        max_position_embeddings: Maximum sequence length. Defaults to ``32768``.
        rms_norm_eps: Epsilon for RMSNorm. Defaults to ``1e-5``.
        rope_theta: RoPE base frequency. Defaults to ``10000.0``.
        rope_traditional: Whether to use traditional RoPE layout. Defaults to ``False``.
        rope_scaling: Optional RoPE scaling configuration.
        num_nextn_predict_layers: Number of MTP layers in the upstream model
            (stripped during sanitization). Defaults to ``2``.

    Example:
        >>> config = MiMoConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
    """

    model_type = "mimo"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        tie_word_embeddings: bool = False,
        num_nextn_predict_layers: int = 2,
        **kwargs,
    ):
        """Initialize MiMo configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden representations.
            intermediate_size: SwiGLU MLP intermediate dimension.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            rope_traditional: Whether to use traditional RoPE layout.
            rope_scaling: Optional RoPE scaling configuration.
            tie_word_embeddings: Whether to tie input/output embeddings.
            num_nextn_predict_layers: Number of MTP layers (stripped at load).
            **kwargs: Additional keyword arguments.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.num_nextn_predict_layers = int(num_nextn_predict_layers)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("MiMoConfig",)
