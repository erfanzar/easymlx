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

"""BitNet configuration for EasyMLX inference.

BitNet is a 1-bit quantized transformer using ternary BitLinear layers,
RMSNorm, and ReLU^2 activation. The architecture is similar to Llama
but with BitLinear replacing standard nn.Linear in attention and MLP,
plus additional sub-norms after attention and MLP projections.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("bitnet")
class BitNetConfig(EasyMLXBaseConfig):
    """Configuration for the BitNet transformer model.

    BitNet is a 1-bit quantized transformer using ternary BitLinear
    layers, RMSNorm, and ReLU^2 activation with additional sub-norms
    after attention and MLP projections. Registered under model type
    ``"bitnet"``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 32000.
        hidden_size: Dimensionality of hidden representations. Defaults to 2048.
        intermediate_size: MLP intermediate dimensionality. Defaults to 5504.
        num_hidden_layers: Number of transformer decoder layers. Defaults to 24.
        num_attention_heads: Number of attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to
            ``num_attention_heads``.
        head_dim: Per-head dimensionality. If ``None``, auto-derived from
            ``hidden_size // num_attention_heads``. Defaults to ``None``.
        max_position_embeddings: Maximum sequence length. Defaults to 2048.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        rope_traditional: Whether to use traditional RoPE layout. Defaults to ``False``.
        rope_scaling: Optional RoPE scaling config dict. Defaults to ``None``.
        attention_bias: Whether attention projections have bias. Defaults to ``False``.
        mlp_bias: Whether MLP projections have bias. Defaults to ``False``.
        tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``True``.

    Attributes:
        model_type: Identifier string (``"bitnet"``).

    Example::

        >>> config = BitNetConfig(hidden_size=2048, num_hidden_layers=24)
        >>> config.model_type
        'bitnet'
    """

    model_type = "bitnet"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5504,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int | None = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the BitNet configuration.

        See class docstring for full parameter documentation.
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
        self.max_position_embeddings = int(max_position_embeddings) if max_position_embeddings is not None else None
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "BitNetConfig"
