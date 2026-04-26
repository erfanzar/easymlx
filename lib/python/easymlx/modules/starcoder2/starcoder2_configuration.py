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

"""Starcoder2 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("starcoder2")
class Starcoder2Config(EasyMLXBaseConfig):
    """Configuration for the Starcoder2 code generation transformer model.

    Starcoder2 is a decoder-only transformer optimized for code generation.
    It features biased attention and MLP projections, standard LayerNorm
    (not RMSNorm), GELU activation (not SwiGLU), Rotary Positional
    Embeddings (RoPE), and Grouped Query Attention (GQA).

    Attributes:
        model_type: Identifier string (``"starcoder2"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality (GELU MLP).
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        norm_epsilon: Epsilon for LayerNorm numerical stability.
        rope_theta: RoPE base frequency.
        tie_word_embeddings: Whether to tie input and output embeddings.
        max_position_embeddings: Maximum sequence length supported by RoPE.

    Example:
        >>> config = Starcoder2Config(hidden_size=3072, num_hidden_layers=30)
        >>> config.model_type
        'starcoder2'
    """

    model_type = "starcoder2"

    def __init__(
        self,
        *,
        vocab_size: int = 49152,
        hidden_size: int = 3072,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        norm_epsilon: float = 1e-5,
        rope_theta: float = 100000.0,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 16384,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize a Starcoder2 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Inner dimensionality of the GELU MLP.
            num_hidden_layers: Number of stacked decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA.
            norm_epsilon: Epsilon for LayerNorm.
            rope_theta: Base frequency for RoPE.
            tie_word_embeddings: Whether to share embedding and LM head weights.
            max_position_embeddings: Maximum sequence length.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end-of-sequence.
            bos_token_id: Token ID for beginning-of-sequence.
            **kwargs: Additional arguments forwarded to the base config.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.norm_epsilon = float(norm_epsilon)
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Starcoder2Config",)
