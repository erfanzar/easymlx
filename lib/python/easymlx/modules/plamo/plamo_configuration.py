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

"""PLaMo configuration (serving/inference only).

PLaMo features parallel attention+MLP, shared K/V heads via n_shared_head,
and SwiGLU activation.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("plamo")
class PlamoConfig(EasyMLXBaseConfig):
    """Configuration for the PLaMo transformer model.

    Attributes:
        model_type: Identifier string (``"plamo"``).
        n_shared_head: Number of query heads that share each KV head.
        rope_traditional: Whether to use traditional RoPE layout.
    """

    model_type = "plamo"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        n_shared_head: int = 8,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize PLaMo configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: SwiGLU MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            n_shared_head: Number of query heads that share each KV head.
                The number of KV heads is computed as
                ``ceil(num_attention_heads / n_shared_head)``.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: Base frequency for RoPE.
            rope_traditional: Whether to use the traditional RoPE layout.
            attention_bias: Whether attention projections use bias.
            mlp_bias: Whether MLP projections use bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.n_shared_head = int(n_shared_head)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        import math

        self.num_key_value_heads = math.ceil(num_attention_heads / n_shared_head)
        self.head_dim = hidden_size // num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "PlamoConfig"
