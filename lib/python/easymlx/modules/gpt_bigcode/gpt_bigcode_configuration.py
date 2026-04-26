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

"""GPT-BigCode configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gpt_bigcode")
class GPTBigCodeConfig(EasyMLXBaseConfig):
    """Configuration for the GPT-BigCode transformer model.

    GPT-BigCode is a GPT2-style model optimized for code generation
    with multi-query attention (MQA) support. When ``multi_query`` is
    True, a single key/value head is shared across all query heads,
    significantly reducing KV cache memory. Uses absolute position
    embeddings and LayerNorm. Registered as model type ``"gpt_bigcode"``.

    Attributes:
        model_type: Identifier string (``"gpt_bigcode"``).
        n_embd: Dimensionality of hidden states.
        n_layer: Number of transformer layers.
        n_inner: MLP intermediate dimensionality.
        n_head: Number of attention heads.
        n_positions: Maximum number of position embeddings.
        layer_norm_epsilon: LayerNorm epsilon.
        vocab_size: Size of the vocabulary.
        num_key_value_heads: Number of KV heads (1 if ``multi_query``).
        multi_query: Whether to use multi-query attention (1 KV head).
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
        hidden_size: Alias for ``n_embd`` (compatibility).
        num_hidden_layers: Alias for ``n_layer`` (compatibility).
        num_attention_heads: Alias for ``n_head`` (compatibility).

    Args:
        n_embd: Hidden dimensionality. Defaults to 1024.
        n_layer: Number of layers. Defaults to 24.
        n_inner: MLP intermediate size. Defaults to 4096.
        n_head: Number of attention heads. Defaults to 16.
        n_positions: Maximum position embeddings. Defaults to 2048.
        layer_norm_epsilon: LayerNorm epsilon. Defaults to 1e-5.
        vocab_size: Vocabulary size. Defaults to 49152.
        num_key_value_heads: KV heads. Defaults to 1 if ``multi_query``
            else ``n_head``.
        multi_query: Enable multi-query attention. Defaults to True.
        attention_bias: Attention bias. Defaults to True.
        mlp_bias: MLP bias. Defaults to True.
        tie_word_embeddings: Tie embeddings. Defaults to True.

    Example::

        >>> config = GPTBigCodeConfig(multi_query=True)
        >>> config.num_key_value_heads
        1
    """

    model_type = "gpt_bigcode"

    def __init__(
        self,
        *,
        n_embd: int = 1024,
        n_layer: int = 24,
        n_inner: int = 4096,
        n_head: int = 16,
        n_positions: int = 2048,
        layer_norm_epsilon: float = 1e-5,
        vocab_size: int = 49152,
        num_key_value_heads: int | None = None,
        multi_query: bool = True,
        attention_bias: bool = True,
        mlp_bias: bool = True,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.n_embd = int(n_embd)
        self.n_layer = int(n_layer)
        self.n_inner = int(n_inner)
        self.n_head = int(n_head)
        self.n_positions = int(n_positions)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.vocab_size = int(vocab_size)
        self.multi_query = bool(multi_query)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        if num_key_value_heads is None:
            self.num_key_value_heads = 1 if self.multi_query else self.n_head
        else:
            self.num_key_value_heads = int(num_key_value_heads)

        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("GPTBigCodeConfig",)
