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

"""GPT-2 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gpt2")
class GPT2Config(EasyMLXBaseConfig):
    """Configuration for the GPT-2 transformer model.

    GPT-2 uses absolute (learned) position embeddings, LayerNorm,
    GELU approximate activation, and biased attention/MLP projections.
    HuggingFace checkpoints store Conv1D weights in transposed layout;
    the ``sanitize`` method in the model handles the transpose.
    Registered as model type ``"gpt2"``.

    Attributes:
        model_type: Identifier string (``"gpt2"``).
        n_ctx: Context window size.
        n_embd: Embedding dimensionality.
        n_head: Number of attention heads.
        n_layer: Number of transformer layers.
        n_positions: Maximum number of position embeddings.
        layer_norm_epsilon: LayerNorm epsilon.
        vocab_size: Size of the vocabulary.
        num_key_value_heads: Number of KV heads (defaults to ``n_head``).
        hidden_size: Alias for ``n_embd`` (compatibility).
        num_hidden_layers: Alias for ``n_layer`` (compatibility).
        num_attention_heads: Alias for ``n_head`` (compatibility).
        max_position_embeddings: Alias for ``n_positions`` (compatibility).

    Args:
        n_ctx: Context window size. Defaults to 1024.
        n_embd: Embedding/hidden dimensionality. Defaults to 768.
        n_head: Number of attention heads. Defaults to 12.
        n_layer: Number of transformer layers. Defaults to 12.
        n_positions: Maximum position embeddings. Defaults to 1024.
        layer_norm_epsilon: LayerNorm epsilon. Defaults to 1e-5.
        vocab_size: Vocabulary size. Defaults to 50257.
        num_key_value_heads: KV heads for GQA. Defaults to ``n_head``.
        tie_word_embeddings: Whether to tie embeddings. Defaults to True.
        pad_token_id: Padding token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.

    Example::

        >>> config = GPT2Config(n_embd=256, n_layer=6)
        >>> config.model_type
        'gpt2'
        >>> config.hidden_size
        256
    """

    model_type = "gpt2"

    def __init__(
        self,
        *,
        n_ctx: int = 1024,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        n_positions: int = 1024,
        layer_norm_epsilon: float = 1e-5,
        vocab_size: int = 50257,
        num_key_value_heads: int | None = None,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        self.n_ctx = int(n_ctx)
        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.n_layer = int(n_layer)
        self.n_positions = int(n_positions)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.vocab_size = int(vocab_size)
        self.num_key_value_heads = int(num_key_value_heads) if num_key_value_heads is not None else self.n_head

        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.max_position_embeddings = self.n_positions

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("GPT2Config",)
