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

"""GPT-NeoX configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gpt_neox")
class GPTNeoXConfig(EasyMLXBaseConfig):
    """Configuration for the GPT-NeoX transformer model.

    GPT-NeoX supports partial rotary embeddings (RoPE applied to only
    ``rotary_pct`` fraction of each head dimension) and optional
    parallel residual connections where attention and MLP are computed
    in parallel: ``h = x + attn(ln1(x)) + mlp(ln2(x))``.
    Registered as model type ``"gpt_neox"``.

    Attributes:
        model_type: Identifier string (``"gpt_neox"``).
        max_position_embeddings: Maximum sequence length.
        hidden_size: Dimensionality of hidden states.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer layers.
        layer_norm_eps: LayerNorm epsilon.
        vocab_size: Size of the vocabulary.
        rotary_emb_base: RoPE base frequency.
        rotary_pct: Fraction of ``head_dim`` to apply RoPE to (e.g., 0.25
            means only the first 25% of each head gets rotary embeddings).
        use_parallel_residual: Whether attention and MLP run in parallel.
        num_key_value_heads: Number of KV heads for GQA.

    Args:
        max_position_embeddings: Maximum sequence length. Defaults to 2048.
        hidden_size: Hidden dimensionality. Defaults to 2560.
        num_attention_heads: Number of heads. Defaults to 32.
        num_hidden_layers: Number of layers. Defaults to 32.
        layer_norm_eps: LayerNorm epsilon. Defaults to 1e-5.
        vocab_size: Vocabulary size. Defaults to 50432.
        rotary_emb_base: RoPE base frequency. Defaults to 10000.
        rotary_pct: Fraction of head_dim for RoPE. Defaults to 0.25.
        use_parallel_residual: Parallel residual mode. Defaults to True.
        num_key_value_heads: KV heads. Defaults to ``num_attention_heads``.
        tie_word_embeddings: Tie embeddings. Defaults to False.

    Example::

        >>> config = GPTNeoXConfig(rotary_pct=0.5, use_parallel_residual=True)
        >>> config.model_type
        'gpt_neox'
    """

    model_type = "gpt_neox"

    def __init__(
        self,
        *,
        max_position_embeddings: int = 2048,
        hidden_size: int = 2560,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        layer_norm_eps: float = 1e-5,
        vocab_size: int = 50432,
        rotary_emb_base: int = 10000,
        rotary_pct: float = 0.25,
        use_parallel_residual: bool = True,
        num_key_value_heads: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.max_position_embeddings = int(max_position_embeddings)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        self.layer_norm_eps = float(layer_norm_eps)
        self.vocab_size = int(vocab_size)
        self.rotary_emb_base = int(rotary_emb_base)
        self.rotary_pct = float(rotary_pct)
        self.use_parallel_residual = bool(use_parallel_residual)

        if num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        else:
            self.num_key_value_heads = int(num_key_value_heads)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("GPTNeoXConfig",)
