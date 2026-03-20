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

"""Cohere2 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("cohere2")
class Cohere2Config(EasyMLXBaseConfig):
    """Configuration for the Cohere2 transformer model with sliding window attention.

    Extends the Cohere architecture with a sliding window attention
    pattern. Every ``sliding_window_pattern``-th layer uses full
    attention; other layers use sliding window attention of size
    ``sliding_window``. Registered under model type ``"cohere2"``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 256000.
        hidden_size: Dimensionality of hidden representations. Defaults to 4096.
        intermediate_size: SwiGLU MLP intermediate dimensionality. Defaults to 14336.
        num_hidden_layers: Number of transformer decoder layers. Defaults to 32.
        num_attention_heads: Number of attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 8.
        head_dim: Per-head dimensionality. Defaults to 128.
        max_position_embeddings: Maximum sequence length. Defaults to 8192.
        rope_theta: RoPE base frequency. Defaults to 50000.0.
        layer_norm_eps: LayerNorm epsilon. Defaults to 1e-5.
        logit_scale: Scaling factor applied to output logits. Defaults to 0.0625.
        attention_bias: Whether attention projections use bias. Defaults to ``False``.
        layer_norm_bias: Whether LayerNorm uses bias. Defaults to ``False``.
        use_qk_norm: Whether to apply QK normalization. Defaults to ``False``.
        sliding_window: Sliding window size for local attention. Defaults to 4096.
        sliding_window_pattern: Every Nth layer uses full attention. Defaults to 4.
        tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``True``.

    Attributes:
        model_type: Identifier string (``"cohere2"``).

    Example::

        >>> config = Cohere2Config(hidden_size=4096, sliding_window=4096)
        >>> config.model_type
        'cohere2'
    """

    model_type = "cohere2"

    def __init__(
        self,
        *,
        vocab_size: int = 256000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 8192,
        rope_theta: float = 50000.0,
        layer_norm_eps: float = 1e-05,
        logit_scale: float = 0.0625,
        attention_bias: bool = False,
        layer_norm_bias: bool = False,
        use_qk_norm: bool = False,
        sliding_window: int = 4096,
        sliding_window_pattern: int = 4,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Cohere2 configuration.

        See class docstring for full parameter documentation.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.layer_norm_eps = float(layer_norm_eps)
        self.logit_scale = float(logit_scale)
        self.attention_bias = bool(attention_bias)
        self.layer_norm_bias = bool(layer_norm_bias)
        self.use_qk_norm = bool(use_qk_norm)
        self.sliding_window = int(sliding_window)
        self.sliding_window_pattern = int(sliding_window_pattern)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Cohere2Config",)
