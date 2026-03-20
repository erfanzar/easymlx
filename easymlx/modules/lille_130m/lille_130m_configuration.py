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

"""Lille-130M configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("lille-130m")
class Lille130mConfig(EasyMLXBaseConfig):
    """Configuration for the Lille-130M compact transformer model.

    Lille-130M uses fused QKV projections, pre-attention and pre-MLP
    RMSNorm inside each sub-block (rather than at the block level),
    traditional RoPE, SwiGLU MLP, and tied input/output embeddings.

    Attributes:
        model_type: Identifier string (``"lille-130m"``).
        vocab_size: Size of the token vocabulary. Defaults to ``32000``.
        n_embd: Embedding / hidden dimensionality. Defaults to ``768``.
        n_head: Number of attention heads. Defaults to ``12``.
        n_kv_heads: Number of KV heads for GQA. Defaults to ``n_head``.
        n_layer: Number of decoder blocks. Defaults to ``12``.
        block_size: Maximum sequence length. Defaults to ``2048``.
        layer_norm_eps: Epsilon for RMSNorm. Defaults to ``1e-5``.
        rope_theta: RoPE base frequency. Defaults to ``10000.0``.
        hidden_size: Alias for ``n_embd`` (for compatibility).
        num_hidden_layers: Alias for ``n_layer`` (for compatibility).

    Example:
        >>> config = Lille130mConfig(vocab_size=1000, n_embd=64, n_layer=2)
    """

    model_type = "lille-130m"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_head: int = 12,
        n_kv_heads: int | None = None,
        n_layer: int = 12,
        block_size: int = 2048,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        """Initialize Lille-130M configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            n_embd: Embedding / hidden dimensionality.
            n_head: Number of attention heads.
            n_kv_heads: Number of KV heads. If ``None``, defaults to ``n_head``.
            n_layer: Number of decoder blocks.
            block_size: Maximum sequence length.
            layer_norm_eps: Epsilon for RMSNorm.
            rope_theta: RoPE base frequency.
            tie_word_embeddings: Whether to tie input/output embedding weights.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if n_kv_heads is None:
            n_kv_heads = n_head

        self.vocab_size = int(vocab_size)
        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.n_kv_heads = int(n_kv_heads)
        self.n_layer = int(n_layer)
        self.block_size = int(block_size)
        self.layer_norm_eps = float(layer_norm_eps)
        self.rope_theta = float(rope_theta)

        # Provide standard aliases for BaseCausalLMModule
        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.n_kv_heads

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("Lille130mConfig",)
