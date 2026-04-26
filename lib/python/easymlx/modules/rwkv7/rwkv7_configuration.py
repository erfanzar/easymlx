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

"""RWKV7 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("rwkv7")
class Rwkv7Config(EasyMLXBaseConfig):
    """Configuration for the RWKV7 linear attention / recurrent model.

    Attributes:
        model_type: Identifier string (``"rwkv7"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states (n_embd).
        num_hidden_layers: Number of RWKV layers (n_layer).
        head_size: Per-head dimensionality.
        num_heads: Number of heads (hidden_size // head_size).
        intermediate_size: Channel-mixing MLP intermediate size.
        norm_eps: LayerNorm epsilon.
        a_low_rank_dim: Low-rank dimension for the ``a`` projection.
        v_low_rank_dim: Low-rank dimension for the ``v`` projection.
        gate_low_rank_dim: Low-rank dimension for the gate projection.
        decay_low_rank_dim: Low-rank dimension for the decay projection.
        rescale_every: Layer rescaling interval (0 disables).
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "rwkv7"

    def __init__(
        self,
        *,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        head_size: int = 64,
        num_heads: int | None = None,
        intermediate_size: int = 5632,
        norm_eps: float = 1e-5,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 128,
        decay_low_rank_dim: int = 64,
        rescale_every: int = 0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize RWKV7 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states (n_embd).
            num_hidden_layers: Number of RWKV layers (n_layer).
            head_size: Per-head dimensionality.
            num_heads: Number of heads. If None, computed as
                ``hidden_size // head_size``.
            intermediate_size: Channel-mixing MLP intermediate size.
            norm_eps: Epsilon for LayerNorm.
            a_low_rank_dim: Low-rank dimension for the ``a`` (iclr)
                projection in time mixing.
            v_low_rank_dim: Low-rank dimension for the ``v`` (value
                interpolation) projection.
            gate_low_rank_dim: Low-rank dimension for the output gate
                projection.
            decay_low_rank_dim: Low-rank dimension for the decay (``w``)
                projection.
            rescale_every: Layer rescaling interval. 0 disables rescaling.
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.head_size = int(head_size)
        self.intermediate_size = int(intermediate_size)
        self.norm_eps = float(norm_eps)
        self.a_low_rank_dim = int(a_low_rank_dim)
        self.v_low_rank_dim = int(v_low_rank_dim)
        self.gate_low_rank_dim = int(gate_low_rank_dim)
        self.decay_low_rank_dim = int(decay_low_rank_dim)
        self.rescale_every = int(rescale_every)

        if num_heads is not None:
            self.num_heads = int(num_heads)
        else:
            self.num_heads = self.hidden_size // self.head_size

        self.head_dim = self.head_size
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Rwkv7Config",)
