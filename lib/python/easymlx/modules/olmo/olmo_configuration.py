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

"""OLMo configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("olmo")
class OlmoConfig(EasyMLXBaseConfig):
    """Configuration for the OLMo transformer model.

    OLMo uses RoPE, SwiGLU MLP, and LayerNorm without affine parameters
    (no learnable scale/bias in normalization layers).

    Attributes:
        model_type: Identifier string (``"olmo"``).
        d_model: Dimensionality of hidden states.
        n_layers: Number of transformer layers.
        mlp_hidden_size: MLP intermediate dimensionality.
        n_heads: Number of attention heads.
        vocab_size: Size of the vocabulary.
        embedding_size: Embedding vocabulary size (defaults to vocab_size).
        rope_theta: RoPE base frequency.
        rope_traditional: Whether to use traditional RoPE layout.
        mlp_ratio: MLP expansion ratio (used if mlp_hidden_size not set).
        weight_tying: Whether to tie input/output embeddings.
    """

    model_type = "olmo"

    def __init__(
        self,
        *,
        d_model: int = 2048,
        n_layers: int = 16,
        mlp_hidden_size: int | None = None,
        n_heads: int = 16,
        vocab_size: int = 50304,
        embedding_size: int | None = None,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        mlp_ratio: int = 4,
        weight_tying: bool = False,
        **kwargs,
    ):
        """Initialize OLMo configuration.

        Args:
            d_model (int): Dimensionality of hidden states.
            n_layers (int): Number of transformer layers.
            mlp_hidden_size (int | None): MLP intermediate size. If None,
                computed as ``mlp_ratio * d_model``.
            n_heads (int): Number of attention heads.
            vocab_size (int): Size of the vocabulary.
            embedding_size (int | None): Embedding vocabulary size.
                Defaults to ``vocab_size`` when None.
            rope_theta (float): RoPE base frequency.
            rope_traditional (bool): Whether to use traditional RoPE layout.
            mlp_ratio (int): MLP expansion ratio (used if ``mlp_hidden_size``
                is not set).
            weight_tying (bool): Whether to tie input/output embeddings.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.vocab_size = int(vocab_size)
        self.embedding_size = int(embedding_size) if embedding_size is not None else self.vocab_size
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.mlp_ratio = int(mlp_ratio)
        self.weight_tying = bool(weight_tying)

        if mlp_hidden_size is not None:
            self.mlp_hidden_size = int(mlp_hidden_size)
        else:
            self.mlp_hidden_size = self.mlp_ratio * self.d_model

        self.hidden_size = self.d_model
        self.num_hidden_layers = self.n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = self.n_heads

        super().__init__(
            tie_word_embeddings=weight_tying,
            **kwargs,
        )


__all__ = ("OlmoConfig",)
