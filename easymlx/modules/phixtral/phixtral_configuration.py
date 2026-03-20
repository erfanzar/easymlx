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

"""Phixtral configuration for serving and inference.

This module defines the configuration class for the Phixtral model,
registered with the EasyMLX factory under ``"phixtral"``.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("phixtral")
class PhixtralConfig(EasyMLXBaseConfig):
    """Configuration for the Phixtral language model.

    Registered with the EasyMLX factory under the ``"phixtral"`` model type.
    Uses parallel attention+MoE blocks with partial RoPE and GELU activation.

    Attributes:
        model_type: The model type identifier (``"phixtral"``).
        num_vocab: Size of the token vocabulary.
        model_dim: Model hidden dimensionality (= hidden_size).
        num_heads: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        rotary_dim: Dimensionality of partial rotary embeddings.
        num_local_experts: Total number of experts.
        num_experts_per_tok: Number of experts activated per token.
        layer_norm_eps: Epsilon for layer normalization.
    """

    model_type = "phixtral"

    def __init__(
        self,
        *,
        num_vocab: int = 51200,
        model_dim: int = 2560,
        num_heads: int = 32,
        num_layers: int = 32,
        rotary_dim: int = 32,
        num_local_experts: int = 4,
        num_experts_per_tok: int = 2,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize Phixtral configuration.

        Args:
            num_vocab: Size of the token vocabulary.
            model_dim: Model hidden dimensionality.
            num_heads: Number of attention heads.
            num_layers: Number of transformer decoder layers.
            rotary_dim: Dimensionality of partial rotary embeddings.
            num_local_experts: Total number of experts in MoE.
            num_experts_per_tok: Number of experts activated per token.
            layer_norm_eps: Epsilon for LayerNorm.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.num_vocab = int(num_vocab)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.rotary_dim = int(rotary_dim)
        self.num_local_experts = int(num_local_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.layer_norm_eps = float(layer_norm_eps)

        # Compatibility aliases for framework utilities.
        self.vocab_size = self.num_vocab
        self.hidden_size = self.model_dim
        self.num_hidden_layers = self.num_layers
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = self.num_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "PhixtralConfig"
