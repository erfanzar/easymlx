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

"""StableLM configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("stablelm")
class StableLMConfig(EasyMLXBaseConfig):
    """Configuration for the StableLM transformer model.

    StableLM is a transformer architecture featuring partial Rotary
    Positional Embeddings (only a fraction of head_dim is rotated),
    optional per-head QK LayerNorm, optional parallel residual
    connections (attention and MLP applied in parallel rather than
    sequentially), SwiGLU MLP, and standard LayerNorm.

    Attributes:
        model_type: Identifier string (``"stablelm"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of hidden states.
        num_attention_heads: Number of query attention heads.
        num_hidden_layers: Number of transformer decoder layers.
        num_key_value_heads: Number of key/value heads for GQA.
        intermediate_size: SwiGLU MLP intermediate dimensionality.
        rope_theta: RoPE base frequency.
        use_qkv_bias: Whether Q/K/V projections include bias terms.
        partial_rotary_factor: Fraction of ``head_dim`` used for partial
            RoPE. For example, ``0.25`` means only the first 25% of
            dimensions receive rotary encoding.
        layer_norm_eps: Epsilon for LayerNorm numerical stability.
        use_parallel_residual: If ``True``, attention and MLP are computed
            in parallel (both from the same normalized input) rather than
            sequentially.
        qk_layernorm: If ``True``, per-head LayerNorm is applied to Q and K
            projections before attention computation.
        max_position_embeddings: Maximum sequence length supported by RoPE.

    Example:
        >>> config = StableLMConfig(hidden_size=2560, partial_rotary_factor=0.25)
        >>> config.partial_rotary_factor
        0.25
    """

    model_type = "stablelm"

    def __init__(
        self,
        *,
        vocab_size: int = 50304,
        hidden_size: int = 2560,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 6912,
        rope_theta: float = 10000.0,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        layer_norm_eps: float = 1e-5,
        use_parallel_residual: bool = False,
        qk_layernorm: bool = False,
        max_position_embeddings: int = 4096,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize a StableLM configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            num_attention_heads: Number of query attention heads.
            num_hidden_layers: Number of stacked decoder layers.
            num_key_value_heads: Number of key/value heads for GQA.
            intermediate_size: Inner dimensionality of the SwiGLU MLP.
            rope_theta: Base frequency for RoPE.
            use_qkv_bias: Whether Q/K/V projections use bias.
            partial_rotary_factor: Fraction of head_dim for partial RoPE.
            layer_norm_eps: Epsilon for LayerNorm.
            use_parallel_residual: Whether to use parallel residual connections.
            qk_layernorm: Whether to apply per-head QK LayerNorm.
            max_position_embeddings: Maximum sequence length.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end-of-sequence.
            bos_token_id: Token ID for beginning-of-sequence.
            **kwargs: Additional arguments forwarded to the base config.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        self.rope_theta = float(rope_theta)
        self.use_qkv_bias = bool(use_qkv_bias)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.layer_norm_eps = float(layer_norm_eps)
        self.use_parallel_residual = bool(use_parallel_residual)
        self.qk_layernorm = bool(qk_layernorm)
        self.max_position_embeddings = int(max_position_embeddings)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("StableLMConfig",)
