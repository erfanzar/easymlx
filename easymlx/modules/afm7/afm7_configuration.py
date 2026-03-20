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

"""AFM7 configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("afm7")
class Afm7Config(EasyMLXBaseConfig):
    """Configuration for the AFM7 transformer model with KV-reuse layers.

    AFM7 (Apple Foundation Model 7) uses standard transformer layers
    followed by KV-reuse layers that share cached key-value states
    from the last standard layer. Uses QK-norm and SwiGLU activation.
    Registered under model type ``"afm7"`` for automatic resolution
    via ``AutoEasyMLXModel``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 32000.
        hidden_dim: Dimensionality of hidden representations. Also
            aliased as ``hidden_size``. Defaults to 4096.
        num_layers: Number of standard transformer decoder layers.
            Defaults to 30.
        num_kv_reuse_layers: Number of KV-reuse layers appended after
            the standard layers. These layers share cached KV states
            from the last standard layer. Defaults to 2.
        num_heads: Number of query attention heads. Also aliased as
            ``num_attention_heads``. Defaults to 32.
        num_kv_heads: Number of key/value heads for grouped-query
            attention (GQA). Defaults to ``num_heads`` (MHA).
        hidden_dim_scale_factor: Scaling factor applied to ``hidden_dim``
            to compute the MLP intermediate size. Defaults to 3.25.
        rope_theta: Base frequency for rotary position embeddings.
            Defaults to 50000.0.
        rms_norm_eps: Epsilon for RMS normalization layers.
            Defaults to 1e-5.
        tie_word_embeddings: Whether to tie input and output embedding
            weights. Defaults to ``True``.
        pad_token_id: Token ID used for padding. Defaults to ``None``.
        eos_token_id: End-of-sequence token ID(s). Defaults to ``None``.
        bos_token_id: Beginning-of-sequence token ID. Defaults to ``None``.

    Attributes:
        model_type: Identifier string (``"afm7"``).
        vocab_size: Size of the vocabulary.
        hidden_dim: Dimensionality of hidden states.
        num_layers: Number of standard transformer layers.
        num_kv_reuse_layers: Number of KV-reuse layers appended after
            the standard layers.
        num_hidden_layers: Total number of layers (``num_layers + num_kv_reuse_layers``).
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality (``hidden_dim // num_heads``).
        hidden_dim_scale_factor: MLP hidden dim scaling factor.
        intermediate_size: MLP intermediate size (``hidden_dim * hidden_dim_scale_factor``).
        rope_theta: RoPE base frequency.
        rms_norm_eps: RMSNorm epsilon.

    Example::

        >>> config = Afm7Config(hidden_dim=2048, num_layers=16)
        >>> config.model_type
        'afm7'
        >>> config.num_hidden_layers
        18
    """

    model_type = "afm7"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_dim: int = 4096,
        num_layers: int = 30,
        num_kv_reuse_layers: int = 2,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        hidden_dim_scale_factor: float = 3.25,
        rope_theta: float = 50000.0,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the AFM7 configuration.

        Args:
            vocab_size: Number of tokens in the vocabulary. Defaults to 32000.
            hidden_dim: Dimensionality of hidden representations.
                Defaults to 4096.
            num_layers: Number of standard transformer decoder layers.
                Defaults to 30.
            num_kv_reuse_layers: Number of KV-reuse layers that share
                cached KV states from the last standard layer. Defaults to 2.
            num_heads: Number of query attention heads. Defaults to 32.
            num_kv_heads: Number of key/value heads for GQA. Defaults
                to ``num_heads`` (multi-head attention).
            hidden_dim_scale_factor: Scaling factor for the MLP intermediate
                size (``hidden_dim * hidden_dim_scale_factor``). Defaults to 3.25.
            rope_theta: Base frequency for rotary position embeddings.
                Defaults to 50000.0.
            rms_norm_eps: Epsilon for RMS normalization. Defaults to 1e-5.
            tie_word_embeddings: Whether to share input/output embedding
                weights. Defaults to ``True``.
            pad_token_id: Padding token ID. Defaults to ``None``.
            eos_token_id: End-of-sequence token ID(s). Defaults to ``None``.
            bos_token_id: Beginning-of-sequence token ID. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to the base config.
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        # Map to standard names for BaseCausalLMModule compatibility
        self.hidden_size = self.hidden_dim
        self.num_layers = int(num_layers)
        self.num_kv_reuse_layers = int(num_kv_reuse_layers)
        self.num_hidden_layers = self.num_layers + self.num_kv_reuse_layers
        self.num_heads = int(num_heads)
        self.num_attention_heads = self.num_heads
        self.num_kv_heads = int(num_kv_heads)
        self.num_key_value_heads = self.num_kv_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.hidden_dim_scale_factor = float(hidden_dim_scale_factor)
        self.intermediate_size = int(self.hidden_dim * self.hidden_dim_scale_factor)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = 4096

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Afm7Config",)
