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

"""OpenELM configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("openelm")
class OpenELMConfig(EasyMLXBaseConfig):
    """Configuration for the OpenELM transformer model.

    OpenELM uses per-layer variable numbers of query heads, KV heads,
    and FFN sizes, with optional Q/K RMSNorm and SwiGLU MLP.

    Attributes:
        model_type: Identifier string (``"openelm"``).
        head_dim: Per-head dimensionality.
        num_transformer_layers: Number of transformer decoder layers.
        model_dim: Model hidden dimensionality.
        vocab_size: Size of the vocabulary.
        ffn_dim_divisor: Divisor for making FFN sizes divisible.
        num_query_heads: Per-layer list of query head counts.
        num_kv_heads: Per-layer list of KV head counts.
        ffn_multipliers: Per-layer list of FFN size multipliers.
        ffn_with_glu: Whether to use gated linear unit in FFN.
        normalize_qk_projections: Whether to apply RMSNorm to Q and K.
        share_input_output_layers: Whether to tie embeddings (=tie_word_embeddings).
        rms_norm_eps: RMSNorm epsilon.
        rope_freq_constant: RoPE base frequency.
    """

    model_type = "openelm"

    def __init__(
        self,
        *,
        head_dim: int = 64,
        num_transformer_layers: int = 16,
        model_dim: int = 2048,
        vocab_size: int = 32000,
        ffn_dim_divisor: int = 256,
        num_query_heads: list[int] | None = None,
        num_kv_heads: list[int] | None = None,
        ffn_multipliers: list[float] | None = None,
        ffn_with_glu: bool = True,
        normalize_qk_projections: bool = True,
        share_input_output_layers: bool = True,
        rms_norm_eps: float = 1e-6,
        rope_freq_constant: float = 10000.0,
        max_position_embeddings: int = 2048,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize OpenELM configuration.

        Args:
            head_dim: Per-head dimensionality for attention.
            num_transformer_layers: Number of transformer decoder layers.
            model_dim: Model hidden dimensionality.
            vocab_size: Size of the token vocabulary.
            ffn_dim_divisor: Divisor used by ``make_divisible`` to round FFN
                intermediate sizes.
            num_query_heads: Per-layer list of query head counts. If None,
                defaults to ``[4] * num_transformer_layers``.
            num_kv_heads: Per-layer list of key/value head counts. If None,
                defaults to ``[2] * num_transformer_layers``.
            ffn_multipliers: Per-layer list of FFN size multipliers applied
                to ``model_dim``. If None, defaults to
                ``[4.0] * num_transformer_layers``.
            ffn_with_glu: Whether to use gated linear unit (SwiGLU) in the
                feed-forward network.
            normalize_qk_projections: Whether to apply RMSNorm to Q and K
                projections before attention.
            share_input_output_layers: Whether to tie the token embedding
                and LM head weights.
            rms_norm_eps: Epsilon for RMSNorm layers.
            rope_freq_constant: Base frequency for rotary positional
                embeddings.
            max_position_embeddings: Maximum sequence length supported.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.head_dim = int(head_dim)
        self.num_transformer_layers = int(num_transformer_layers)
        self.model_dim = int(model_dim)
        self.hidden_size = int(model_dim)
        self.vocab_size = int(vocab_size)
        self.ffn_dim_divisor = int(ffn_dim_divisor)
        self.num_query_heads = (
            list(num_query_heads) if num_query_heads is not None else [4] * self.num_transformer_layers
        )
        self.num_kv_heads = list(num_kv_heads) if num_kv_heads is not None else [2] * self.num_transformer_layers
        self.ffn_multipliers = (
            list(ffn_multipliers) if ffn_multipliers is not None else [4.0] * self.num_transformer_layers
        )
        self.ffn_with_glu = bool(ffn_with_glu)
        self.normalize_qk_projections = bool(normalize_qk_projections)
        self.share_input_output_layers = bool(share_input_output_layers)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_freq_constant = float(rope_freq_constant)
        self.max_position_embeddings = int(max_position_embeddings)
        self.num_hidden_layers = self.num_transformer_layers
        self.num_attention_heads = max(self.num_query_heads) if self.num_query_heads else 4
        self.num_key_value_heads = max(self.num_kv_heads) if self.num_kv_heads else 2

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=share_input_output_layers,
            **kwargs,
        )


__all__ = ("OpenELMConfig",)
