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

"""RecurrentGemma configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("recurrent_gemma")
class RecurrentGemmaConfig(EasyMLXBaseConfig):
    """Configuration for the RecurrentGemma hybrid attention/RG-LRU model.

    RecurrentGemma (Griffin architecture) alternates between local
    sliding-window attention blocks and Real-Gated Linear Recurrent Unit
    (RG-LRU) recurrent blocks. Embeddings are scaled by sqrt(hidden_size),
    and output logits use soft capping.

    Attributes:
        model_type: Identifier string (``"recurrent_gemma"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Total number of residual blocks.
        num_attention_heads: Number of attention heads.
        head_dim: Per-head dimensionality for attention.
        num_key_value_heads: Number of KV heads for attention.
        rms_norm_eps: RMSNorm epsilon.
        lru_width: Width of the RG-LRU recurrent layer. Defaults to
            hidden_size if None.
        conv1d_width: Temporal conv1d kernel size in recurrent blocks.
        logits_soft_cap: Soft cap applied to output logits via tanh.
        attention_window_size: Sliding window size for local attention.
        block_types: List of ``"attention"`` or ``"recurrent"`` strings
            determining the type of each layer.
        embeddings_scale_by_sqrt_dim: Whether to scale embeddings by
            sqrt(hidden_size).
        rope_theta: RoPE base frequency for attention blocks.
        attention_bias: Whether attention uses bias.
    """

    model_type = "recurrent_gemma"

    def __init__(
        self,
        *,
        vocab_size: int = 256000,
        hidden_size: int = 2560,
        intermediate_size: int = 7680,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 10,
        head_dim: int = 256,
        num_key_value_heads: int = 1,
        rms_norm_eps: float = 1e-6,
        lru_width: int | None = None,
        conv1d_width: int = 4,
        logits_soft_cap: float = 30.0,
        attention_window_size: int = 2048,
        block_types: list[str] | None = None,
        _block_types: list[str] | None = None,
        embeddings_scale_by_sqrt_dim: bool = True,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize RecurrentGemma configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: MLP intermediate dimensionality.
            num_hidden_layers: Total number of residual blocks.
            num_attention_heads: Number of attention heads.
            head_dim: Per-head dimensionality for attention.
            num_key_value_heads: Number of KV heads (MQA when 1).
            rms_norm_eps: Epsilon for RMSNorm.
            lru_width: Width of the RG-LRU recurrent layer. Defaults to
                ``hidden_size`` if None.
            conv1d_width: Temporal conv1d kernel size in recurrent blocks.
            logits_soft_cap: Soft cap applied to output logits via tanh.
            attention_window_size: Sliding window size for local attention.
            block_types: List of ``"attention"`` or ``"recurrent"`` strings
                determining the type of each layer. If None, defaults to
                alternating ``["recurrent", "attention"]`` pattern.
            _block_types: Alias for ``block_types`` (for 9B variant
                compatibility).
            embeddings_scale_by_sqrt_dim: Whether to scale embeddings by
                ``sqrt(hidden_size)``.
            rope_theta: RoPE base frequency for attention blocks.
            attention_bias: Whether attention uses bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_heads = int(num_key_value_heads)
        self.rms_norm_eps = float(rms_norm_eps)
        self.lru_width = int(lru_width) if lru_width is not None else self.hidden_size
        self.conv1d_width = int(conv1d_width)
        self.logits_soft_cap = float(logits_soft_cap)
        self.attention_window_size = int(attention_window_size)
        self.embeddings_scale_by_sqrt_dim = bool(embeddings_scale_by_sqrt_dim)
        self.rope_theta = float(rope_theta)
        self.attention_bias = bool(attention_bias)

        # Handle the naming inconsistency between 2B and 9B variants
        if block_types is not None:
            self.block_types = list(block_types)
        elif _block_types is not None:
            self.block_types = list(_block_types)
        else:
            self.block_types = ["recurrent", "attention"] * (self.num_hidden_layers // 2)
            if len(self.block_types) < self.num_hidden_layers:
                self.block_types.append("recurrent")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("RecurrentGemmaConfig",)
