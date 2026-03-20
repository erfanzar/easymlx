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

"""OLMo3 configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("olmo3")
class OLMo3Config(EasyMLXBaseConfig):
    """Configuration for the OLMo3 transformer model.

    OLMo3 uses post-norm (norm after attention/MLP, not before), Q/K
    RMSNorm, sliding window attention on alternating layers, and SwiGLU.

    Attributes:
        model_type: Identifier string (``"olmo3"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        max_position_embeddings: Maximum sequence length.
        sliding_window: Sliding window size for local attention layers.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention includes bias.
        layer_types: Per-layer attention type list.
    """

    model_type = "olmo3"

    def __init__(
        self,
        *,
        vocab_size: int = 100352,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 4096,
        sliding_window: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize OLMo3 configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads for GQA.
            head_dim (int | None): Per-head dimensionality. Auto-derived when None.
            max_position_embeddings (int): Maximum sequence length.
            sliding_window (int): Sliding window size for local attention.
            rms_norm_eps (float): RMSNorm epsilon.
            rope_theta (float): RoPE base frequency.
            rope_scaling (dict[str, Any] | None): RoPE scaling config for
                full attention layers (sliding layers use unscaled RoPE).
            attention_bias (bool): Whether attention includes bias.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            layer_types (list[str] | None): Per-layer attention type list.
                Uses ``"full_attention"`` every 4th layer and
                ``"sliding_attention"`` otherwise when None.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.sliding_window = int(sliding_window)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "OLMo3Config"
