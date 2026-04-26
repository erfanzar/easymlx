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

"""MiniCPM3 configuration for EasyMLX inference.

MiniCPM3 uses Multi-head Latent Attention (MLA) similar to DeepSeek V2,
with LoRA-based compressed KV projections, SuScaledRoPE, and depth/embedding
scaling for stable training.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("minicpm3")
class MiniCPM3Config(EasyMLXBaseConfig):
    """Configuration for the MiniCPM3 transformer model with MLA.

    Attributes:
        model_type: Identifier string (``"minicpm3"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        dim_model_base: Base dimension for depth scaling.
        intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (used for v_head_dim derivation).
        q_lora_rank: Rank for query LoRA compression.
        qk_nope_head_dim: Non-RoPE portion of query/key head dimension.
        qk_rope_head_dim: RoPE portion of query/key head dimension.
        kv_lora_rank: Rank for KV LoRA compression.
        scale_depth: Depth scaling factor.
        scale_emb: Embedding scaling factor.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rope_traditional: Whether to use traditional RoPE layout.
        rope_scaling: RoPE scaling configuration (SuScaledRoPE).
        attention_bias: Whether attention projections have bias.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "minicpm3"

    def __init__(
        self,
        *,
        vocab_size: int = 73440,
        hidden_size: int = 2560,
        dim_model_base: int = 256,
        intermediate_size: int = 6400,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 40,
        num_key_value_heads: int | None = None,
        q_lora_rank: int = 768,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        kv_lora_rank: int = 256,
        scale_depth: float = 1.4,
        scale_emb: float = 12.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        rope_traditional: bool = False,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize MiniCPM3 configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            dim_model_base (int): Base dimension for depth scaling factor
                computation (``scale_depth / sqrt(num_hidden_layers)``).
            intermediate_size (int): MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads. Defaults to
                ``num_attention_heads`` when None.
            q_lora_rank (int): Rank for query LoRA compression in MLA.
            qk_nope_head_dim (int): Non-RoPE portion of query/key head dimension.
            qk_rope_head_dim (int): RoPE portion of query/key head dimension.
            kv_lora_rank (int): Rank for KV LoRA compression in MLA.
            scale_depth (float): Depth scaling factor applied to each layer
                output as ``scale_depth / sqrt(num_hidden_layers)``.
            scale_emb (float): Embedding scaling factor multiplied to token
                embeddings.
            max_position_embeddings (int): Maximum sequence length.
            rms_norm_eps (float): RMSNorm epsilon for numerical stability.
            rope_theta (float): RoPE base frequency.
            rope_traditional (bool): Whether to use traditional RoPE layout.
            rope_scaling (dict[str, Any] | None): SuScaledRoPE configuration
                with ``short_factor``, ``long_factor``, and
                ``original_max_position_embeddings``.
            attention_bias (bool): Whether attention projections have bias.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.dim_model_base = int(dim_model_base)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.q_lora_rank = int(q_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.kv_lora_rank = int(kv_lora_rank)
        self.scale_depth = float(scale_depth)
        self.scale_emb = float(scale_emb)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "MiniCPM3Config"
