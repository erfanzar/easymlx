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

"""YouTu LLM configuration for EasyMLX.

YouTu LLM uses Multi-head Latent Attention (MLA) with optional
q_lora_rank compression, similar to DeepSeek V2 MLA but in a dense
(non-MoE) setting.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("youtu_llm")
class YouTuLLMConfig(EasyMLXBaseConfig):
    """Configuration for the YouTu LLM transformer model.

    YouTu LLM uses Multi-head Latent Attention (MLA), a technique similar
    to DeepSeek V2's MLA but in a dense (non-MoE) setting. MLA compresses
    key/value projections through a low-rank latent bottleneck
    (``kv_lora_rank``), with optional query compression (``q_lora_rank``).
    Each attention head is split into a non-RoPE portion
    (``qk_nope_head_dim``) and a RoPE portion (``qk_rope_head_dim``).

    Attributes:
        model_type: The model type identifier (``"youtu_llm"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: SwiGLU MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of KV heads (used in KV expansion).
        kv_lora_rank: Rank of the KV latent compression bottleneck.
        q_lora_rank: Rank of the query LoRA compression. When ``None``,
            queries are projected directly without compression.
        qk_rope_head_dim: Number of head dimensions that receive RoPE.
        v_head_dim: Dimensionality of each value head.
        qk_nope_head_dim: Number of head dimensions without RoPE.
        max_position_embeddings: Maximum sequence length for RoPE.
        rms_norm_eps: Epsilon for RMSNorm.
        rope_theta: RoPE base frequency.
        rope_traditional: Whether to use traditional (interleaved) RoPE layout.
        rope_scaling: Optional RoPE scaling configuration dict.
        attention_bias: Whether attention projections include bias.
        mlp_bias: Whether MLP projections include bias.

    Example:
        >>> config = YouTuLLMConfig(kv_lora_rank=512, q_lora_rank=1536)
        >>> config.kv_lora_rank
        512
    """

    model_type = "youtu_llm"

    def __init__(
        self,
        *,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1600000.0,
        rope_traditional: bool = True,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize a YouTu LLM configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the hidden states.
            intermediate_size: Inner dimensionality of the SwiGLU MLP.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads.
            kv_lora_rank: Rank of the KV latent compression.
            q_lora_rank: Rank of the query LoRA compression (``None`` for
                direct projection).
            qk_rope_head_dim: Dimensions per head for RoPE.
            v_head_dim: Dimensions per value head.
            qk_nope_head_dim: Dimensions per head without RoPE.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: RMSNorm epsilon.
            rope_theta: Base frequency for RoPE.
            rope_traditional: Whether to use traditional RoPE layout.
            rope_scaling: Optional RoPE scaling config dict.
            attention_bias: Whether attention projections use bias.
            mlp_bias: Whether MLP projections use bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional arguments forwarded to the base config.
        """
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.kv_lora_rank = int(kv_lora_rank)
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("YouTuLLMConfig",)
