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

"""GraniteMoE configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("granitemoe")
class GraniteMoeConfig(EasyMLXBaseConfig):
    """Configuration for the GraniteMoE transformer model.

    GraniteMoE extends the Granite architecture with Mixture-of-Experts
    (MoE) routing using ``SwitchGLU``, where each token is routed to
    ``num_experts_per_tok`` experts out of ``num_local_experts`` total.
    Retains Granite's custom scaling multipliers for embeddings,
    attention, residuals, and logits.
    Registered as model type ``"granitemoe"``.

    Attributes:
        model_type: Identifier string (``"granitemoe"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MLP intermediate dimensionality per expert.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        num_local_experts: Total number of experts in each MoE layer.
        num_experts_per_tok: Number of experts activated per token.
        rms_norm_eps: RMSNorm epsilon.
        max_position_embeddings: Maximum sequence length.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
        logits_scaling: Divisor applied to output logits.
        attention_multiplier: Multiplier for attention scale.
        embedding_multiplier: Multiplier for token embeddings.
        residual_multiplier: Multiplier for residual connections.

    Args:
        vocab_size: Vocabulary size. Defaults to 32000.
        hidden_size: Hidden dimensionality. Defaults to 4096.
        intermediate_size: Per-expert intermediate size. Defaults to 11008.
        num_hidden_layers: Number of layers. Defaults to 32.
        num_attention_heads: Number of heads. Defaults to 32.
        num_key_value_heads: KV heads. Defaults to ``num_attention_heads``.
        num_local_experts: Total experts per layer. Defaults to 8.
        num_experts_per_tok: Active experts per token. Defaults to 2.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
        max_position_embeddings: Max sequence length. Defaults to 2048.
        rope_theta: RoPE base. Defaults to 10000.0.
        rope_scaling: RoPE scaling config. Defaults to None.
        attention_bias: Attention bias. Defaults to False.
        mlp_bias: MLP bias. Defaults to False.
        tie_word_embeddings: Tie embeddings. Defaults to True.
        logits_scaling: Logit divisor. Defaults to 1.0.
        attention_multiplier: Attention scale factor. Defaults to 1.0.
        embedding_multiplier: Embedding scale. Defaults to 1.0.
        residual_multiplier: Residual scale. Defaults to 1.0.
        pad_token_id: Padding token id. Defaults to None.
        eos_token_id: EOS token id(s). Defaults to None.
        bos_token_id: BOS token id. Defaults to None.

    Example::

        >>> config = GraniteMoeConfig(num_local_experts=8, num_experts_per_tok=2)
        >>> config.model_type
        'granitemoe'
    """

    model_type = "granitemoe"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        logits_scaling: float = 1.0,
        attention_multiplier: float = 1.0,
        embedding_multiplier: float = 1.0,
        residual_multiplier: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
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
        self.num_local_experts = int(num_local_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.logits_scaling = float(logits_scaling)
        self.attention_multiplier = float(attention_multiplier)
        self.embedding_multiplier = float(embedding_multiplier)
        self.residual_multiplier = float(residual_multiplier)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "GraniteMoeConfig"
