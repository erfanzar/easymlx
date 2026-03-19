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

"""Qwen2-MoE configuration for serving and inference.

This module defines the configuration class for the Qwen2 Mixture-of-Experts
model, including expert routing, shared expert, and sliding-window attention
parameters, registered with the EasyMLX factory under ``"qwen2_moe"``.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("qwen2_moe")
class Qwen2MoeConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen2 Mixture-of-Experts language model.

    Registered with the EasyMLX factory under the ``"qwen2_moe"`` model type.
    Extends the dense Qwen2 configuration with MoE-specific parameters such
    as expert count, routing top-k, and shared expert settings.

    Attributes:
        model_type: The model type identifier (``"qwen2_moe"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled.
        qkv_bias: Whether to use bias in QKV projections.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        use_sliding_window: Whether sliding-window attention is enabled.
        sliding_window: Size of the sliding attention window.
        max_window_layers: Layer index threshold for sliding window.
        attention_dropout: Dropout rate for attention weights.
        decoder_sparse_step: Layer interval step for MoE layers.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        shared_expert_intermediate_size: Shared expert intermediate dimensionality.
        num_experts_per_tok: Number of experts activated per token.
        num_experts: Total number of routing experts.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        output_router_logits: Whether to return router logits in the output.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        mlp_only_layers: Layer indices that use dense MLP instead of MoE.
        layer_types: Per-layer attention type list.
        gradient_checkpointing: Gradient checkpointing strategy.
        bits: Quantization bit width, or ``None`` for full precision.
    """

    model_type = "qwen2_moe"

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = 16,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 1408,
        shared_expert_intermediate_size: int = 5632,
        num_experts_per_tok: int = 4,
        num_experts: int = 60,
        norm_topk_prob: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list["int"] | None = None,
        layer_types: list["str"] | None = None,
        gradient_checkpointing: str | None = None,
        bits: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Qwen2-MoE configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA. Defaults
                to ``num_attention_heads`` when ``None``.
            hidden_act: Activation function name.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMS normalization.
            use_cache: Whether KV caching is enabled.
            tie_word_embeddings: Whether to tie input/output embeddings.
            qkv_bias: Whether to use bias in QKV projections.
            rope_theta: Base frequency for RoPE.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            use_sliding_window: Whether to enable sliding-window attention.
            sliding_window: Size of the sliding attention window.
            max_window_layers: Layer index threshold for sliding window.
            attention_dropout: Dropout rate for attention weights.
            decoder_sparse_step: Layer interval for MoE routing.
            moe_intermediate_size: Per-expert intermediate dimensionality.
            shared_expert_intermediate_size: Shared expert intermediate size.
            num_experts_per_tok: Number of experts activated per token.
            num_experts: Total number of routing experts.
            norm_topk_prob: Whether to normalize top-k routing probabilities.
            output_router_logits: Whether to return router logits.
            router_aux_loss_coef: Router auxiliary loss coefficient.
            mlp_only_layers: Indices of layers that skip MoE and use dense MLP.
            layer_types: Per-layer attention type strings. Auto-generated when ``None``.
            gradient_checkpointing: Gradient checkpointing strategy name.
            bits: Quantization bit width, or ``None`` for full precision.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.qkv_bias = bool(qkv_bias)
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_dropout = float(attention_dropout)
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window)
        self.max_window_layers = int(max_window_layers)

        self.decoder_sparse_step = int(decoder_sparse_step)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.output_router_logits = bool(output_router_logits)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)
        self.gradient_checkpointing = gradient_checkpointing
        self.bits = bits

        if layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and self.use_sliding_window and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            bits=bits,
            **kwargs,
        )


__all__ = "Qwen2MoeConfig"
