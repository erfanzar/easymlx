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

"""Qwen3-Next configuration for serving and inference.

This module defines the configuration class for the Qwen3-Next hybrid model,
which combines full softmax attention with linear attention layers and MoE
routing, registered with the EasyMLX factory under ``"qwen3_next"``.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("qwen3_next")
class Qwen3NextConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3-Next hybrid attention model.

    Registered with the EasyMLX factory under the ``"qwen3_next"`` model type.
    Supports a mix of full softmax attention and linear attention layers,
    partial rotary embeddings, and MoE routing with shared experts.

    Attributes:
        model_type: The model type identifier (``"qwen3_next"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads (full attention).
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        attention_bias: Whether to use bias in attention projections.
        attention_dropout: Dropout rate for attention weights.
        partial_rotary_factor: Fraction of head_dim to apply RoPE to.
        layer_types: Per-layer attention type list.
        full_attention_interval: Interval for inserting full attention layers.
        linear_conv_kernel_dim: Kernel size for the 1-D convolution in linear attention.
        linear_key_head_dim: Key head dimensionality for linear attention.
        linear_value_head_dim: Value head dimensionality for linear attention.
        linear_num_key_heads: Number of key heads in linear attention.
        linear_num_value_heads: Number of value heads in linear attention.
        decoder_sparse_step: Layer interval step for MoE layers.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        shared_expert_intermediate_size: Shared expert intermediate dimensionality.
        num_experts_per_tok: Number of experts activated per token.
        num_experts: Total number of routing experts.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        output_router_logits: Whether to return router logits in the output.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        mlp_only_layers: Layer indices that use dense MLP instead of MoE.
    """

    model_type = "qwen3_next"

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.25,
        layer_types: list["str"] | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 10,
        num_experts: int = 512,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list["int"] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Qwen3-Next configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query heads for full attention.
            num_key_value_heads: Number of key/value heads for GQA.
            head_dim: Dimensionality per attention head.
            hidden_act: Activation function name.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMS normalization.
            use_cache: Whether KV caching is enabled.
            tie_word_embeddings: Whether to tie input/output embeddings.
            rope_theta: Base frequency for RoPE.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            attention_bias: Whether to use bias in attention projections.
            attention_dropout: Dropout rate for attention weights.
            partial_rotary_factor: Fraction of head_dim for partial RoPE.
            layer_types: Per-layer attention type strings. Auto-generated
                based on ``full_attention_interval`` when ``None``.
            full_attention_interval: Interval for full attention layers.
            linear_conv_kernel_dim: Conv1d kernel size in linear attention.
            linear_key_head_dim: Key head dim for linear attention.
            linear_value_head_dim: Value head dim for linear attention.
            linear_num_key_heads: Number of key heads in linear attention.
            linear_num_value_heads: Number of value heads in linear attention.
            decoder_sparse_step: Layer interval for MoE routing.
            moe_intermediate_size: Per-expert intermediate dimensionality.
            shared_expert_intermediate_size: Shared expert intermediate size.
            num_experts_per_tok: Number of experts activated per token.
            num_experts: Total number of routing experts.
            norm_topk_prob: Whether to normalize top-k routing probabilities.
            output_router_logits: Whether to return router logits.
            router_aux_loss_coef: Router auxiliary loss coefficient.
            mlp_only_layers: Indices of layers that skip MoE.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
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
        self.head_dim = int(head_dim)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.partial_rotary_factor = float(partial_rotary_factor)

        self.full_attention_interval = int(full_attention_interval)
        if layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_num_value_heads = int(linear_num_value_heads)

        self.decoder_sparse_step = int(decoder_sparse_step)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.output_router_logits = bool(output_router_logits)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rotary_dim(self) -> int:
        """Compute the number of dimensions that receive rotary embeddings.

        Returns:
            The rotary embedding dimensionality, derived from ``head_dim``
            and ``partial_rotary_factor``.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_d_inner(self) -> int:
        """Compute the inner dimensionality for the linear attention conv.

        Returns:
            The sum of ``2 * key_dim + value_dim``.
        """
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        return key_dim * 2 + value_dim

    @property
    def linear_d_state(self) -> int:
        """Compute the state dimensionality for linear attention.

        Returns:
            The linear value head dimensionality.
        """
        return int(self.linear_value_head_dim)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Determine whether a given layer uses full softmax attention.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            ``True`` if the layer uses full attention, ``False`` otherwise.
        """
        return self.layer_types[layer_idx] == "full_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Determine whether a given layer uses MoE routing.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            ``True`` if the layer uses MoE routing, ``False`` otherwise.
        """
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % max(self.decoder_sparse_step, 1) == 0


__all__ = "Qwen3NextConfig"
