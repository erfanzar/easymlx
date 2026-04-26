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

"""GLM-4 MoE Lite configuration (serving/inference only).

This module defines the configuration for the GLM-4 MoE Lite model, which
uses Multi-head Latent Attention (MLA) with LoRA-compressed KV projections
alongside a Mixture-of-Experts feed-forward architecture.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


def _rope_scaling_from_rope_parameters(
    rope_parameters: dict[str, tp.Any] | None,
    rope_scaling: dict[str, tp.Any] | None,
) -> dict[str, tp.Any] | None:
    """Derives a RoPE scaling configuration from either explicit scaling or rope parameters.

    Normalizes the ``"type"`` key to ``"rope_type"`` for compatibility and
    extracts relevant scaling parameters from the ``rope_parameters`` dict
    when ``rope_scaling`` is not provided.

    Args:
        rope_parameters: Optional dictionary of RoPE parameters (e.g., from
            a HuggingFace config). Used as fallback when ``rope_scaling``
            is None.
        rope_scaling: Optional explicit RoPE scaling configuration. Takes
            precedence over ``rope_parameters``.

    Returns:
        A normalized RoPE scaling dictionary, or None if neither input
        provides scaling information.
    """
    if rope_scaling is not None:
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]
        return rope_scaling

    if rope_parameters is None:
        return None

    rope_scaling_out: dict[str, tp.Any] = {"rope_type": rope_parameters.get("rope_type", "default")}
    for key in (
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
        "short_factor",
        "long_factor",
        "beta_fast",
        "beta_slow",
        "extrapolation_factor",
        "attn_factor",
        "mscale",
        "mscale_all_dim",
    ):
        if key in rope_parameters:
            rope_scaling_out[key] = rope_parameters[key]
    return rope_scaling_out


@register_config("glm4_moe_lite")
class Glm4MoeLiteConfig(EasyMLXBaseConfig):
    """Configuration for the GLM-4 MoE Lite model.

    Defines hyperparameters for the MLA attention mechanism (with LoRA-compressed
    KV projections), MoE routing, and the transformer architecture. Registered
    under the model type ``"glm4_moe_lite"``.

    Attributes:
        model_type: Identifier string (``"glm4_moe_lite"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden state dimensionality.
        intermediate_size: Dense MLP intermediate size.
        moe_intermediate_size: Routed expert intermediate size.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads.
        n_shared_experts: Number of shared (always-active) experts.
        n_routed_experts: Total number of routed experts.
        routed_scaling_factor: Scaling factor for routed expert scores.
        kv_lora_rank: Rank of the LoRA-compressed KV projection.
        q_lora_rank: Rank of the LoRA-compressed Q projection, or None to
            use a direct projection.
        qk_rope_head_dim: Per-head dimension for the RoPE-applied portion.
        v_head_dim: Per-head dimension for value projections.
        qk_nope_head_dim: Per-head dimension for the non-RoPE portion.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to keep during routing.
        num_experts_per_tok: Number of experts activated per token.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length supported.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base frequency for rotary position embeddings.
        rope_interleave: Whether to use interleaved RoPE.
        rope_scaling: RoPE scaling configuration dictionary.
        mlp_layer_types: Per-layer MLP type (``"dense"`` or ``"sparse"``).
        attention_bias: Whether attention projections include bias.
        attention_dropout: Dropout rate for attention weights.
    """

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        *,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_shared_experts: int = 1,
        n_routed_experts: int = 64,
        routed_scaling_factor: float = 1.8,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 768,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int | None = 4,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_parameters: dict[str, tp.Any] | None = None,
        rope_scaling: dict[str, tp.Any] | None = None,
        rope_interleave: bool = True,
        mlp_layer_types: list["str"] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        """Initializes a GLM-4 MoE Lite configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to 154880.
            hidden_size: Hidden state dimensionality. Defaults to 2048.
            intermediate_size: Dense MLP intermediate size. Defaults to 10240.
            moe_intermediate_size: Expert intermediate size. Defaults to 1536.
            num_hidden_layers: Number of decoder layers. Defaults to 47.
            num_attention_heads: Number of attention heads. Defaults to 20.
            num_key_value_heads: Number of KV heads. Defaults to 20.
            n_shared_experts: Shared expert count. Defaults to 1.
            n_routed_experts: Routed expert count. Defaults to 64.
            routed_scaling_factor: Expert score scaling. Defaults to 1.8.
            kv_lora_rank: KV LoRA rank. Defaults to 512.
            q_lora_rank: Query LoRA rank, or None for direct projection.
                Defaults to 768.
            qk_rope_head_dim: RoPE head dimension. Defaults to 64.
            v_head_dim: Value head dimension. Defaults to 256.
            qk_nope_head_dim: Non-RoPE head dimension. Defaults to 192.
            n_group: Expert group count. Defaults to 1.
            topk_group: Top groups to keep. Defaults to 1.
            num_experts_per_tok: Experts per token. Defaults to 4.
            norm_topk_prob: Normalize routing probs. Defaults to True.
            hidden_act: Activation function name. Defaults to ``"silu"``.
            max_position_embeddings: Max sequence length. Defaults to 202752.
            initializer_range: Weight initialization range. Defaults to 0.02.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            use_cache: Enable KV caching. Defaults to True.
            pad_token_id: Padding token ID. Defaults to None.
            bos_token_id: Beginning-of-sequence token ID. Defaults to 0.
            eos_token_id: End-of-sequence token ID. Defaults to 1.
            pretraining_tp: Tensor parallelism degree used during pretraining.
                Defaults to 1.
            tie_word_embeddings: Tie input/output embeddings. Defaults to False.
            rope_theta: RoPE base frequency. Defaults to None (derived from
                ``rope_parameters`` or 10000.0).
            rope_parameters: Alternative RoPE parameter dict. Defaults to None.
            rope_scaling: RoPE scaling config. Defaults to None.
            rope_interleave: Use interleaved RoPE. Defaults to True.
            mlp_layer_types: Per-layer MLP types. Defaults to
                ``["dense"] + ["sparse"] * (num_hidden_layers - 1)``.
            attention_bias: Include attention bias. Defaults to False.
            attention_dropout: Attention dropout rate. Defaults to 0.0.
            **kwargs: Additional keyword arguments forwarded to the base class.

        Raises:
            ValueError: If ``mlp_layer_types`` length does not match
                ``num_hidden_layers`` or contains invalid values.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.n_shared_experts = int(n_shared_experts)
        self.n_routed_experts = int(n_routed_experts)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.kv_lora_rank = int(kv_lora_rank)
        self.q_lora_rank = int(q_lora_rank) if q_lora_rank is not None else None
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.num_experts_per_tok = int(num_experts_per_tok) if num_experts_per_tok is not None else None
        self.norm_topk_prob = bool(norm_topk_prob)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.pretraining_tp = int(pretraining_tp)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.rope_interleave = bool(rope_interleave)
        self.rope_scaling = _rope_scaling_from_rope_parameters(rope_parameters, rope_scaling)

        if rope_theta is None and rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.rope_theta = float(rope_theta) if rope_theta is not None else 10000.0

        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types must have length {self.num_hidden_layers}, got {len(self.mlp_layer_types)}."
            )
        for layer_type in self.mlp_layer_types:
            if layer_type not in ("dense", "sparse"):
                raise ValueError(f"Invalid layer type {layer_type}. Expected 'dense' or 'sparse'.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Glm4MoeLiteConfig"
