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

"""GLM-4 MoE Lite MLX implementation (serving/inference only).

This module implements the GLM-4 MoE Lite transformer architecture on MLX,
featuring Multi-head Latent Attention (MLA) with LoRA-compressed KV projections
and Mixture-of-Experts feed-forward layers with grouped routing.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .glm4_moe_lite_configuration import Glm4MoeLiteConfig

CacheView = TransformerCacheView | PageCacheView


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Resolves an activation function by name.

    Args:
        name: Name of the activation function (case-insensitive).
            Supported values: ``"silu"``, ``"swish"``, ``"gelu"``.

    Returns:
        The corresponding MLX activation function.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


class Glm4MoeLiteAttention(nn.Module):
    """Multi-head Latent Attention (MLA) for GLM-4 MoE Lite.

    Implements attention with LoRA-compressed KV projections and optional
    LoRA-compressed Q projections. The query and key are split into
    non-RoPE (nope) and RoPE portions before attention computation.

    Attributes:
        num_heads: Number of attention heads.
        q_head_dim: Total per-head query dimension (nope + rope).
        qk_nope_head_dim: Non-RoPE portion of the QK head dimension.
        qk_rope_head_dim: RoPE portion of the QK head dimension.
        v_head_dim: Value per-head dimension.
        kv_lora_rank: Rank of the LoRA-compressed KV projection.
        use_mla_lora: Whether LoRA compression is used for Q projections.
        scale: Attention scaling factor.
    """

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the MLA attention layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.q_head_dim = int(config.qk_nope_head_dim + config.qk_rope_head_dim)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.use_mla_lora = config.q_lora_rank is not None and config.q_lora_rank > 0

        if not self.use_mla_lora:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)

        self.scale = self.q_head_dim**-0.5
        self.rope = (
            get_rope(
                dims=config.qk_rope_head_dim,
                base=config.rope_theta,
                traditional=not config.rope_interleave,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )
            if config.qk_rope_head_dim > 0
            else None
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Computes multi-head latent attention.

        Args:
            hidden_states: Input tensor of shape
                ``(batch_size, seq_len, hidden_size)``.
            mask: Attention mask, a mask string, or None.
            cache_view: Optional KV cache view for incremental decoding.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.use_mla_lora:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pe = kv[..., self.kv_lora_rank :]
        kv = kv[..., : self.kv_lora_rank]
        k_pe = k_pe.reshape(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = self.kv_b_proj(self.kv_a_layernorm(kv))
        kv = kv.reshape(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(
            0, 2, 1, 3
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        if self.rope is not None:
            q_pe = self.rope(q_pe, offset=0)
            k_pe = self.rope(k_pe, offset=0)

        if self.qk_rope_head_dim > 0:
            k_pe = mx.broadcast_to(k_pe, (batch_size, self.num_heads, seq_len, self.qk_rope_head_dim))
            query_states = mx.concatenate([q_nope, q_pe], axis=-1)
            key_states = mx.concatenate([k_nope, k_pe], axis=-1)
        else:
            query_states = q_nope
            key_states = k_nope

        attn = self.attention_performer.forward(query_states, key_states, v, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn)


class Glm4MoeLiteMLP(nn.Module):
    """Dense feed-forward network (SwiGLU variant) for GLM-4 MoE Lite.

    Used as the shared expert MLP or for dense layers that do not use MoE.

    Attributes:
        gate_proj: Linear projection for the gate path.
        up_proj: Linear projection for the up path.
        down_proj: Linear projection for the down path.
        act_fn: Activation function applied to the gate path.
    """

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the MLP layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@mx.compile
def group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """Selects top-k experts per token using grouped routing with score correction.

    This compiled function applies sigmoid gating, optional group-level
    top-k filtering, and score normalization to route tokens to experts.

    Args:
        gates: Raw gate logits of shape ``(..., n_routed_experts)``.
        e_score_correction_bias: Additive bias for expert score correction
            of shape ``(n_routed_experts,)``.
        top_k: Number of experts to select per token.
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to retain.
        routed_scaling_factor: Multiplicative scaling for final scores.
        norm_topk_prob: Whether to normalize the selected expert scores.

    Returns:
        A tuple of ``(indices, scores)`` where ``indices`` has shape
        ``(..., top_k)`` containing selected expert indices and ``scores``
        has the same shape containing the corresponding routing weights.
    """
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    scores = scores * routed_scaling_factor
    return inds, scores


class MoEGate(nn.Module):
    """Mixture-of-Experts gating network for GLM-4 MoE Lite.

    Computes routing scores for each token to determine which experts
    should process it, using sigmoid gating with group-level selection.

    Attributes:
        top_k: Number of experts selected per token.
        norm_topk_prob: Whether to normalize routing probabilities.
        n_routed_experts: Total number of routed experts.
        routed_scaling_factor: Score scaling factor.
        n_group: Number of expert groups.
        topk_group: Number of top groups to keep.
        weight: Gate projection weight matrix.
        e_score_correction_bias: Expert score correction bias.
    """

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the MoE gate.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``num_experts_per_tok`` is None.
        """
        super().__init__()
        if config.num_experts_per_tok is None:
            raise ValueError("num_experts_per_tok must be set for MoE routing")
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        """Routes tokens to experts.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            A tuple of ``(indices, scores)`` where ``indices`` contains the
            selected expert indices and ``scores`` the routing weights, both
            of shape ``(..., top_k)``.
        """
        return group_expert_select(
            hidden_states @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm4MoeLiteMoE(nn.Module):
    """Mixture-of-Experts layer for GLM-4 MoE Lite.

    Combines routed experts (via ``SwitchGLU``) with optional shared experts.

    Attributes:
        switch_mlp: The ``SwitchGLU`` layer containing all routed experts.
        gate: The gating network for expert selection.
        shared_experts: Optional shared expert MLP applied to all tokens.
    """

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the MoE layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = Glm4MoeLiteMLP(config)
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies mixture-of-experts routing and computation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        inds, scores = self.gate(hidden_states)
        out = self.switch_mlp(hidden_states, inds)
        out = (out * scores[..., None]).sum(axis=-2).astype(out.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class Glm4MoeLiteDecoderLayer(nn.Module):
    """Single transformer decoder layer for GLM-4 MoE Lite.

    Applies pre-norm MLA attention followed by either an MoE or dense MLP,
    with residual connections.

    Attributes:
        self_attn: MLA attention sub-layer.
        mlp: Either a ``Glm4MoeLiteMoE`` or dense ``Glm4MoeLiteMLP`` sub-layer.
        input_layernorm: RMS normalization before attention.
        post_attention_layernorm: RMS normalization before the MLP.
    """

    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        """Initializes a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index. Determines whether this layer
                uses a sparse (MoE) or dense MLP based on
                ``config.mlp_layer_types``.
        """
        super().__init__()
        self.self_attn = Glm4MoeLiteAttention(config)
        if config.mlp_layer_types[layer_idx] == "sparse" and config.n_routed_experts:
            self.mlp = Glm4MoeLiteMoE(config)
        else:
            self.mlp = Glm4MoeLiteMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the decoder layer forward pass.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output hidden states tensor.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteModel(EasyMLXBaseModule):
    """Base GLM-4 MoE Lite transformer model.

    Consists of token embeddings, a stack of MLA decoder layers (with
    sparse/dense MLP), and a final RMS normalization.

    Attributes:
        config_class: The configuration class (``Glm4MoeLiteConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``Glm4MoeLiteDecoderLayer`` instances.
        norm: Final RMS normalization layer.
    """

    config_class = Glm4MoeLiteConfig

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Glm4MoeLiteDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the transformer forward pass.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``
                or ``(seq_len,)`` for paged mode.
            attention_mask: Optional attention mask.
            input_embeddings: Optional pre-computed input embeddings. If
                provided, ``input_ids`` is ignored for embedding lookup.
            cache_views: Optional list of KV cache views, one per layer.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Final hidden states after all layers and normalization.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of layers.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=Glm4MoeLiteConfig, model_type="glm4_moe_lite")
class Glm4MoeLiteForCausalLM(BaseCausalLMModule[Glm4MoeLiteModel, Glm4MoeLiteConfig]):
    """GLM-4 MoE Lite model with a causal language modeling head.

    Extends ``BaseCausalLMModule`` with GLM-4 MoE Lite-specific weight
    sanitization for loading pretrained checkpoints.

    Attributes:
        config_class: The configuration class (``Glm4MoeLiteConfig``).
    """

    config_class = Glm4MoeLiteConfig

    def __init__(self, config: Glm4MoeLiteConfig):
        """Initializes the causal LM model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Glm4MoeLiteModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes weights by stacking per-expert parameters into SwitchGLU format.

        Converts individual expert weight tensors into the stacked
        ``SwitchGLU`` format expected by the model.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary with stacked expert weights.
        """
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for name in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{name}.{suffix}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{name}.{suffix}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{name}.{suffix}"] = mx.stack(to_join)
        return super().sanitize(weights)


__all__ = ("Glm4MoeLiteForCausalLM", "Glm4MoeLiteModel")
