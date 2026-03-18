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

"""GLM-4 MoE MLX implementation (serving/inference only).

This module implements the GLM-4 Mixture-of-Experts transformer architecture
on MLX, including grouped expert routing, shared experts, and optional QK
normalization.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCache, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .glm4_moe_configuration import Glm4MoeConfig

CacheView = TransformerCacheView | PageCache


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


class Glm4MoeAttention(nn.Module):
    """Multi-head attention layer for the GLM-4 MoE model.

    Implements grouped-query attention (GQA) with optional QK normalization
    and rotary position embeddings.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention scores.
        use_qk_norm: Whether QK normalization is applied.
    """

    def __init__(self, config: Glm4MoeConfig):
        """Initializes the attention layer.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = bool(config.use_qk_norm)
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = (
            get_rope(
                dims=config.rotary_dim,
                base=config.rope_theta,
                traditional=False,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )
            if config.rotary_dim > 0
            else None
        )
        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Computes multi-head attention.

        Args:
            hidden_states: Input tensor of shape
                ``(batch_size, seq_len, hidden_size)`` or
                ``(seq_len, hidden_size)`` for paged mode.
            mask: Attention mask, a mask string, or None.
            cache_view: Optional KV cache view for incremental decoding.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output tensor with the same leading dimensions as the input.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class Glm4MoeMLP(nn.Module):
    """Dense feed-forward network (SwiGLU variant) for GLM-4 MoE.

    Used as the shared expert MLP or for dense layers that do not use MoE.

    Attributes:
        hidden_size: Input/output dimensionality.
        intermediate_size: Intermediate expansion dimensionality.
    """

    def __init__(self, config: Glm4MoeConfig, hidden_size: int | None = None, intermediate_size: int | None = None):
        """Initializes the MLP layer.

        Args:
            config: Model configuration.
            hidden_size: Override for the input/output size. Defaults to
                ``config.hidden_size``.
            intermediate_size: Override for the intermediate size. Defaults to
                ``config.intermediate_size``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
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
    """Mixture-of-Experts gating network for GLM-4 MoE.

    Computes routing scores for each token to determine which experts
    should process it. Uses sigmoid gating with group-level selection.

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

    def __init__(self, config: Glm4MoeConfig):
        """Initializes the MoE gate.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.top_k = config.num_experts_per_tok
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


class MoE(nn.Module):
    """Mixture-of-Experts layer combining routed and shared experts.

    Routes tokens to a subset of routed experts via ``MoEGate`` and
    optionally adds the output of shared experts.

    Attributes:
        num_experts_per_tok: Number of routed experts per token.
        switch_mlp: The ``SwitchGLU`` layer containing all routed experts.
        gate: The gating network for expert selection.
        shared_experts: Optional shared expert MLP applied to all tokens.
    """

    def __init__(self, config: Glm4MoeConfig):
        """Initializes the MoE layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(config=config, intermediate_size=intermediate_size)
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


class Glm4MoeDecoderLayer(nn.Module):
    """Single transformer decoder layer for GLM-4 MoE.

    Applies pre-norm self-attention followed by either an MoE or dense MLP,
    with residual connections.

    Attributes:
        self_attn: Multi-head attention sub-layer.
        mlp: Either an ``MoE`` or dense ``Glm4MoeMLP`` sub-layer.
        input_layernorm: RMS normalization before attention.
        post_attention_layernorm: RMS normalization before the MLP.
    """

    def __init__(self, config: Glm4MoeConfig, layer_idx: int):
        """Initializes a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index. Layers before
                ``config.first_k_dense_replace`` use a dense MLP.
        """
        super().__init__()
        self.self_attn = Glm4MoeAttention(config)
        self.mlp = (
            MoE(config) if config.n_routed_experts and layer_idx >= config.first_k_dense_replace else Glm4MoeMLP(config)
        )
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


@register_module(task_type=TaskType.BASE_MODULE, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeModel(EasyMLXBaseModule):
    """Base GLM-4 MoE transformer model.

    Consists of token embeddings, a stack of decoder layers, and a final
    RMS normalization. Produces hidden states suitable for downstream heads.

    Attributes:
        config_class: The configuration class (``Glm4MoeConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``Glm4MoeDecoderLayer`` instances.
        norm: Final RMS normalization layer.
    """

    config_class = Glm4MoeConfig

    def __init__(self, config: Glm4MoeConfig):
        """Initializes the base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Glm4MoeDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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


@register_module(task_type=TaskType.CAUSAL_LM, config=Glm4MoeConfig, model_type="glm4_moe")
class Glm4MoeForCausalLM(BaseCausalLMModule[Glm4MoeModel, Glm4MoeConfig]):
    """GLM-4 MoE model with a causal language modeling head.

    Extends ``BaseCausalLMModule`` with GLM-4 MoE-specific weight
    sanitization for loading pretrained checkpoints with per-expert weights.

    Attributes:
        config_class: The configuration class (``Glm4MoeConfig``).
    """

    config_class = Glm4MoeConfig

    def __init__(self, config: Glm4MoeConfig):
        """Initializes the causal LM model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Glm4MoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes weights by stacking per-expert parameters into SwitchGLU format.

        Converts individual expert weight tensors (e.g.,
        ``model.layers.N.mlp.experts.E.gate_proj.weight``) into stacked
        ``SwitchGLU`` format (``model.layers.N.mlp.switch_mlp.gate_proj.weight``).
        Also removes any extra layers beyond ``num_hidden_layers``.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary with stacked expert weights.
        """
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for _src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for suffix in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{dst}.{suffix}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{dst}.{suffix}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{dst}.{suffix}"] = mx.stack(to_join)

        mpt_layer = self.config.num_hidden_layers
        weights = {k: v for k, v in weights.items() if not k.startswith(f"model.layers.{mpt_layer}")}
        return super().sanitize(weights)


__all__ = ("Glm4MoeForCausalLM", "Glm4MoeModel")
