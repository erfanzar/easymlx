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

"""Step 3.5 MLX model implementation for serving and inference.

Step 3.5 is a Mixture-of-Experts model with:
- Sliding window attention on designated layers
- Zero-centered RMSNorm
- Head-wise attention gating
- Clamped SwiGLU activations
- Sigmoid-scored MoE routing with router bias
- Shared expert alongside routed experts
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .step3p5_configuration import Step3p5Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an ``mx.array`` of dtype ``int32``.

    Args:
        values: Input values. Accepts ``mx.array``, sequences, or ``None``.

    Returns:
        An ``mx.array`` with ``int32`` dtype, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x: mx.array, gate: mx.array, limit: mx.array) -> mx.array:
    """Compute clamped SwiGLU: ``clip(silu(gate), max=limit) * clip(x, -limit, limit)``.

    This compiled kernel fuses the SiLU gating and element-wise clamping
    into a single operation for better Metal performance.

    Args:
        x: Up-projection output tensor.
        gate: Gate-projection output tensor (before SiLU).
        limit: Clamping bound (scalar array).

    Returns:
        Element-wise product of clamped gate and clamped input.
    """
    gate = mx.clip(nn.silu(gate), a_min=None, a_max=limit)
    x = mx.clip(x, a_min=-limit, a_max=limit)
    return gate * x


class Step3p5MLP(nn.Module):
    """SwiGLU feed-forward network with optional output clamping for Step 3.5.

    When a non-zero ``swiglu_limit`` is provided, the SiLU gate output
    and the up-projection are clamped to ``[-limit, limit]`` before their
    element-wise product. This stabilizes training/inference for large
    MoE models.

    Attributes:
        gate_proj: Linear projection for the SiLU gate.
        up_proj: Linear projection for the element-wise product branch.
        down_proj: Linear projection back to ``hidden_size``.
        limit: Clamping limit, or ``None`` if clamping is disabled.

    Example:
        >>> config = Step3p5Config(hidden_size=4096)
        >>> mlp = Step3p5MLP(config, intermediate_size=11008, swiglu_limit=20.0)
    """

    def __init__(self, config: Step3p5Config, intermediate_size: int, swiglu_limit: float = 0):
        """Initialize the SwiGLU MLP.

        Args:
            config: Model configuration (used for ``hidden_size``).
            intermediate_size: Inner dimensionality of the MLP.
            swiglu_limit: Clamping bound for the SwiGLU activation. Set to
                ``0`` or negative to disable clamping.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.limit = swiglu_limit if swiglu_limit and swiglu_limit > 0 else None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU forward pass with optional clamping.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        if self.limit is not None:
            return self.down_proj(
                _clamped_swiglu(self.up_proj(hidden_states), self.gate_proj(hidden_states), mx.array(self.limit))
            )
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@mx.compile
def _moe_gate_select(
    gates: mx.array,
    router_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Select top-k experts using sigmoid scoring with router bias.

    This compiled function computes sigmoid scores, adds a learned router
    bias for expert selection, selects the top-k experts, and optionally
    normalizes the resulting weights.

    Args:
        gates: Raw gate logits of shape ``(..., num_experts)``.
        router_bias: Learned bias added to scores before top-k selection,
            shape ``(num_experts,)``.
        top_k: Number of experts to select per token.
        routed_scaling_factor: Multiplicative factor applied to the final
            expert weights.
        norm_topk_prob: Whether to L1-normalize top-k expert weights.

    Returns:
        A tuple of ``(topk_indices, topk_weights)`` where indices have
        shape ``(..., top_k)`` and weights have the same shape.
    """
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected_scores = scores + router_bias

    topk_indices = mx.argpartition(-corrected_scores, kth=top_k - 1, axis=-1)[..., :top_k]
    topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

    if norm_topk_prob:
        topk_weights = topk_weights / (mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)

    return topk_indices, topk_weights * routed_scaling_factor


class Step3p5MoEGate(nn.Module):
    """Sigmoid-scored MoE gate with learned router bias for Step 3.5.

    Projects hidden states to expert logits, then uses sigmoid scoring
    with a learned per-expert bias to select the top-k experts. The
    bias is added only for selection ordering, not to the final weights.

    Attributes:
        top_k: Number of experts selected per token.
        n_routed_experts: Total number of routing experts.
        routed_scaling_factor: Multiplicative scaling for expert weights.
        norm_topk_prob: Whether to normalize top-k weights.
        gate: Linear projection to expert logits (no bias).
        router_bias: Learned per-expert bias for selection.

    Example:
        >>> config = Step3p5Config(moe_num_experts=288, moe_top_k=8)
        >>> gate = Step3p5MoEGate(config)
    """

    def __init__(self, config: Step3p5Config):
        """Initialize the MoE gate.

        Args:
            config: Model configuration specifying expert counts,
                top-k, scaling factor, and normalization.
        """
        super().__init__()
        self.top_k = config.moe_top_k
        self.n_routed_experts = config.moe_num_experts
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.norm_topk_prob = config.norm_expert_weight

        self.gate = nn.Linear(config.hidden_size, self.n_routed_experts, bias=False)
        self.router_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute expert selection scores and indices.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            A tuple of ``(topk_indices, topk_weights)`` with shape
            ``(..., top_k)`` each.
        """
        return _moe_gate_select(
            self.gate(x),
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Step3p5MoE(nn.Module):
    """Mixture-of-Experts block for Step 3.5 with a shared expert.

    Each token is routed to ``top_k`` experts via sigmoid scoring. The
    routed output is combined with a shared expert's output (always
    active) to produce the final MLP result.

    Attributes:
        gate: MoE gate for expert selection.
        switch_mlp: Fused SwiGLU expert bank (``SwitchGLU``).
        share_expert: Always-active shared expert MLP.

    Example:
        >>> config = Step3p5Config(moe_num_experts=288, moe_top_k=8)
        >>> moe = Step3p5MoE(config, layer_idx=1)
    """

    def __init__(self, config: Step3p5Config, layer_idx: int):
        """Initialize the MoE block.

        Args:
            config: Model configuration.
            layer_idx: Index of the current layer (used for per-layer
                SwiGLU clamping limits).
        """
        super().__init__()
        if config.swiglu_limits and layer_idx < len(config.swiglu_limits):
            config.swiglu_limits[layer_idx] or 0.0

        swiglu_limit_shared = 0.0
        if config.swiglu_limits_shared and layer_idx < len(config.swiglu_limits_shared):
            swiglu_limit_shared = config.swiglu_limits_shared[layer_idx] or 0.0

        self.gate = Step3p5MoEGate(config)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.moe_num_experts)
        self.share_expert = Step3p5MLP(
            config,
            intermediate_size=config.share_expert_dim,
            swiglu_limit=swiglu_limit_shared,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute MoE forward pass with routed + shared expert outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Combined expert output of shape ``(..., hidden_size)``.
        """
        topk_indices, topk_weights = self.gate(hidden_states)
        routed_output = self.switch_mlp(hidden_states, topk_indices)
        routed_output = (routed_output * topk_weights[..., None]).sum(axis=-2).astype(routed_output.dtype)
        return routed_output + self.share_expert(hidden_states)


class Step3p5Attention(nn.Module):
    """Attention module for Step 3.5 with QK RMSNorm and head-wise gating.

    Features zero-centered RMSNorm applied to Q and K projections, optional
    head-wise attention gating (a learned sigmoid gate per head applied to
    the attention output), per-layer RoPE theta and partial rotary factors,
    and sliding vs. full attention controlled by ``layer_types``.

    Attributes:
        is_sliding: Whether this layer uses sliding window attention.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: RMSNorm applied to query heads.
        k_norm: RMSNorm applied to key heads.
        use_head_wise_attn_gate: Whether head-wise gating is active.
        g_proj: Per-head gate projection (only when gating is enabled).
        rope: RoPE module (may use per-layer theta and partial rotation).

    Example:
        >>> config = Step3p5Config(use_head_wise_attn_gate=True)
        >>> attn = Step3p5Attention(config, layer_idx=0)
    """

    def __init__(self, config: Step3p5Config, layer_idx: int):
        """Initialize Step 3.5 attention.

        Args:
            config: Model configuration.
            layer_idx: Index of the current layer (determines sliding vs.
                full attention, per-layer RoPE theta, and partial rotary
                factors).
        """
        super().__init__()
        dim = config.hidden_size
        layer_types = config.layer_types or []
        if layer_types:
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            self.is_sliding = layer_idx % 2 == 0

        if self.is_sliding and config.attention_other_setting:
            self.num_heads = config.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = config.attention_other_setting["num_attention_groups"]
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups

        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(dim, self.num_heads, bias=False)

        rope_theta = config.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[layer_idx]

        partial_rotary_factor = 1.0
        if config.partial_rotary_factors and layer_idx < len(config.partial_rotary_factors):
            partial_rotary_factor = config.partial_rotary_factors[layer_idx]

        rope_dims = int(self.head_dim * partial_rotary_factor)

        yarn_only_types = config.yarn_only_types or []
        layer_type = (
            (config.layer_types or [])[layer_idx]
            if config.layer_types and layer_idx < len(config.layer_types)
            else "full_attention"
        )
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None
        else:
            rope_scaling = config.rope_scaling

        self.rope = get_rope(
            dims=rope_dims,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute attention with QK RMSNorm, RoPE, GQA, and optional head gating.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask (boolean array or ``None``).
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = self.q_norm(queries.reshape(B, L, self.num_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache_view.offset if cache_view is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache_view is not None:
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)

        # Repeat KV heads for GQA
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        scores = (queries * self.scale) @ keys.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3)

        if self.use_head_wise_attn_gate:
            output = output * mx.sigmoid(self.g_proj(hidden_states))[..., None]

        return self.o_proj(output.reshape(B, L, -1))


class Step3p5DecoderLayer(nn.Module):
    """Single Step 3.5 decoder layer.

    Each layer contains attention (sliding or full) and either a dense
    MLP or a MoE block, determined by ``moe_layers_enum``. Pre-norm
    residual connections use RMSNorm.

    Attributes:
        self_attn: Attention sub-layer.
        is_sliding: Whether this layer uses sliding window attention.
        is_moe_layer: Whether this layer uses MoE.
        mlp: Either a ``Step3p5MoE`` or a dense ``Step3p5MLP``.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before the MLP/MoE.

    Example:
        >>> config = Step3p5Config(num_hidden_layers=4)
        >>> layer = Step3p5DecoderLayer(config, layer_idx=1)
    """

    def __init__(self, config: Step3p5Config, layer_idx: int):
        """Initialize a Step 3.5 decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of the current layer.
        """
        super().__init__()
        self.self_attn = Step3p5Attention(config, layer_idx)
        self.is_sliding = self.self_attn.is_sliding

        moe_layers_idx: set[int] = set()
        if config.moe_layers_enum:
            moe_layers_idx = {int(i) for i in config.moe_layers_enum.strip().split(",")}
        else:
            moe_layers_idx = set(range(1, config.num_hidden_layers))

        self.is_moe_layer = layer_idx in moe_layers_idx

        if self.is_moe_layer:
            self.mlp = Step3p5MoE(config, layer_idx)
        else:
            swiglu_limit = 0.0
            if config.swiglu_limits_shared and layer_idx < len(config.swiglu_limits_shared):
                swiglu_limit = config.swiglu_limits_shared[layer_idx] or 0.0
            self.mlp = Step3p5MLP(config, intermediate_size=config.intermediate_size, swiglu_limit=swiglu_limit)

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
        """Run the decoder layer forward pass.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Step3p5Config, model_type="step3p5")
class Step3p5Model(EasyMLXBaseModule):
    """Base Step 3.5 transformer model with MoE and sliding window attention.

    A decoder-only transformer with Mixture-of-Experts on most layers,
    sliding window attention on alternating layers, head-wise attention
    gating, zero-centered RMSNorm, and clamped SwiGLU activations.

    Attributes:
        config_class: Associated configuration class (``Step3p5Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``Step3p5DecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = Step3p5Config(hidden_size=4096, num_hidden_layers=4)
        >>> model = Step3p5Model(config)
    """

    config_class = Step3p5Config

    def __init__(self, config: Step3p5Config):
        """Initialize the base Step 3.5 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step3p5DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
        """Run the forward pass through all decoder layers.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

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
                attention_mask_arr = _as_int_array(attention_mask)
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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap upstream Step 3.5 weight names and filter non-model keys.

        Performs the following transformations:
        - Remaps ``.moe.gate_proj.`` to ``.mlp.switch_mlp.gate_proj.`` (and
          similar for up_proj, down_proj, gate, router_bias, share_expert).
        - Filters out MTP (multi-token prediction) layer weights.
        - Removes rotary embedding inverse-frequency buffers.
        - Strips layers beyond ``num_hidden_layers``.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned and remapped weight dict.
        """
        remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]

        new_weights: dict[str, mx.array] = {}
        for k, v in weights.items():
            if ".mtp" in k:
                continue
            if "rotary_emb.inv_freq" in k or "rope.inv_freq" in k:
                continue
            if "model.layers." in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    if int(parts[2]) >= self.config.num_hidden_layers:
                        continue

            for src, dst in remappings:
                if src in k and dst not in k:
                    k = k.replace(src, dst)
                    break

            new_weights[k] = v

        return new_weights


@register_module(task_type=TaskType.CAUSAL_LM, config=Step3p5Config, model_type="step3p5")
class Step3p5ForCausalLM(BaseCausalLMModule[Step3p5Model, Step3p5Config]):
    """Step 3.5 causal language model with an LM head.

    Wraps ``Step3p5Model`` with a linear language-model head for
    next-token prediction.

    Attributes:
        config_class: Associated configuration class (``Step3p5Config``).

    Example:
        >>> config = Step3p5Config(hidden_size=4096, num_hidden_layers=4)
        >>> model = Step3p5ForCausalLM(config)
    """

    config_class = Step3p5Config

    def __init__(self, config: Step3p5Config):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Step3p5Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("Step3p5ForCausalLM", "Step3p5Model")
