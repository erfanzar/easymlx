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

"""Bailing MoE Linear MLX model implementation for serving and inference.

Hybrid architecture combining full attention and GLA-style linear attention
layers, with MoE feed-forward blocks and group expert selection.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .bailing_moe_linear_configuration import BailingMoeLinearConfig

CacheView = TransformerCacheView | PageCacheView


def _recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    h: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute Gated Linear Attention (GLA) recurrence.

    For each timestep t:
        ``h_t = h_{t-1} * exp(g_t) + k_t^T @ v_t``
        ``y_t = (q_t @ h_t) * scale``

    Args:
        q: Query tensor of shape ``(B, H, L, K)``.
        k: Key tensor of shape ``(B, H, L, K)``.
        v: Value tensor of shape ``(B, H, L, V)``.
        g: Gate (decay) tensor, broadcast to ``(B, 1, 1)`` per head.
        scale: Scaling factor applied to the output.
        h: Optional initial recurrent state of shape ``(B, H, K, V)``.

    Returns:
        Tuple of ``(output, final_state)`` where output has shape
        ``(B, H, L, V)`` and final_state has shape ``(B, H, K, V)``.
    """
    _B, _Hq, L, _K = q.shape
    v.shape[-1]

    outputs = []
    exp_g = mx.exp(g)[:, None, None].astype(q.dtype)
    q = q * scale
    for t in range(L):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        h_up = k_t.transpose(0, 1, 3, 2) @ v_t
        if h is not None:
            h = h * exp_g + h_up
        else:
            h = h_up
        o_t = q_t @ h
        outputs.append(o_t)

    return mx.concatenate(outputs, axis=2), h


class GroupRMSNorm(nn.Module):
    """RMS normalization applied per group.

    Splits the last dimension into groups, applies RMS normalization
    independently per group, then rescales with a learned weight.

    Attributes:
        weight: Learnable scaling weight of shape ``(dims,)``.
        groups: Number of groups to split the last dimension into.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1):
        """Initialize GroupRMSNorm.

        Args:
            dims: Total dimensionality of the input's last axis.
            eps: Epsilon for numerical stability. Defaults to 1e-5.
            groups: Number of groups. Defaults to 1.
        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply grouped RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dims)``.

        Returns:
            Normalized tensor of the same shape.
        """
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


class BailingMoeLinearFullAttention(nn.Module):
    """Full multi-head attention for global layers in Bailing MoE Linear.

    Used at every ``layer_group_size``-th layer and trailing layers.
    Identical structure to ``BailingMoeAttention`` with fused QKV,
    optional QK-norm, and partial RoPE.

    Attributes:
        use_qk_norm: Whether QK normalization is applied.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        query_key_value: Fused QKV linear projection.
        dense: Output linear projection.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.
    """

    def __init__(self, config: BailingMoeLinearConfig):
        """Initialize the full attention module.

        Args:
            config: Model configuration with attention hyperparameters.
        """
        super().__init__()
        self.use_qk_norm = config.use_qk_norm
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=int(self.head_dim * config.partial_rotary_factor),
            base=config.rope_theta,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
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
        """Run the full attention forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]

        qkv = self.query_key_value(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q.reshape(*lead, self.num_heads, self.head_dim)
        k = k.reshape(*lead, self.num_kv_heads, self.head_dim)
        v = v.reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.dense(attn.reshape(*lead, -1))


class BailingMoeLinearAttention(nn.Module):
    """GLA-style gated linear attention for non-global layers.

    Implements Gated Linear Attention with ALiBi-style decay slopes
    that vary by layer depth. Uses a gating mechanism with GroupRMSNorm
    on the output before the final projection.

    Attributes:
        layer_idx: Index of this layer in the stack.
        use_qk_norm: Whether QK normalization is applied.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (same as num_heads for linear attn).
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        query_key_value: Fused QKV linear projection.
        dense: Output linear projection.
        g_proj: Gate projection for output gating.
        g_norm: GroupRMSNorm applied to attention output.
        rope: Rotary position embedding module.

    Example::

        >>> config = BailingMoeLinearConfig(hidden_size=2048)
        >>> attn = BailingMoeLinearAttention(config, layer_idx=1)
    """

    def __init__(self, config: BailingMoeLinearConfig, layer_idx: int):
        """Initialize the GLA-style linear attention module.

        Args:
            config: Model configuration with attention hyperparameters.
            layer_idx: Index of this layer, used to compute ALiBi-style
                decay slopes.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = config.use_qk_norm
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_attention_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )
        self.g_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.g_norm = GroupRMSNorm(
            config.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
            groups=config.group_norm_size,
        )

        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=int(self.head_dim * config.partial_rotary_factor),
            base=config.rope_theta,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self._slope = self._get_slopes()

    def _get_slopes(self) -> mx.array:
        """Compute ALiBi-style decay slopes scaled by layer depth.

        Returns:
            Slope tensor of shape ``(num_heads,)`` with negative values
            representing exponential decay rates.
        """
        n = self.num_heads

        def power_of_2_slopes(n):
            return [2 ** (-(2 ** -(math.log2(n) - 3)) * (i + 1)) for i in range(n)]

        if math.log2(n).is_integer():
            slopes = power_of_2_slopes(n)
        else:
            p = 2 ** math.floor(math.log2(n))
            slopes = power_of_2_slopes(p) + power_of_2_slopes(2 * p)[::2][: n - p]

        slopes = mx.array(slopes, dtype=mx.float32)
        denom = max(1, self.num_hidden_layers - 1)
        layer_pos = max(0, self.layer_idx - 1)
        layer_factor = 1 - (layer_pos / denom) + 1e-5
        return -slopes * layer_factor

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the GLA-style linear attention forward pass.

        Computes gated linear attention with ALiBi-style decay,
        stores recurrent state in the cache for autoregressive
        generation, and applies output gating with GroupRMSNorm.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask (unused for linear attention).
            cache_view: Per-layer cache view storing the recurrent state.
            cache_metadata: Paged-attention metadata (unused).

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _D = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv_mix = qkv.reshape(
            B,
            L,
            (self.num_heads + 2 * self.num_kv_heads),
            self.head_dim,
        )
        q, k, v = mx.split(
            qkv_mix,
            [self.num_heads, self.num_heads + self.num_kv_heads],
            axis=2,
        )

        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        offset = cache_view.offset if cache_view is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        h_state = None
        if cache_view is not None:
            h_state = getattr(cache_view, "_linear_state", None)

        output, h_state = _recurrent_gla(
            q=queries,
            k=keys,
            v=values,
            g=self._slope,
            scale=self.scale,
            h=h_state,
        )

        if cache_view is not None:
            cache_view._linear_state = h_state

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.g_norm(output) * mx.sigmoid(self.g_proj(hidden_states))
        return self.dense(output)


class BailingMoeLinearDenseMLP(nn.Module):
    """SiLU-gated feed-forward MLP for dense and shared expert layers.

    Uses ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: BailingMoeLinearConfig, intermediate_size: int | None = None):
        """Initialize the dense MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimensionality.
        """
        super().__init__()
        intermediate = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=config.use_bias)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class BailingMoeLinearSparseBlock(nn.Module):
    """Mixture-of-Experts block with group expert selection.

    Routes tokens to top-k experts via ``TopKRouter`` with configurable
    scoring function and optional shared experts.

    Attributes:
        router: Top-k expert routing module.
        switch_mlp: Fused SwitchGLU expert module.
        shared_experts: Optional shared expert MLP.
    """

    def __init__(self, config: BailingMoeLinearConfig):
        """Initialize the sparse MoE block.

        Args:
            config: Model configuration with MoE hyperparameters.
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.score_function,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group,
            topk_group=config.topk_group,
            use_score_bias=config.moe_router_enable_expert_bias,
        )
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            bias=config.use_bias,
        )

        shared_dim = config.moe_shared_expert_intermediate_size or config.moe_intermediate_size
        if config.num_shared_experts > 0 and config.moe_router_enable_shared_expert:
            self.shared_experts = BailingMoeLinearDenseMLP(
                config,
                intermediate_size=shared_dim * config.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of the same shape.
        """
        inds, scores = self.router(hidden_states)
        out = self.switch_mlp(hidden_states, inds)
        out = (out * scores[..., None]).sum(axis=-2).astype(out.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class BailingMoeLinearDecoderLayer(nn.Module):
    """Single decoder layer with hybrid attention selection.

    Selects between full (global) attention and GLA-style linear
    attention based on the layer index. Every ``layer_group_size``-th
    layer and trailing layers use full attention; others use linear.

    Attributes:
        is_global: Whether this layer uses full attention.
        attention: Full or linear attention module.
        mlp: Dense MLP or MoE block.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
    """

    def __init__(self, config: BailingMoeLinearConfig, layer_idx: int):
        """Initialize the hybrid decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack.
        """
        super().__init__()
        self.is_global = (
            (layer_idx + 1) % config.layer_group_size == 0
            or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
        )

        if self.is_global:
            self.attention = BailingMoeLinearFullAttention(config)
        else:
            self.attention = BailingMoeLinearAttention(config, layer_idx=layer_idx)

        self.mlp = (
            BailingMoeLinearSparseBlock(config)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeLinearDenseMLP(config)
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
        """Run the hybrid decoder layer forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output hidden states of the same shape.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.attention(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=BailingMoeLinearConfig, model_type="bailing_moe_linear")
class BailingMoeLinearModel(EasyMLXBaseModule):
    """Base Bailing MoE Linear transformer model with hybrid attention.

    Attributes:
        config_class: The associated configuration class (``BailingMoeLinearConfig``).
        word_embeddings: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
    """

    config_class = BailingMoeLinearConfig

    def __init__(self, config: BailingMoeLinearConfig):
        """Initialize the Bailing MoE Linear base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BailingMoeLinearDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Alias for word_embeddings to match BaseCausalLMModule expectations."""
        return self.word_embeddings

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the transformer forward pass.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length does not match layers.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.word_embeddings(input_ids)

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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap upstream checkpoint keys to EasyMLX parameter names.

        Stacks per-expert weights into fused SwitchGLU format, remaps
        ``gate`` to ``router``, normalizes ``lm_head`` if configured,
        and removes rotary embedding buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with expert weights stacked and
            keys remapped.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)

        if self.config.norm_head and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= self.config.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(self.config.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

                gate_key = f"{prefix}.mlp.gate.weight"
                router_key = f"{prefix}.mlp.router.weight"
                if gate_key in weights and router_key not in weights:
                    weights[router_key] = weights.pop(gate_key)

                bias_key = f"{prefix}.mlp.gate.bias"
                router_bias_key = f"{prefix}.mlp.router.score_correction_bias"
                if bias_key in weights and router_bias_key not in weights:
                    weights[router_bias_key] = weights.pop(bias_key)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=BailingMoeLinearConfig, model_type="bailing_moe_linear")
class BailingMoeLinearForCausalLM(BaseCausalLMModule[BailingMoeLinearModel, BailingMoeLinearConfig]):
    """Bailing MoE Linear model with a causal language modeling head.

    Wraps ``BailingMoeLinearModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``BailingMoeLinearConfig``).

    Example::

        >>> config = BailingMoeLinearConfig(vocab_size=102400, hidden_size=2048)
        >>> model = BailingMoeLinearForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = BailingMoeLinearConfig

    def __init__(self, config: BailingMoeLinearConfig):
        """Initialize the Bailing MoE Linear causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=BailingMoeLinearModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("BailingMoeLinearForCausalLM", "BailingMoeLinearModel")
