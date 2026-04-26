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

"""GraniteMoeHybrid MLX implementation (serving/inference only).

Structure mirrors upstream GraniteMoeHybrid architecture:
  GraniteMoeHybridConfig -> GraniteMoeHybridAttention -> GraniteMoeHybridMoE
  -> GraniteMoeHybridLayer -> GraniteMoeHybridModel -> GraniteMoeHybridForCausalLM

GraniteMoeHybrid combines Mamba2 (SSM) and Attention layers with optional
MoE, plus custom scaling multipliers for embeddings, attention, residuals,
and logits.

Note: Mamba layers require ArraysCache support. For attention-only configs
(no Mamba layers), this module works with standard TransformerCacheView.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .granitemoehybrid_configuration import GraniteMoeHybridConfig

CacheView = TransformerCacheView | PageCacheView


class GraniteMoeHybridAttention(nn.Module):
    """Multi-head attention for GraniteMoeHybrid with optional RoPE.

    Supports both ``"rope"`` and ``"nope"`` (no position embeddings)
    modes via ``position_embedding_type``. Uses ``attention_multiplier``
    as the attention scale factor.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scale (``attention_multiplier``).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary embedding module, or None if ``"nope"``.
        attention_performer: Attention backend.

    Example::

        >>> attn = GraniteMoeHybridAttention(
        ...     GraniteMoeHybridConfig(hidden_size=64, num_attention_heads=4)
        ... )
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        """Initialize GraniteMoeHybrid attention.

        Args:
            config: Model configuration. The ``position_embedding_type``
                field determines whether RoPE is used.
        """
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = config.attention_multiplier

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=config.attention_bias)

        use_rope = config.position_embedding_type != "nope"
        if use_rope:
            self.rope = get_rope(
                dims=self.head_dim,
                base=config.rope_theta,
                traditional=False,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.rope = None

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
        """Compute attention with optional RoPE.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Output of shape ``[batch, seq_len, hidden_size]``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
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


class GraniteMoeHybridTopKGating(nn.Module):
    """Top-K expert gating for GraniteMoeHybrid.

    Routes each token to top-K experts via softmax-normalized logits.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        layer: Linear gating projection (no bias).

    Args:
        input_size: Input feature dimensionality.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
    """

    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """Initialize top-K gating.

        Args:
            input_size: Input feature dimensionality.
            num_experts: Total experts.
            top_k: Experts per token.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        """Compute top-K expert selection.

        Args:
            hidden_states: Input of shape ``[..., input_size]``.

        Returns:
            Tuple of (expert_indices, gate_weights).
        """
        logits = self.layer(hidden_states)
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeHybridMoE(nn.Module):
    """Mixture-of-Experts block using SwitchGLU for GraniteMoeHybrid.

    Attributes:
        switch_mlp: SwitchGLU expert block.
        router: Top-K gating module.

    Example::

        >>> moe = GraniteMoeHybridMoE(GraniteMoeHybridConfig(
        ...     hidden_size=64, num_local_experts=4, num_experts_per_tok=2,
        ... ))
        >>> out = moe(mx.zeros((1, 8, 64)))
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        """Initialize MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, config.num_local_experts)
        self.router = GraniteMoeHybridTopKGating(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to experts and combine outputs.

        Args:
            x: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Weighted expert output of the same shape.
        """
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class GraniteMoeHybridSharedMLP(nn.Module):
    """Shared MLP added to MoE output in GraniteMoeHybrid layers.

    A fused gate+up SwiGLU projection that runs for every token
    regardless of expert routing, providing a shared representation.

    Attributes:
        input_linear: Fused gate+up projection to
            ``2 * shared_intermediate_size``.
        output_linear: Down-projection back to ``hidden_size``.

    Example::

        >>> mlp = GraniteMoeHybridSharedMLP(GraniteMoeHybridConfig(
        ...     hidden_size=64, shared_intermediate_size=32,
        ... ))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        """Initialize shared MLP.

        Args:
            config: Model configuration with ``shared_intermediate_size``.
        """
        super().__init__()
        self.input_linear = nn.Linear(config.hidden_size, config.shared_intermediate_size * 2, bias=False)
        self.output_linear = nn.Linear(config.shared_intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply shared SwiGLU MLP.

        Args:
            x: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(nn.silu(gate) * up)


class GraniteMoeHybridDenseMLP(nn.Module):
    """Dense SwiGLU MLP for GraniteMoeHybrid non-MoE mode.

    Used when ``num_local_experts`` is not set.

    Attributes:
        gate_proj: Gating projection.
        down_proj: Output projection.
        up_proj: Value projection.

    Example::

        >>> mlp = GraniteMoeHybridDenseMLP(GraniteMoeHybridConfig(
        ...     hidden_size=64, intermediate_size=128,
        ... ))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        """Initialize dense SwiGLU MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply dense SwiGLU MLP.

        Args:
            x: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GraniteMoeHybridLayer(nn.Module):
    """Single GraniteMoeHybrid layer (attention-only variant).

    Supports attention layers with either MoE (SwitchGLU experts +
    shared MLP) or dense SwiGLU MLP, plus ``residual_multiplier``
    scaling. Mamba layers are not supported in this implementation
    as they require SSM state management.

    Attributes:
        layer_type: ``"attention"`` (only supported type).
        residual_multiplier: Scale for sub-layer outputs.
        input_layernorm: RMSNorm before attention.
        self_attn: Attention module (attention layers only).
        shared_mlp: Shared MLP (MoE mode only).
        block_sparse_moe: MoE block (MoE mode only).
        mlp: Dense MLP (non-MoE mode only).
        post_attention_layernorm: RMSNorm before MLP/MoE.

    Example::

        >>> layer = GraniteMoeHybridLayer(
        ...     GraniteMoeHybridConfig(hidden_size=64), "attention"
        ... )
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeHybridConfig, layer_type: str):
        """Initialize GraniteMoeHybrid layer.

        Args:
            config: Model configuration.
            layer_type: Layer type string. Only ``"attention"`` is
                supported.

        Raises:
            ValueError: If ``layer_type`` is not ``"attention"``.
        """
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = config.residual_multiplier

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_type == "attention":
            self.self_attn = GraniteMoeHybridAttention(config)
        else:
            raise ValueError(
                f"Layer type '{layer_type}' is not supported in this implementation. "
                f"Only 'attention' layers are supported."
            )

        if config.use_moe:
            self.shared_mlp = GraniteMoeHybridSharedMLP(config)
            self.block_sparse_moe = GraniteMoeHybridMoE(config)
        else:
            self.mlp = GraniteMoeHybridDenseMLP(config)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._use_moe = config.use_moe

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with attention and MoE/dense MLP.

        For MoE mode, the MLP output is the sum of MoE expert
        output and shared MLP output.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after attention and MLP/MoE.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        if self._use_moe:
            moe_out = self.block_sparse_moe(normed)
            shared_out = self.shared_mlp(normed)
            mlp_out = moe_out + shared_out
        else:
            mlp_out = self.mlp(normed)

        hidden_states = residual + mlp_out * self.residual_multiplier
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=GraniteMoeHybridConfig, model_type="granitemoehybrid")
class GraniteMoeHybridModel(EasyMLXBaseModule):
    """Base GraniteMoeHybrid transformer with MoE and scaling multipliers.

    Supports attention layers with either MoE routing or dense MLP.
    Token embeddings are scaled by ``embedding_multiplier``. For full
    hybrid (Mamba+Attention) support, Mamba layers would require
    ``ArraysCache`` for SSM state management.

    Attributes:
        embed_tokens: Token embedding layer.
        embedding_multiplier: Scale factor for embeddings.
        layers: Stack of ``GraniteMoeHybridLayer`` instances.
        norm: Final RMSNorm.

    Example::

        >>> model = GraniteMoeHybridModel(GraniteMoeHybridConfig(
        ...     vocab_size=256, hidden_size=64,
        ... ))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GraniteMoeHybridConfig

    def __init__(self, config: GraniteMoeHybridConfig):
        """Initialize GraniteMoeHybrid base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_multiplier = config.embedding_multiplier

        self.layers = [GraniteMoeHybridLayer(config, layer_type) for layer_type in config.layer_types]
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
        """Forward pass with embedding scaling and hybrid layers.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata.

        Returns:
            Normalized hidden states.

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
            hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier

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
        """Handle MoE, conv, and dense MLP weight transformations.

        Performs several transformations:

        1. Transposes Mamba conv1d weights to MLX layout.
        2. For MoE mode: splits fused ``input_linear`` into
           ``gate_proj`` / ``up_proj`` for SwitchGLU format.
        3. For dense mode: splits ``shared_mlp.input_linear`` into
           separate ``gate_proj`` / ``up_proj`` for the dense MLP.
        4. Removes tied ``lm_head`` and ``rotary_emb.inv_freq`` keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """

        for k in list(weights.keys()):
            if "conv1d.weight" in k and weights[k].shape[-1] != 1:
                weights[k] = weights[k].moveaxis(2, 1)

        if self.config.use_moe and "model.layers.0.block_sparse_moe.input_linear.weight" in weights:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}.block_sparse_moe"
                input_key = f"{prefix}.input_linear.weight"
                if input_key in weights:
                    input_weight = weights.pop(input_key)
                    _, expert_hidden, _ = input_weight.shape
                    gate_proj = input_weight[:, : expert_hidden // 2, :]
                    up_proj = input_weight[:, expert_hidden // 2 :, :]
                    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
                    weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_proj
                    out_key = f"{prefix}.output_linear.weight"
                    if out_key in weights:
                        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(out_key)

        elif not self.config.use_moe and "model.layers.0.shared_mlp.input_linear.weight" in weights:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}.shared_mlp"
                input_key = f"{prefix}.input_linear.weight"
                if input_key in weights:
                    input_weight = weights.pop(input_key)
                    gate_proj, up_proj = mx.split(input_weight, 2, axis=0)
                    weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
                    weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
                    out_key = f"{prefix}.output_linear.weight"
                    if out_key in weights:
                        weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = weights.pop(out_key)

        if getattr(self.config, "tie_word_embeddings", True):
            weights.pop("lm_head.weight", None)

        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        return weights


@register_module(task_type=TaskType.CAUSAL_LM, config=GraniteMoeHybridConfig, model_type="granitemoehybrid")
class GraniteMoeHybridForCausalLM(BaseCausalLMModule[GraniteMoeHybridModel, GraniteMoeHybridConfig]):
    """GraniteMoeHybrid causal language model with logits_scaling.

    Wraps ``GraniteMoeHybridModel`` and divides output logits by
    ``logits_scaling``. Supports MoE and dense MLP configurations.

    Attributes:
        config_class: ``GraniteMoeHybridConfig``.

    Example::

        >>> model = GraniteMoeHybridForCausalLM(
        ...     GraniteMoeHybridConfig(vocab_size=256, hidden_size=64)
        ... )
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GraniteMoeHybridConfig

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(
            config=config,
            base_model_class=GraniteMoeHybridModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project to logits and divide by ``logits_scaling``.

        Args:
            hidden_states: Transformer output.

        Returns:
            Scaled logits of shape ``[batch, seq_len, vocab_size]``.
        """
        logits = super().compute_lm_logits(hidden_states)
        return logits / self.config.logits_scaling


__all__ = ("GraniteMoeHybridForCausalLM", "GraniteMoeHybridModel")
