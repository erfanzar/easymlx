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

"""DeepSeek MLX model implementation for serving and inference.

This module provides the DeepSeek architecture on MLX, featuring standard
GQA attention with RoPE, Mixture-of-Experts with shared experts, and a
causal language model wrapper.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .deepseek_configuration import DeepseekConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class DeepseekAttention(nn.Module):
    """Standard grouped-query attention for DeepSeek with RoPE.

    Uses separate Q, K, V projections with rotary position embeddings.
    Supports optional linear RoPE scaling.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query linear projection.
        k_proj: Key linear projection.
        v_proj: Value linear projection.
        o_proj: Output linear projection.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = DeepseekConfig(hidden_size=2048, num_attention_heads=16)
        >>> attn = DeepseekAttention(config)
    """

    def __init__(self, config: DeepseekConfig):
        """Initialize DeepSeek attention.

        Args:
            config: Model configuration with attention hyperparameters.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        if config.rope_scaling and config.rope_scaling.get("type") == "linear":
            factor = config.rope_scaling.get("factor", 1.0)
            1.0 / float(factor)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Run the GQA attention forward pass with RoPE.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class DeepseekMLP(nn.Module):
    """SiLU-gated feed-forward MLP for DeepSeek.

    Uses ``down_proj(silu(gate_proj(x)) * up_proj(x))``. Used both
    as the dense MLP and as the shared expert MLP.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: DeepseekConfig, intermediate_size: int | None = None):
        """Initialize the DeepSeek MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimensionality.
                If ``None``, uses ``config.intermediate_size``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DeepseekMoE(nn.Module):
    """Mixture-of-Experts block for DeepSeek with shared experts.

    Routes tokens to top-k experts via ``TopKRouter`` with softmax
    scoring. Optional shared experts are added to the routed output.
    Uses fused ``SwitchGLU`` for efficient expert computation.

    Attributes:
        config: Model configuration.
        router: Top-k expert routing module with softmax scoring.
        switch_mlp: Fused SwitchGLU expert module.
        shared_experts: Optional shared expert MLP (always active).

    Example::

        >>> config = DeepseekConfig(hidden_size=2048, n_routed_experts=64)
        >>> moe = DeepseekMoE(config)
    """

    def __init__(self, config: DeepseekConfig):
        """Initialize the DeepSeek MoE block.

        Args:
            config: Model configuration with MoE hyperparameters.
        """
        super().__init__()
        self.config = config
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
        )
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)

        if config.n_shared_experts is not None:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of the same shape.
        """
        inds, scores = self.router(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class DeepseekDecoderLayer(nn.Module):
    """Single DeepSeek decoder layer with optional MoE.

    Uses pre-norm attention and pre-norm MLP/MoE with residual
    connections. Layers meeting the MoE criteria (after
    ``first_k_dense_replace`` and at ``moe_layer_freq`` intervals)
    use the MoE block; others use dense MLP.

    Attributes:
        self_attn: GQA attention module.
        mlp: Dense MLP or MoE block.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
    """

    def __init__(self, config: DeepseekConfig, layer_idx: int):
        """Initialize the DeepSeek decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack. Determines
                whether to use MoE or dense MLP.
        """
        super().__init__()
        self.self_attn = DeepseekAttention(config)
        self.mlp = (
            DeepseekMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekMLP(config)
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
        """Run the decoder layer forward pass.

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


@register_module(task_type=TaskType.BASE_MODULE, config=DeepseekConfig, model_type="deepseek")
class DeepseekModel(EasyMLXBaseModule):
    """Base DeepSeek transformer model for inference.

    Implements a decoder-only transformer with GQA attention, RoPE,
    and optional Mixture-of-Experts feed-forward with shared experts.

    Attributes:
        config_class: The configuration class (``DeepseekConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``DeepseekDecoderLayer`` decoder blocks.
        norm: Final RMS normalization.

    Example::

        >>> config = DeepseekConfig(vocab_size=102400, hidden_size=2048)
        >>> model = DeepseekModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekConfig

    def __init__(self, config: DeepseekConfig):
        """Initialize the DeepSeek base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
        """Transform checkpoint weights for compatibility.

        Stacks per-expert weights into fused SwitchGLU format, remaps
        ``gate`` to ``router``, and removes rotary embedding buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with expert weights stacked and
            keys remapped.
        """
        n_routed = getattr(self.config, "n_routed_experts", None)
        if n_routed is not None:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(n_routed)]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
                # Remap router gate weight
                gate_key = f"{prefix}.mlp.gate.weight"
                router_key = f"{prefix}.mlp.router.weight"
                if gate_key in weights and router_key not in weights:
                    weights[router_key] = weights.pop(gate_key)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=DeepseekConfig, model_type="deepseek")
class DeepseekForCausalLM(BaseCausalLMModule[DeepseekModel, DeepseekConfig]):
    """DeepSeek model with a causal language modeling head.

    Wraps ``DeepseekModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``DeepseekConfig``).

    Example::

        >>> config = DeepseekConfig(vocab_size=102400, hidden_size=2048)
        >>> model = DeepseekForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekConfig

    def __init__(self, config: DeepseekConfig):
        """Initialize the DeepSeek causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("DeepseekForCausalLM", "DeepseekModel")
