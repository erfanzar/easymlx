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

"""Qwen3 MoE MLX model implementation for serving and inference.

This module provides the Qwen3-MoE architecture on MLX, featuring QK
normalization, sparse expert routing, optional sliding-window attention,
and a causal language model wrapper.
"""

from __future__ import annotations

import typing as tp

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

from .qwen3_moe_configuration import Qwen3MoeConfig

CacheView = TransformerCacheView | PageCacheView


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Look up an activation function by name.

    Args:
        name: Case-insensitive activation name. Supported values are
            ``"silu"``, ``"swish"``, and ``"gelu"``.

    Returns:
        The corresponding MLX activation callable.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


class Qwen3MoeAttention(nn.Module):
    """Multi-head attention with QK normalization for Qwen3-MoE.

    Features grouped-query attention, mandatory RMS normalization on query
    and key projections, and configurable attention bias.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: RMS normalization for queries.
        k_norm: RMS normalization for keys.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.
    """

    def __init__(self, config: Qwen3MoeConfig):
        """Initialize the Qwen3-MoE attention module.

        Args:
            config: Qwen3-MoE configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        """Compute grouped-query attention with QK normalization.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
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


class Qwen3MoeMLP(nn.Module):
    """SiLU-gated feed-forward MLP for Qwen3-MoE.

    Used as the dense MLP in non-MoE layers.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
        act_fn: Activation function.
    """

    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Qwen3-MoE configuration.
            intermediate_size: Override for the intermediate dimensionality.
                Defaults to ``config.intermediate_size`` when ``None``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3MoeSparseBlock(nn.Module):
    """Mixture-of-Experts block for Qwen3-MoE.

    Routes tokens to top-k experts and aggregates weighted outputs.

    Attributes:
        router: Top-k expert routing module.
        experts: SwitchGLU expert bank.
    """

    def __init__(self, config: Qwen3MoeConfig):
        """Initialize the MoE sparse block.

        Args:
            config: Qwen3-MoE configuration.
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
            norm_topk_prob=config.norm_topk_prob,
        )
        self.experts = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and aggregate outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Weighted sum of expert outputs, same shape as input.
        """
        inds, scores = self.router(hidden_states)
        out = self.experts(hidden_states, inds)
        return (out * scores[..., None]).sum(axis=-2).astype(out.dtype)


class Qwen3MoeDecoderLayer(nn.Module):
    """Single decoder layer for Qwen3-MoE.

    Alternates between dense MLP and MoE sparse block based on layer index.

    Attributes:
        use_sliding: Whether this layer uses sliding-window attention.
        self_attn: Multi-head attention with QK normalization.
        mlp: Dense MLP or MoE sparse block.
        input_layernorm: Pre-attention RMS normalization.
        post_attention_layernorm: Post-attention RMS normalization.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, *, use_sliding: bool = False):
        """Initialize the decoder layer.

        Args:
            config: Qwen3-MoE configuration.
            layer_idx: Zero-based index of this layer.
            use_sliding: Whether this layer uses sliding-window attention.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Qwen3MoeAttention(config)
        use_moe = (layer_idx % max(config.decoder_sparse_step, 1) == 0) and layer_idx not in config.mlp_only_layers
        self.mlp = Qwen3MoeSparseBlock(config) if use_moe else Qwen3MoeMLP(config)
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
        """Run one decoder layer.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same shape as the input.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3MoeConfig, model_type="qwen3_moe")
class Qwen3MoeModel(EasyMLXBaseModule):
    """Base Qwen3-MoE transformer model.

    Attributes:
        config_class: The associated configuration class (``Qwen3MoeConfig``).
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
        sliding_window: Sliding window size for applicable layers.
    """

    config_class = Qwen3MoeConfig

    def __init__(self, config: Qwen3MoeConfig):
        """Initialize the base Qwen3-MoE model.

        Args:
            config: Qwen3-MoE configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(config, idx, use_sliding=layer_type == "sliding_attention")
            for idx, layer_type in enumerate(config.layer_types)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Qwen3-MoE transformer stack.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states from the transformer stack.

        Raises:
            ValueError: If ``cache_views`` length does not match layer count.
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
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.use_sliding for layer in self.layers):
                    sliding_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.sliding_window,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.use_sliding:
                layer_metadata = cache_metadata.with_sliding_window(self.sliding_window)
            layer_mask = sliding_mask if layer.use_sliding else mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3MoeConfig, model_type="qwen3_moe")
class Qwen3MoeForCausalLM(BaseCausalLMModule[Qwen3MoeModel, Qwen3MoeConfig]):
    """Qwen3-MoE model with a causal language modeling head.

    Wraps ``Qwen3MoeModel`` and adds vocabulary projection to produce
    next-token logits.

    Attributes:
        config_class: The associated configuration class (``Qwen3MoeConfig``).
    """

    config_class = Qwen3MoeConfig

    def __init__(self, config: Qwen3MoeConfig):
        """Initialize the causal language model.

        Args:
            config: Qwen3-MoE configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3MoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("Qwen3MoeForCausalLM", "Qwen3MoeModel")
