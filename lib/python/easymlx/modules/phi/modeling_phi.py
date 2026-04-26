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

"""Phi MLX implementation (serving/inference only).

Structure:
  PhiConfig -> PhiAttention -> PhiMLP -> PhiDecoderLayer -> PhiModel -> PhiForCausalLM

Key differences from Llama:
  - LayerNorm instead of RMSNorm
  - Parallel residuals: h = x + attn(ln(x)) + mlp(ln(x))
  - Partial RoPE: only applied to first (partial_rotary_factor * head_dim) dims
  - GELU approx activation
  - Bias in attention and MLP projections
  - No gate_proj in MLP (simple fc1 -> gelu -> fc2)
"""

from __future__ import annotations

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
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .phi_configuration import PhiConfig

CacheView = TransformerCacheView | PageCacheView


class PhiAttention(nn.Module):
    """Multi-head attention layer for the Phi model.

    Uses partial RoPE (only applied to a fraction of head dimensions
    controlled by ``partial_rotary_factor``) and includes bias in all
    projections. Supports grouped-query attention when
    ``num_key_value_heads < num_attention_heads``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality (``hidden_size // num_heads``).
        scale: Attention logit scaling factor.
        q_proj: Query projection with bias.
        k_proj: Key projection with bias.
        v_proj: Value projection with bias.
        dense: Output projection with bias.
        rope: Partial rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example:
        >>> attn = PhiAttention(config)
        >>> out = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: PhiConfig):
        """Initialize Phi attention layer.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

        rope_dim = int(config.partial_rotary_factor * self.head_dim)
        self.rope = get_rope(
            dims=rope_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Compute multi-head attention with partial RoPE.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
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
        return self.dense(attn.reshape(*lead, -1))


class PhiMLP(nn.Module):
    """Feed-forward network for the Phi model.

    Uses a simple two-layer MLP with GELU approximate activation and bias
    in both layers. Unlike Llama-style models, there is no gating projection.

    Attributes:
        fc1: First linear layer (hidden_size -> intermediate_size) with bias.
        fc2: Second linear layer (intermediate_size -> hidden_size) with bias.

    Example:
        >>> mlp = PhiMLP(config)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: PhiConfig):
        """Initialize Phi MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply fc1 -> GELU approx -> fc2.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        return self.fc2(nn.gelu_approx(self.fc1(hidden_states)))


class PhiDecoderLayer(nn.Module):
    """Single transformer decoder layer for the Phi model.

    Uses parallel residuals: ``output = x + attn(ln(x)) + mlp(ln(x))``,
    where the same LayerNorm output is fed to both attention and MLP
    branches simultaneously.

    Attributes:
        self_attn: Multi-head attention module.
        mlp: Feed-forward network.
        input_layernorm: Shared LayerNorm applied before both branches.

    Example:
        >>> layer = PhiDecoderLayer(config)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: PhiConfig):
        """Initialize Phi decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = PhiAttention(config)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer with parallel attention + MLP residuals.

        Args:
            hidden_states: Input tensor of shape
                ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        ff_out = self.mlp(normed)
        return hidden_states + attn_out + ff_out


@register_module(task_type=TaskType.BASE_MODULE, config=PhiConfig, model_type="phi")
class PhiModel(EasyMLXBaseModule):
    """Base Phi transformer model without a language modeling head.

    Phi uses parallel residuals (attention and MLP share the same
    normalized input), partial RoPE, LayerNorm, GELU approximate
    activation, and bias in all projections.

    Attributes:
        config_class: Associated configuration class (``PhiConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``PhiDecoderLayer`` instances.
        final_layernorm: Final LayerNorm applied after all layers.

    Example:
        >>> model = PhiModel(config)
        >>> hidden = model(input_ids)
    """

    config_class = PhiConfig

    def __init__(self, config: PhiConfig):
        """Initialize the base Phi model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def norm(self):
        """Alias for final_layernorm to match the sanitize pattern."""
        return self.final_layernorm

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
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides
                ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final LayerNorm.

        Raises:
            ValueError: If ``cache_views`` length does not match the
                number of layers.
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

        return self.final_layernorm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=PhiConfig, model_type="phi")
class PhiForCausalLM(BaseCausalLMModule[PhiModel, PhiConfig]):
    """Phi model with a causal language modeling head.

    Wraps ``PhiModel`` and adds a biased LM head for next-token prediction.

    Attributes:
        config_class: Associated configuration class (``PhiConfig``).

    Example:
        >>> model = PhiForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = PhiConfig

    def __init__(self, config: PhiConfig):
        """Initialize the Phi causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=PhiModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            lm_head_bias=True,
        )


__all__ = ("PhiForCausalLM", "PhiModel")
