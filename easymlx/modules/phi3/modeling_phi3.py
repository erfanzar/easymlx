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

"""Phi3 MLX implementation (serving/inference only).

Structure:
  Phi3Config -> Phi3Attention -> Phi3MLP -> Phi3DecoderLayer -> Phi3Model -> Phi3ForCausalLM

Very similar to Llama with:
  - partial_rotary_factor support
  - SwiGLU MLP
  - RMSNorm
  - rope_scaling support
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

from .phi3_configuration import Phi3Config

CacheView = TransformerCacheView | PageCacheView


class Phi3Attention(nn.Module):
    """Multi-head attention layer for the Phi3 model.

    Supports partial RoPE via ``partial_rotary_factor`` and grouped-query
    attention. Very similar to Llama attention but with configurable
    rotation fraction and SuScaledRoPE support.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention logit scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary positional embedding module (partial or full).
        attention_performer: Attention computation backend.

    Example:
        >>> attn = Phi3Attention(config)
        >>> out = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: Phi3Config):
        """Initialize Phi3 attention layer.

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
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope = get_rope(
            dims=rope_dim,
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
        return self.o_proj(attn.reshape(*lead, -1))


class Phi3MLP(nn.Module):
    """SwiGLU feed-forward network for the Phi3 model.

    Uses gate/up/down projections with SiLU activation, identical to
    the Llama MLP architecture.

    Attributes:
        gate_proj: Gate projection linear layer.
        up_proj: Up projection linear layer.
        down_proj: Down projection linear layer.

    Example:
        >>> mlp = Phi3MLP(config)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: Phi3Config):
        """Initialize Phi3 MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU feed-forward transformation.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Phi3DecoderLayer(nn.Module):
    """Single transformer decoder layer for the Phi3 model.

    Uses standard pre-norm with RMSNorm before both attention and MLP,
    with sequential residual connections.

    Attributes:
        self_attn: Multi-head attention module.
        mlp: SwiGLU feed-forward network.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.

    Example:
        >>> layer = Phi3DecoderLayer(config)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: Phi3Config):
        """Initialize Phi3 decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = Phi3Attention(config)
        self.mlp = Phi3MLP(config)
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
        """Run one decoder layer with pre-norm and residual connections.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Phi3Config, model_type="phi3")
class Phi3Model(EasyMLXBaseModule):
    """Base Phi3 transformer model without a language modeling head.

    Architecturally very similar to Llama with optional partial RoPE
    (via ``partial_rotary_factor``) and SuScaledRoPE support via
    ``rope_scaling``.

    Attributes:
        config_class: Associated configuration class (``Phi3Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``Phi3DecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> model = Phi3Model(config)
        >>> hidden = model(input_ids)
    """

    config_class = Phi3Config

    def __init__(self, config: Phi3Config):
        """Initialize the base Phi3 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides
                ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final RMSNorm.

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

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=Phi3Config, model_type="phi3")
class Phi3ForCausalLM(BaseCausalLMModule[Phi3Model, Phi3Config]):
    """Phi3 model with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class (``Phi3Config``).

    Example:
        >>> model = Phi3ForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = Phi3Config

    def __init__(self, config: Phi3Config):
        """Initialize the Phi3 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Phi3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("Phi3ForCausalLM", "Phi3Model")
