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

"""Granite MLX implementation (serving/inference only).

Structure mirrors EasyDeL's granite:
  GraniteConfig -> GraniteAttention -> GraniteMLP -> GraniteDecoderLayer
  -> GraniteModel -> GraniteForCausalLM

Granite extends the Llama architecture with multiplier scaling for
embeddings, attention, residuals, and logits.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .granite_configuration import GraniteConfig

CacheView = TransformerCacheView | PageCacheView


class GraniteAttention(nn.Module):
    """Multi-head attention with Granite's attention_multiplier scaling.

    The attention scale is computed as
    ``attention_multiplier / sqrt(head_dim)`` instead of the standard
    ``1 / sqrt(head_dim)``, allowing the model to learn different
    effective temperatures.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scale incorporating ``attention_multiplier``.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding.
        attention_performer: Attention backend.

    Example::

        >>> attn = GraniteAttention(GraniteConfig(hidden_size=64, num_attention_heads=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteConfig):
        """Initialize Granite attention.

        Args:
            config: Model configuration with ``attention_multiplier``.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        # Granite uses attention_multiplier instead of just 1/sqrt(head_dim)
        self.scale = config.attention_multiplier / (self.head_dim**0.5)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.rope = get_rope(
            dims=self.head_dim,
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
        """Compute attention with multiplier-scaled softmax.

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


class GraniteMLP(nn.Module):
    """SwiGLU (SiLU-gated) feed-forward network for Granite.

    Attributes:
        gate_proj: Gating projection.
        up_proj: Value projection.
        down_proj: Output projection.

    Example::

        >>> mlp = GraniteMLP(GraniteConfig(hidden_size=64, intermediate_size=128))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteConfig):
        """Initialize Granite SwiGLU MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class GraniteDecoderLayer(nn.Module):
    """Single Granite decoder layer with residual_multiplier scaling.

    Both attention and MLP outputs are scaled by ``residual_multiplier``
    before adding to the residual stream:
    ``h = h + attn(norm1(h)) * multiplier``.

    Attributes:
        self_attn: Granite attention with multiplier scaling.
        mlp: SwiGLU MLP.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
        residual_multiplier: Scale factor for sub-layer outputs.

    Example::

        >>> layer = GraniteDecoderLayer(GraniteConfig(hidden_size=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteConfig):
        """Initialize Granite decoder layer.

        Args:
            config: Model configuration with ``residual_multiplier``.
        """
        super().__init__()
        self.self_attn = GraniteAttention(config)
        self.mlp = GraniteMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with residual_multiplier scaling.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after scaled attention and MLP.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + attn_out * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states) * self.residual_multiplier
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=GraniteConfig, model_type="granite")
class GraniteModel(EasyMLXBaseModule):
    """Base Granite transformer with embedding_multiplier scaling.

    Token embeddings are scaled by ``embedding_multiplier`` after
    lookup. Each decoder layer applies ``residual_multiplier`` to
    sub-layer outputs, and the attention uses ``attention_multiplier``
    for the scaling factor.

    Attributes:
        embed_tokens: Token embedding layer.
        embedding_multiplier: Scale factor for embeddings.
        layers: Stack of ``GraniteDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example::

        >>> model = GraniteModel(GraniteConfig(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GraniteConfig

    def __init__(self, config: GraniteConfig):
        """Initialize Granite base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_multiplier = config.embedding_multiplier
        self.layers = [GraniteDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass with embedding_multiplier scaling.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=GraniteConfig, model_type="granite")
class GraniteForCausalLM(BaseCausalLMModule[GraniteModel, GraniteConfig]):
    """Granite causal language model with logits_scaling.

    Wraps ``GraniteModel`` and divides output logits by
    ``logits_scaling`` to control the effective temperature of the
    language model output distribution.

    Attributes:
        config_class: ``GraniteConfig``.

    Example::

        >>> model = GraniteForCausalLM(GraniteConfig(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GraniteConfig

    def __init__(self, config: GraniteConfig):
        super().__init__(
            config=config,
            base_model_class=GraniteModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project to logits and divide by ``logits_scaling``.

        Args:
            hidden_states: Transformer output of shape
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Scaled logits of shape ``[batch, seq_len, vocab_size]``.
        """
        logits = super().compute_lm_logits(hidden_states)
        return logits / self.config.logits_scaling


__all__ = ("GraniteForCausalLM", "GraniteModel")
