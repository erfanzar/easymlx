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

"""Gemma MLX implementation (serving/inference only).

Structure:
  GemmaConfig -> GemmaAttention -> GemmaMLP -> GemmaDecoderLayer -> GemmaModel -> GemmaForCausalLM

Key differences from Llama:
  - Embeddings scaled by sqrt(hidden_size)
  - GELU activation instead of SiLU
  - Tied word embeddings by default
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

from .gemma_configuration import GemmaConfig

CacheView = TransformerCacheView | PageCacheView


class GemmaAttention(nn.Module):
    """Multi-head attention with grouped-query support for the Gemma model.

    Uses standard ``1/sqrt(head_dim)`` scaling and rotary position
    embeddings (RoPE). Supports grouped-query attention (GQA) when
    ``num_key_value_heads < num_attention_heads``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimensionality per attention head.
        scale: Attention logit scaling factor (``head_dim ** -0.5``).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> attn = GemmaAttention(GemmaConfig(hidden_size=64, num_attention_heads=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GemmaConfig):
        """Initialize Gemma attention layer.

        Args:
            config: Model configuration containing attention parameters.

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
        """Compute multi-head attention with RoPE.

        Args:
            hidden_states: Input tensor of shape ``[batch, seq_len, hidden_size]``
                or ``[seq_len, hidden_size]`` for paged attention.
            mask: Attention mask. Can be a float mask array, a string
                identifier, or None for no masking.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Output tensor with the same leading shape as ``hidden_states``
            and last dimension ``hidden_size``.
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


class GemmaMLP(nn.Module):
    """Gated feed-forward network with GELU activation for the Gemma model.

    Unlike Llama which uses SiLU, Gemma uses GELU as the gating
    activation: ``down_proj(gelu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear projection for the gating branch.
        up_proj: Linear projection for the value branch.
        down_proj: Linear projection back to hidden size.

    Example::

        >>> mlp = GemmaMLP(GemmaConfig(hidden_size=64, intermediate_size=128))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GemmaConfig):
        """Initialize Gemma MLP.

        Args:
            config: Model configuration with ``hidden_size``,
                ``intermediate_size``, and ``mlp_bias``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply gated MLP with GELU activation.

        Args:
            hidden_states: Input tensor of shape
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Output tensor of the same shape as the input.
        """
        return self.down_proj(nn.gelu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class GemmaDecoderLayer(nn.Module):
    """Single transformer decoder layer for the Gemma model.

    Applies pre-norm (RMSNorm) before attention and MLP with residual
    connections: ``h = h + attn(norm1(h)); h = h + mlp(norm2(h))``.

    Attributes:
        self_attn: Multi-head attention sub-layer.
        mlp: Gated GELU feed-forward sub-layer.
        input_layernorm: RMSNorm applied before attention.
        post_attention_layernorm: RMSNorm applied before MLP.

    Example::

        >>> layer = GemmaDecoderLayer(GemmaConfig(hidden_size=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GemmaConfig):
        """Initialize Gemma decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
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
        """Forward pass through one decoder layer.

        Args:
            hidden_states: Input tensor of shape
                ``[batch, seq_len, hidden_size]``.
            mask: Attention mask (float array, string, or None).
            cache_view: Optional KV cache for autoregressive decoding.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Hidden states after attention and MLP with residuals.
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


@register_module(task_type=TaskType.BASE_MODULE, config=GemmaConfig, model_type="gemma")
class GemmaModel(EasyMLXBaseModule):
    """Base Gemma transformer with scaled embeddings and GELU MLP.

    Implements the Gemma architecture which differs from Llama by
    scaling input embeddings by ``sqrt(hidden_size)`` and using
    GELU activation instead of SiLU in the feed-forward network.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: Stack of ``GemmaDecoderLayer`` instances.
        norm: Final RMSNorm applied to transformer output.

    Example::

        >>> model = GemmaModel(GemmaConfig(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GemmaConfig

    def __init__(self, config: GemmaConfig):
        """Initialize Gemma base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the full Gemma transformer.

        Embeds tokens, scales by ``sqrt(hidden_size)``, passes through
        all decoder layers, and applies final normalization.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]`` or
                ``[seq_len]``.
            attention_mask: Optional attention mask of shape
                ``[batch, seq_len]``.
            input_embeddings: Pre-computed embeddings. If provided,
                ``input_ids`` is ignored for embedding lookup.
            cache_views: Per-layer KV cache views for autoregressive
                decoding. Must match the number of layers.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Normalized hidden states of shape
            ``[batch, seq_len, hidden_size]``.

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

        hidden_states = hidden_states * (self.config.hidden_size**0.5)

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


@register_module(task_type=TaskType.CAUSAL_LM, config=GemmaConfig, model_type="gemma")
class GemmaForCausalLM(BaseCausalLMModule[GemmaModel, GemmaConfig]):
    """Gemma model with a causal language modeling head.

    Wraps ``GemmaModel`` and adds a linear LM head that projects hidden
    states to vocabulary logits. Supports tied word embeddings by default.

    Attributes:
        config_class: ``GemmaConfig``.
        model: The underlying ``GemmaModel`` base model.

    Example::

        >>> model = GemmaForCausalLM(GemmaConfig(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GemmaConfig

    def __init__(self, config: GemmaConfig):
        """Initialize Gemma causal LM.

        Args:
            config: Model configuration. Uses ``tie_word_embeddings``
                (default True) to share input/output embedding weights.
        """
        super().__init__(
            config=config,
            base_model_class=GemmaModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("GemmaForCausalLM", "GemmaModel")
