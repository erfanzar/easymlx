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

"""Starcoder2 MLX model implementation for serving and inference.

This module provides the Starcoder2 architecture on MLX, featuring
LayerNorm, biased attention and MLP projections, GELU activation,
and RoPE positional embeddings.
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

from .starcoder2_configuration import Starcoder2Config

CacheView = TransformerCacheView | PageCacheView


class Starcoder2Attention(nn.Module):
    """Starcoder2 multi-head attention with RoPE, bias, and GQA support.

    All Q/K/V/O projections include bias terms, distinguishing Starcoder2
    from Llama-style attention modules. Uses RoPE for positional encoding
    and GQA for key/value head sharing.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor (``head_dim ** -0.5``).
        q_proj: Query projection with bias.
        k_proj: Key projection with bias.
        v_proj: Value projection with bias.
        o_proj: Output projection with bias.
        rope: RoPE module for positional encoding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = Starcoder2Config(hidden_size=3072, num_attention_heads=24)
        >>> attn = Starcoder2Attention(config)
    """

    def __init__(self, config: Starcoder2Config):
        """Initialize Starcoder2 attention.

        Args:
            config: Model configuration specifying head counts, dimensions,
                RoPE parameters, and attention mechanism.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: Optional KV cache view for autoregressive decoding.
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
            rope=self.rope,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class Starcoder2MLP(nn.Module):
    """Starcoder2 feed-forward network with GELU activation and bias.

    Unlike the SwiGLU MLP used in Llama-style models, Starcoder2 uses a
    simple two-layer MLP with GELU activation and bias on both projections.

    Attributes:
        c_fc: Linear projection to intermediate size (with bias).
        c_proj: Linear projection back to hidden size (with bias).

    Example:
        >>> config = Starcoder2Config(hidden_size=3072, intermediate_size=12288)
        >>> mlp = Starcoder2MLP(config)
    """

    def __init__(self, config: Starcoder2Config):
        """Initialize the GELU MLP.

        Args:
            config: Model configuration specifying hidden/intermediate sizes.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the GELU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.c_proj(nn.gelu(self.c_fc(hidden_states)))


class Starcoder2DecoderLayer(nn.Module):
    """Single Starcoder2 decoder layer with LayerNorm.

    Applies pre-norm attention followed by pre-norm GELU MLP, each with
    a residual connection. Uses standard LayerNorm instead of RMSNorm.

    Attributes:
        self_attn: Multi-head attention sub-layer.
        mlp: GELU feed-forward sub-layer.
        input_layernorm: LayerNorm before attention.
        post_attention_layernorm: LayerNorm before the MLP.

    Example:
        >>> config = Starcoder2Config(hidden_size=3072)
        >>> layer = Starcoder2DecoderLayer(config)
    """

    def __init__(self, config: Starcoder2Config):
        """Initialize a Starcoder2 decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = Starcoder2Attention(config)
        self.mlp = Starcoder2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

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


@register_module(task_type=TaskType.BASE_MODULE, config=Starcoder2Config, model_type="starcoder2")
class Starcoder2Model(EasyMLXBaseModule):
    """Base Starcoder2 transformer model for code generation.

    A decoder-only transformer with biased attention/MLP, LayerNorm, GELU
    activation, RoPE positional embeddings, and GQA.

    Attributes:
        config_class: Associated configuration class (``Starcoder2Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``Starcoder2DecoderLayer`` modules.
        norm: Final LayerNorm.

    Example:
        >>> config = Starcoder2Config(hidden_size=3072, num_hidden_layers=4)
        >>> model = Starcoder2Model(config)
    """

    config_class = Starcoder2Config

    def __init__(self, config: Starcoder2Config):
        """Initialize the base Starcoder2 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Starcoder2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

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
        """Remove non-persistent rotary embedding buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Starcoder2Config, model_type="starcoder2")
class Starcoder2ForCausalLM(BaseCausalLMModule[Starcoder2Model, Starcoder2Config]):
    """Starcoder2 causal language model with an LM head.

    Wraps ``Starcoder2Model`` with a linear language-model head for
    next-token prediction. Supports weight tying.

    Attributes:
        config_class: Associated configuration class (``Starcoder2Config``).

    Example:
        >>> config = Starcoder2Config(hidden_size=3072, num_hidden_layers=4)
        >>> model = Starcoder2ForCausalLM(config)
    """

    config_class = Starcoder2Config

    def __init__(self, config: Starcoder2Config):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Starcoder2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Starcoder2ForCausalLM", "Starcoder2Model")
