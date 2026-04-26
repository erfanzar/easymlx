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

"""Seed OSS MLX model implementation for serving and inference.

Seed OSS is a standard dense transformer similar to Llama with GQA,
RoPE, and SwiGLU MLP.  It supports optional attention and MLP biases
and optional attention output bias.
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

from .seed_oss_configuration import SeedOssConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an ``mx.array`` of dtype ``int32``.

    Args:
        values: Input values to convert. Accepts ``mx.array``, Python
            sequences, or ``None``.

    Returns:
        An ``mx.array`` with ``int32`` dtype, or ``None`` if the input
        is ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class SeedOssAttention(nn.Module):
    """Multi-head attention module for the Seed OSS architecture.

    Implements Grouped Query Attention (GQA) with Rotary Positional
    Embeddings (RoPE). Supports optional bias on Q/K/V projections
    (controlled by ``attention_bias``) and on the output projection
    (controlled by ``attention_out_bias``).

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor applied to attention scores (``head_dim ** -0.5``).
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
        o_proj: Linear output projection.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example:
        >>> config = SeedOssConfig(hidden_size=2048, num_attention_heads=16)
        >>> attn = SeedOssAttention(config)
        >>> out = attn(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: SeedOssConfig):
        """Initialize Seed OSS attention.

        Args:
            config: Model configuration specifying head counts, dimensions,
                RoPE parameters, and bias settings.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        input_bias = config.attention_bias
        output_bias = config.attention_out_bias

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=input_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=input_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=input_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=output_bias)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(scale=self.scale)

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
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``
                or ``(total_tokens, hidden_size)`` for paged mode.
            mask: Attention mask. Can be a boolean ``mx.array``, the string
                ``"causal"``, or ``None`` for no masking.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata when using ``PageCacheView``.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``
            with trailing dimension ``hidden_size``.
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


class SeedOssMLP(nn.Module):
    """SwiGLU feed-forward network for the Seed OSS architecture.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``. Supports
    optional bias on all three projections.

    Attributes:
        gate_proj: Linear projection for the SiLU gate.
        up_proj: Linear projection for the element-wise product branch.
        down_proj: Linear projection back to ``hidden_size``.

    Example:
        >>> config = SeedOssConfig(hidden_size=2048, intermediate_size=5504)
        >>> mlp = SeedOssMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: SeedOssConfig):
        """Initialize the SwiGLU MLP.

        Args:
            config: Model configuration specifying hidden/intermediate sizes
                and whether MLP projections include bias.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SeedOssDecoderLayer(nn.Module):
    """Single Seed OSS transformer decoder layer.

    Applies pre-norm attention followed by pre-norm SwiGLU MLP, each
    with a residual connection: ``x = x + attn(norm(x))`` then
    ``x = x + mlp(norm(x))``.

    Attributes:
        self_attn: Multi-head attention sub-layer.
        mlp: SwiGLU feed-forward sub-layer.
        input_layernorm: RMSNorm applied before attention.
        post_attention_layernorm: RMSNorm applied before the MLP.

    Example:
        >>> config = SeedOssConfig(hidden_size=2048)
        >>> layer = SeedOssDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: SeedOssConfig):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = SeedOssAttention(config)
        self.mlp = SeedOssMLP(config)
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
            mask: Attention mask (boolean array, ``"causal"`` string, or ``None``).
            cache_view: Optional KV cache view for autoregressive decoding.
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


@register_module(task_type=TaskType.BASE_MODULE, config=SeedOssConfig, model_type="seed_oss")
class SeedOssModel(EasyMLXBaseModule):
    """Base Seed OSS transformer model.

    A standard dense Llama-style decoder-only transformer with token
    embeddings, a stack of ``SeedOssDecoderLayer`` blocks, and a final
    RMSNorm. Supports both batched 3-D input and flat 2-D paged-attention
    input.

    Attributes:
        config_class: Associated configuration class (``SeedOssConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``SeedOssDecoderLayer`` modules.
        norm: Final RMSNorm applied to the last hidden state.

    Example:
        >>> config = SeedOssConfig(hidden_size=2048, num_hidden_layers=4)
        >>> model = SeedOssModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
        >>> hidden.shape
        [1, 3, 2048]
    """

    config_class = SeedOssConfig

    def __init__(self, config: SeedOssConfig):
        """Initialize the base Seed OSS model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [SeedOssDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
                Ignored when ``input_embeddings`` is provided.
            attention_mask: Optional attention mask broadcastable to
                ``(batch, 1, seq_len, seq_len)``.
            input_embeddings: Pre-computed embeddings of shape
                ``(batch, seq_len, hidden_size)``. When provided, ``input_ids``
                is ignored.
            cache_views: Per-layer KV cache views for autoregressive decoding.
                Must have the same length as ``self.layers`` if provided.
            cache_metadata: Paged-attention metadata for ``PageCacheView``.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` after
            the final RMSNorm.

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
        """Remove non-persistent rotary embedding buffers from checkpoint weights.

        Filters out ``rotary_emb.inv_freq`` and ``rope.inv_freq`` keys that
        are stored in some upstream checkpoints but are recomputed at
        initialization.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=SeedOssConfig, model_type="seed_oss")
class SeedOssForCausalLM(BaseCausalLMModule[SeedOssModel, SeedOssConfig]):
    """Seed OSS causal language model with an LM head.

    Wraps ``SeedOssModel`` with a linear language-model head for next-token
    prediction. Supports optional weight tying between the embedding layer
    and the LM head.

    Attributes:
        config_class: Associated configuration class (``SeedOssConfig``).

    Example:
        >>> config = SeedOssConfig(hidden_size=2048, num_hidden_layers=4)
        >>> model = SeedOssForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = SeedOssConfig

    def __init__(self, config: SeedOssConfig):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration. ``tie_word_embeddings`` controls
                whether the LM head shares weights with the embedding layer.
        """
        super().__init__(
            config=config,
            base_model_class=SeedOssModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("SeedOssForCausalLM", "SeedOssModel")
