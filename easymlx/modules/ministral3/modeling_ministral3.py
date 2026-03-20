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

"""Ministral3 MLX model implementation for serving and inference."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.llama.modeling_llama import LlamaMLP

from .ministral3_configuration import Ministral3Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_list(values: mx.ArrayLike | None) -> list[int]:
    """Convert an array-like to a Python list of ints.

    Args:
        values: Input values. Accepts ``mx.array``, sequences, or ``None``.

    Returns:
        A list of ``int``, or an empty list if input is ``None``.
    """
    if values is None:
        return []
    if isinstance(values, mx.array):
        return [int(value) for value in values.tolist()]
    return [int(value) for value in values]


def _build_attn_scale(
    queries: mx.array,
    *,
    beta: float,
    max_position_embeddings: int,
    cache_view: CacheView | None = None,
    cache_metadata: PageMetadata | None = None,
) -> mx.array:
    """Build per-position Llama-4-style attention scaling factors.

    Computes ``1 + beta * log(1 + floor(pos / max_position_embeddings))``
    for each query position, producing a multiplicative scale applied to
    queries before the dot product.

    Supports both batched 4-D queries ``(B, L, H, D)`` and flat 3-D
    paged queries ``(T, H, D)``.

    Args:
        queries: Query tensor, 4-D ``(B, L, H, D)`` or 3-D ``(T, H, D)``.
        beta: Llama-4 scaling beta factor.
        max_position_embeddings: Original max position embeddings before
            scaling.
        cache_view: KV cache view (provides offset for 4-D mode).
        cache_metadata: Paged-attention metadata (provides kv_lens and
            query_start_loc for 3-D mode).

    Returns:
        Scale tensor broadcastable to the query shape.
    """
    if queries.ndim == 4:
        offset = int(getattr(cache_view, "offset", 0) or 0)
        positions = mx.arange(queries.shape[1], dtype=mx.float32) + offset
        scale = 1.0 + beta * mx.log(1.0 + mx.floor(positions / float(max_position_embeddings)))
        return scale.reshape(1, queries.shape[1], 1, 1).astype(queries.dtype)

    positions: list[int] = []
    kv_lens = _as_int_list(getattr(cache_metadata, "kv_lens", None))
    qsl = _as_int_list(getattr(cache_metadata, "query_start_loc", None))
    if kv_lens and len(qsl) == len(kv_lens) + 1:
        for seq_idx, base in enumerate(kv_lens):
            start, end = qsl[seq_idx], qsl[seq_idx + 1]
            positions.extend(range(base, base + (end - start)))
    else:
        offset = int(getattr(cache_view, "offset", 0) or 0)
        positions.extend(range(offset, offset + queries.shape[0]))

    scale = 1.0 + beta * mx.log(1.0 + mx.floor(mx.array(positions, dtype=mx.float32) / float(max_position_embeddings)))
    return scale.reshape(queries.shape[0], 1, 1).astype(queries.dtype)


class Ministral3Attention(nn.Module):
    """Llama-style GQA attention with optional Llama-4 scaling for Ministral3.

    Extends standard Llama attention with an optional per-position scaling
    factor (``llama_4_scaling_beta``) that modulates query magnitudes based
    on absolute position, improving length generalization.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Base attention scaling factor (``head_dim ** -0.5``).
        llama_4_scaling_beta: Beta for Llama-4 attention scaling (0 = disabled).
        original_max_position_embeddings: Reference max position for scaling.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: RoPE module.
        attention_performer: Attention computation backend.

    Example:
        >>> config = Ministral3Config(hidden_size=4096)
        >>> attn = Ministral3Attention(config)
    """

    def __init__(self, config: Ministral3Config):
        """Initialize Ministral3 attention.

        Args:
            config: Model configuration specifying head counts, dimensions,
                RoPE parameters, and Llama-4 scaling settings.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5
        self.llama_4_scaling_beta = float(getattr(config, "llama_4_scaling_beta", 0.0))
        self.original_max_position_embeddings = int(
            getattr(config, "original_max_position_embeddings", config.max_position_embeddings)
        )

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
        """Compute attention with RoPE and optional Llama-4 scaling.

        When ``llama_4_scaling_beta`` is non-zero, query vectors are
        element-wise multiplied by a position-dependent scale factor
        before the attention dot product.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``
                or ``(total_tokens, hidden_size)`` for paged mode.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of the same leading shape with trailing
            dimension ``hidden_size``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        if self.llama_4_scaling_beta:
            queries = queries * _build_attn_scale(
                queries,
                beta=self.llama_4_scaling_beta,
                max_position_embeddings=self.original_max_position_embeddings,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
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


class Ministral3DecoderLayer(nn.Module):
    """Single Ministral3 decoder layer with configurable sliding window.

    Each layer can operate in either full-attention or sliding-window
    mode, controlled by the ``use_sliding`` flag. Uses the standard
    Llama MLP (SwiGLU) and RMSNorm.

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: Ministral3 attention sub-layer.
        mlp: Llama SwiGLU MLP sub-layer.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before the MLP.

    Example:
        >>> config = Ministral3Config(hidden_size=4096)
        >>> layer = Ministral3DecoderLayer(config, use_sliding=True)
    """

    def __init__(self, config: Ministral3Config, *, use_sliding: bool = False):
        """Initialize a Ministral3 decoder layer.

        Args:
            config: Model configuration.
            use_sliding: Whether to use sliding window attention.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Ministral3Attention(config)
        self.mlp = LlamaMLP(config)
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


@register_module(task_type=TaskType.BASE_MODULE, config=Ministral3Config, model_type="ministral3")
class Ministral3Model(EasyMLXBaseModule):
    """Base Ministral3 transformer model with per-layer sliding/full attention.

    A decoder-only transformer where each layer is independently
    configured for either full or sliding window attention via
    ``config.layer_types``. Sliding-window layers use a narrower
    attention mask and receive ``cache_metadata`` with the window size.

    Attributes:
        config_class: Associated configuration class (``Ministral3Config``).
        embed_tokens: Token embedding layer.
        layer_types: Per-layer attention type strings from config.
        sliding_window: Sliding window size.
        layers: List of ``Ministral3DecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = Ministral3Config(
        ...     hidden_size=4096,
        ...     num_hidden_layers=4,
        ...     layer_types=["full_attention", "sliding_attention"] * 2,
        ... )
        >>> model = Ministral3Model(config)
    """

    config_class = Ministral3Config

    def __init__(self, config: Ministral3Config):
        """Initialize the base Ministral3 model.

        Args:
            config: Model configuration. ``layer_types`` determines which
                layers use sliding vs. full attention.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window
        self.layers = [
            Ministral3DecoderLayer(config, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
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

        Builds separate attention masks for full and sliding-window layers.
        Sliding-window layers receive ``cache_metadata.with_sliding_window()``
        for proper paged-attention window handling.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=Ministral3Config, model_type="ministral3")
class Ministral3ForCausalLM(BaseCausalLMModule[Ministral3Model, Ministral3Config]):
    """Ministral3 causal language model with an LM head.

    Wraps ``Ministral3Model`` with a linear language-model head for
    next-token prediction. Supports weight tying.

    Attributes:
        config_class: Associated configuration class (``Ministral3Config``).

    Example:
        >>> config = Ministral3Config(hidden_size=4096, num_hidden_layers=4,
        ...     layer_types=["full_attention"] * 4)
        >>> model = Ministral3ForCausalLM(config)
    """

    config_class = Ministral3Config

    def __init__(self, config: Ministral3Config):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Ministral3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Ministral3ForCausalLM", "Ministral3Model")
