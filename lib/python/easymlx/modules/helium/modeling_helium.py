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

"""Helium MLX model implementation for serving and inference."""

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

from .helium_configuration import HeliumConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert. May be ``None``, an ``mx.array``,
            or any sequence convertible to ``mx.array``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None`` if *values* is
        ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class HeliumAttention(nn.Module):
    """Multi-head attention for Helium with traditional RoPE and bias-free output projection.

    Helium attention uses grouped-query attention (GQA) with traditional
    (interleaved) rotary position embeddings. The output projection
    (``o_proj``) never uses bias, distinguishing it from standard Llama
    attention.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Scaling factor (``head_dim ** -0.5``).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection (always bias-free).
        rope: Traditional rotary position embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = HeliumConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=4)
        >>> attn = HeliumAttention(config)
        >>> h = mx.zeros((1, 10, 256))
        >>> out = attn(h)
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HeliumConfig):
        """Initialize HeliumAttention.

        Args:
            config: Helium model configuration.

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
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
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
        """Compute attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask (broadcastable) or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata, if applicable.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
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


class HeliumDecoderLayer(nn.Module):
    """Single Helium transformer decoder layer.

    Applies pre-norm attention followed by pre-norm SwiGLU MLP with
    residual connections. Uses the LlamaMLP implementation for the
    feed-forward block.

    Attributes:
        self_attn: Helium attention sub-layer.
        mlp: SwiGLU MLP sub-layer (LlamaMLP).
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = HeliumConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=4)
        >>> layer = HeliumDecoderLayer(config)
        >>> h = mx.zeros((1, 10, 256))
        >>> out = layer(h)
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HeliumConfig):
        """Initialize HeliumDecoderLayer.

        Args:
            config: Helium model configuration.
        """
        super().__init__()
        self.self_attn = HeliumAttention(config)
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
        """Execute a single decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata, if applicable.

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


@register_module(task_type=TaskType.BASE_MODULE, config=HeliumConfig, model_type="helium")
class HeliumModel(EasyMLXBaseModule):
    """Base Helium transformer model (no LM head).

    Stacks ``num_hidden_layers`` HeliumDecoderLayer instances preceded by
    token embeddings and followed by a final RMSNorm. Helium extends
    Llama with traditional RoPE and bias-free ``o_proj``.

    Attributes:
        config_class: Associated configuration class (``HeliumConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``HeliumDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = HeliumConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4)
        >>> model = HeliumModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = HeliumConfig

    def __init__(self, config: HeliumConfig):
        """Initialize HeliumModel.

        Args:
            config: Helium model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [HeliumDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the Helium transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask broadcastable to
                ``(batch, 1, seq_len, seq_len)``.
            input_embeddings: Pre-computed embeddings; if provided,
                ``input_ids`` is ignored.
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-attention metadata, if applicable.

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
        """Remove rotary embedding inverse-frequency buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Filtered weight dictionary without ``rotary_emb.inv_freq`` or
            ``rope.inv_freq`` entries.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=HeliumConfig, model_type="helium")
class HeliumForCausalLM(BaseCausalLMModule[HeliumModel, HeliumConfig]):
    """Helium transformer with a causal language modeling head.

    Wraps ``HeliumModel`` and adds an LM head for next-token prediction.
    Word embeddings are not tied by default.

    Attributes:
        config_class: Associated configuration class (``HeliumConfig``).

    Example:
        >>> config = HeliumConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4)
        >>> model = HeliumForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = HeliumConfig

    def __init__(self, config: HeliumConfig):
        """Initialize HeliumForCausalLM.

        Args:
            config: Helium model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=HeliumModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("HeliumForCausalLM", "HeliumModel")
