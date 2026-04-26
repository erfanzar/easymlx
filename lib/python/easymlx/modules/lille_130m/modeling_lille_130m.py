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

"""Lille-130M MLX model implementation for serving and inference.

Lille-130M is a compact transformer with fused QKV projections, sub-block
RMSNorm, traditional RoPE, SwiGLU MLP, and tied input/output embeddings.
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

from .lille_130m_configuration import Lille130mConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like value to an int32 mx.array.

    Args:
        values: Input values to convert, or ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Lille130mAttention(nn.Module):
    """Lille-130M attention with fused QKV projection, sub-block norm, and traditional RoPE.

    Uses a single fused linear projection for Q, K, and V, applies
    sub-block RMSNorm before the projection, and uses traditional
    (interleaved) RoPE for positional encoding.

    Attributes:
        n_head: Number of query attention heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        qkv_proj: Fused Q/K/V linear projection.
        out_proj: Output projection.
        norm: Sub-block RMSNorm applied before QKV projection.
        rope: Traditional rotary position embedding.

    Example:
        >>> config = Lille130mConfig(n_embd=64, n_head=4)
        >>> attn = Lille130mAttention(config)
    """

    def __init__(self, config: Lille130mConfig):
        """Initialize Lille-130M attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.n_head = int(config.n_head)
        self.n_kv_heads = int(config.n_kv_heads)
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_kv_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.norm = nn.RMSNorm(config.n_embd, eps=config.layer_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
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
        """Compute attention with fused QKV, sub-block norm, and RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., n_embd)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(..., n_embd)``.
        """
        lead = hidden_states.shape[:-1]
        qkv = self.qkv_proj(self.norm(hidden_states))

        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = queries.reshape(*lead, self.n_head, self.head_dim)
        keys = keys.reshape(*lead, self.n_kv_heads, self.head_dim)
        values = values.reshape(*lead, self.n_kv_heads, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.out_proj(attn.reshape(*lead, -1))


class Lille130mMLP(nn.Module):
    """Lille-130M SwiGLU MLP with sub-block RMSNorm.

    Applies RMSNorm before the SwiGLU computation. The intermediate
    dimension is computed as ``256 * round(8/3 * n_embd / 256)``.

    Attributes:
        norm: Sub-block RMSNorm applied before MLP projections.
        gate_proj: Gate projection for SwiGLU.
        up_proj: Up projection for SwiGLU.
        down_proj: Down projection.

    Example:
        >>> config = Lille130mConfig(n_embd=64)
        >>> mlp = Lille130mMLP(config)
    """

    def __init__(self, config: Lille130mConfig):
        """Initialize Lille-130M MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        hidden_dim = 256 * round(int(8 * config.n_embd / 3) / 256)
        self.norm = nn.RMSNorm(config.n_embd, eps=config.layer_norm_eps)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU feed-forward pass with sub-block norm.

        Args:
            hidden_states: Input tensor of shape ``(..., n_embd)``.

        Returns:
            Output tensor of shape ``(..., n_embd)``.
        """
        h = self.norm(hidden_states)
        return self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))


class Lille130mBlock(nn.Module):
    """Single Lille-130M decoder block with residual connections.

    Contains an attention sub-layer and an MLP sub-layer, each with its
    own internal sub-block normalization and residual connection.

    Attributes:
        attention: Fused QKV attention sub-layer with sub-block norm.
        feed_forward: SwiGLU MLP sub-layer with sub-block norm.

    Example:
        >>> config = Lille130mConfig(n_embd=64, n_head=4)
        >>> block = Lille130mBlock(config)
    """

    def __init__(self, config: Lille130mConfig):
        """Initialize a decoder block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attention = Lille130mAttention(config)
        self.feed_forward = Lille130mMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run attention and MLP with residual connections.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, n_embd)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, n_embd)``.
        """
        hidden_states = hidden_states + self.attention(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Lille130mConfig, model_type="lille-130m")
class Lille130mModel(EasyMLXBaseModule):
    """Base Lille-130M compact transformer model.

    Embeds tokens, passes through decoder blocks with sub-block norms,
    and applies final RMSNorm.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        layers: List of ``Lille130mBlock`` decoder blocks.
        norm: Final RMSNorm.

    Example:
        >>> config = Lille130mConfig(vocab_size=1000, n_embd=64, n_layer=2)
        >>> model = Lille130mModel(config)
    """

    config_class = Lille130mConfig

    def __init__(self, config: Lille130mConfig):
        """Initialize the base Lille-130M model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = [Lille130mBlock(config) for _ in range(config.n_layer)]
        self.norm = nn.RMSNorm(config.n_embd, eps=config.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the Lille-130M backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, n_embd)``.

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
        """Sanitize upstream weights for Lille-130M.

        Drops ``rotary_emb`` keys and remaps ``tok_embeddings`` to
        ``embed_tokens``.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb" in k:
                continue

            k = k.replace("tok_embeddings", "embed_tokens")
            sanitized[k] = v
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=Lille130mConfig, model_type="lille-130m")
class Lille130mForCausalLM(BaseCausalLMModule[Lille130mModel, Lille130mConfig]):
    """Lille-130M causal language model with an LM head.

    Wraps ``Lille130mModel`` with a language modeling head. Supports
    tied input/output embeddings (default).

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = Lille130mConfig(vocab_size=1000, n_embd=64, n_layer=2)
        >>> model = Lille130mForCausalLM(config)
    """

    config_class = Lille130mConfig

    def __init__(self, config: Lille130mConfig):
        """Initialize the Lille-130M causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Lille130mModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.base_model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("Lille130mForCausalLM", "Lille130mModel")
