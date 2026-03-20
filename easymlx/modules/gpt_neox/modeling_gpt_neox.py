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

"""GPT-NeoX MLX implementation for serving and inference.

GPT-NeoX supports partial rotary embeddings and optional parallel
residual connections (attention + MLP computed in parallel).
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

from .gpt_neox_configuration import GPTNeoXConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert values to an int32 mx.array, or return None.

    Args:
        values: Array-like input or None.

    Returns:
        An ``mx.array`` with dtype ``int32``, or None.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class GPTNeoXAttention(nn.Module):
    """GPT-NeoX attention with partial rotary embeddings.

    RoPE is applied to only a fraction (``rotary_pct``) of each head
    dimension. For example, with ``rotary_pct=0.25`` and ``head_dim=80``,
    only the first 20 dimensions get rotary embeddings while the
    remaining 60 use absolute position information.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection (with bias).
        k_proj: Key projection (with bias).
        v_proj: Value projection (with bias).
        dense: Output projection (with bias).
        rope: Partial rotary embedding module.
        attention_performer: Attention backend.

    Example::

        >>> attn = GPTNeoXAttention(GPTNeoXConfig(hidden_size=64, num_attention_heads=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPTNeoXConfig):
        """Initialize GPT-NeoX attention with partial RoPE.

        Args:
            config: Model configuration. The ``rotary_pct`` field
                determines how many dimensions of each head get RoPE.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

        rope_dims = int(self.head_dim * config.rotary_pct)
        self.rope = get_rope(
            dims=rope_dims,
            base=config.rotary_emb_base,
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
        """Compute attention with partial RoPE.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Output of shape ``[batch, seq_len, hidden_size]``.
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
        return self.dense(attn.reshape(*lead, -1))


class GPTNeoXMLP(nn.Module):
    """GPT-NeoX feed-forward network with GELU approximate activation.

    A simple 4x expansion MLP without gating:
    ``dense_4h_to_h(gelu_approx(dense_h_to_4h(x)))``.

    Attributes:
        dense_h_to_4h: Up-projection to ``4 * hidden_size``.
        dense_4h_to_h: Down-projection back to ``hidden_size``.

    Example::

        >>> mlp = GPTNeoXMLP(GPTNeoXConfig(hidden_size=64))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPTNeoXConfig):
        """Initialize GPT-NeoX MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * config.hidden_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply MLP with GELU approximate.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        return self.dense_4h_to_h(nn.gelu_approx(self.dense_h_to_4h(hidden_states)))


class GPTNeoXDecoderLayer(nn.Module):
    """Single GPT-NeoX decoder layer with optional parallel residual.

    When ``use_parallel_residual`` is True (default), attention and MLP
    are computed in parallel from the same input:
    ``h = x + attn(ln1(x)) + mlp(ln2(x))``.
    Otherwise, they run sequentially (standard pre-norm).

    Attributes:
        use_parallel_residual: Whether parallel residual mode is active.
        self_attn: GPT-NeoX attention with partial RoPE.
        mlp: GELU MLP.
        input_layernorm: LayerNorm before attention.
        post_attention_layernorm: LayerNorm before MLP (parallel)
            or after attention residual (sequential).

    Example::

        >>> layer = GPTNeoXDecoderLayer(GPTNeoXConfig(hidden_size=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPTNeoXConfig):
        """Initialize GPT-NeoX decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.self_attn = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with parallel or sequential residuals.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after attention and MLP.
        """
        if self.use_parallel_residual:
            # Parallel residual: h = x + attn(ln1(x)) + mlp(ln2(x))
            residual = hidden_states
            attn_out = self.self_attn(
                self.input_layernorm(hidden_states),
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
            mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = residual + attn_out + mlp_out
        else:
            # Sequential residual
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


@register_module(task_type=TaskType.BASE_MODULE, config=GPTNeoXConfig, model_type="gpt_neox")
class GPTNeoXModel(EasyMLXBaseModule):
    """Base GPT-NeoX transformer with partial RoPE and parallel residuals.

    Supports partial rotary embeddings (only a fraction of head
    dimensions get RoPE) and optional parallel residual connections.
    Uses LayerNorm (not RMSNorm).

    Attributes:
        embed_in: Token embedding layer.
        layers: Stack of ``GPTNeoXDecoderLayer`` instances.
        final_layer_norm: Final LayerNorm.

    Example::

        >>> model = GPTNeoXModel(GPTNeoXConfig(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GPTNeoXConfig

    def __init__(self, config: GPTNeoXConfig):
        """Initialize GPT-NeoX base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GPTNeoXDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def embed_tokens(self):
        """Alias for compatibility with BaseCausalLMModule embedding lookup."""
        return self.embed_in

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the GPT-NeoX transformer.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata.

        Returns:
            Normalized hidden states of shape
            ``[batch, seq_len, hidden_size]``.

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
            hidden_states = self.embed_in(input_ids)

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

        return self.final_layer_norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove rotary buffers and attention bias/masked_bias from checkpoints.

        Strips ``rotary_emb.inv_freq``, ``rope.inv_freq``,
        ``attention.bias``, and ``attention.masked_bias`` keys since
        rotary embeddings are computed dynamically and causal masks
        are built at runtime.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        ignore_suffixes = (
            ".attention.bias",
            ".attention.masked_bias",
            ".attention.rotary_emb.inv_freq",
            "rotary_emb.inv_freq",
            "rope.inv_freq",
        )
        return {
            key: value for key, value in weights.items() if not any(key.endswith(suffix) for suffix in ignore_suffixes)
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=GPTNeoXConfig, model_type="gpt_neox")
class GPTNeoXForCausalLM(BaseCausalLMModule[GPTNeoXModel, GPTNeoXConfig]):
    """GPT-NeoX causal language model with LM head.

    Wraps ``GPTNeoXModel`` and adds a linear LM head. Unlike most
    models in this codebase, ``tie_word_embeddings`` defaults to
    False for GPT-NeoX.

    Attributes:
        config_class: ``GPTNeoXConfig``.

    Example::

        >>> model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GPTNeoXConfig

    def __init__(self, config: GPTNeoXConfig):
        super().__init__(
            config=config,
            base_model_class=GPTNeoXModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("GPTNeoXForCausalLM", "GPTNeoXModel")
