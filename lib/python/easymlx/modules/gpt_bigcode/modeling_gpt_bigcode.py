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

"""GPT-BigCode MLX implementation for serving and inference.

GPT-BigCode is a GPT2-style model with multi-query attention support.
Uses absolute position embeddings (no RoPE) and LayerNorm.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from .gpt_bigcode_configuration import GPTBigCodeConfig

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


class GPTBigCodeAttention(nn.Module):
    """GPT-BigCode attention with multi-query attention (MQA) support.

    When ``multi_query`` is enabled in the config, a single key/value
    head is used for all query heads, reducing KV cache size by
    ``n_head`` times. No rotary embeddings are used; position info
    comes from absolute position embeddings at the model level.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (1 for MQA).
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection (to ``num_kv_heads * head_dim``).
        v_proj: Value projection (to ``num_kv_heads * head_dim``).
        c_proj: Output projection.
        attention_performer: Attention backend.

    Example::

        >>> attn = GPTBigCodeAttention(GPTBigCodeConfig(n_embd=64, n_head=4, multi_query=True))
        >>> attn.num_kv_heads
        1
    """

    def __init__(self, config: GPTBigCodeConfig):
        """Initialize GPT-BigCode attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.n_head)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.n_embd // config.n_head)
        self.scale = self.head_dim**-0.5

        bias = config.attention_bias

        self.q_proj = nn.Linear(config.n_embd, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.n_embd, self.num_kv_heads * self.head_dim, bias=bias)
        self.c_proj = nn.Linear(self.num_heads * self.head_dim, config.n_embd, bias=bias)

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
        """Compute multi-query attention (no RoPE).

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Output of shape ``[batch, seq_len, n_embd]``.
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
            rope=None,
        )
        return self.c_proj(attn.reshape(*lead, -1))


class GPTBigCodeMLP(nn.Module):
    """GPT-BigCode feed-forward network with GELU approximate activation.

    Attributes:
        fc_in: Up-projection to ``n_inner``.
        fc_out: Down-projection back to ``n_embd``.

    Example::

        >>> mlp = GPTBigCodeMLP(GPTBigCodeConfig(n_embd=64, n_inner=256))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPTBigCodeConfig):
        """Initialize GPT-BigCode MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        bias = config.mlp_bias
        self.fc_in = nn.Linear(config.n_embd, config.n_inner, bias=bias)
        self.fc_out = nn.Linear(config.n_inner, config.n_embd, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply MLP with GELU approximate activation.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.

        Returns:
            Output of the same shape.
        """
        return self.fc_out(nn.gelu_approx(self.fc_in(hidden_states)))


class GPTBigCodeBlock(nn.Module):
    """Single GPT-BigCode transformer block with pre-norm and residuals.

    Attributes:
        self_attn: Multi-query attention.
        mlp: GELU MLP.
        ln_1: LayerNorm before attention.
        ln_2: LayerNorm before MLP.

    Example::

        >>> block = GPTBigCodeBlock(GPTBigCodeConfig(n_embd=64))
        >>> out = block(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPTBigCodeConfig):
        """Initialize GPT-BigCode block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = GPTBigCodeAttention(config)
        self.mlp = GPTBigCodeMLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through one GPT-BigCode block.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after attention and MLP.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=GPTBigCodeConfig, model_type="gpt_bigcode")
class GPTBigCodeModel(EasyMLXBaseModule):
    """Base GPT-BigCode transformer with MQA and absolute position embeddings.

    Uses learned position embeddings (``wpe``) and supports multi-query
    attention for efficient KV caching during code generation.

    Attributes:
        wte: Token embedding layer.
        wpe: Absolute position embedding layer.
        layers: Stack of ``GPTBigCodeBlock`` instances.
        ln_f: Final LayerNorm.

    Example::

        >>> model = GPTBigCodeModel(GPTBigCodeConfig(vocab_size=256, n_embd=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GPTBigCodeConfig

    def __init__(self, config: GPTBigCodeConfig):
        """Initialize GPT-BigCode base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = [GPTBigCodeBlock(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    @property
    def embed_tokens(self):
        """Return token embedding layer (alias ``wte`` for compatibility)."""
        return self.wte

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with absolute position embeddings.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata.

        Returns:
            Normalized hidden states of shape ``[batch, seq_len, n_embd]``.

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
            hidden_states = self.wte(input_ids)

        if hidden_states.ndim == 3:
            seq_len = hidden_states.shape[1]
        else:
            seq_len = hidden_states.shape[0]

        if cache_views is not None and len(cache_views) > 0:
            cache_view_0 = cache_views[0]
            offset = cache_view_0.offset if hasattr(cache_view_0, "offset") else 0
        else:
            offset = 0

        position_ids = mx.arange(offset, offset + seq_len, dtype=mx.int32)
        hidden_states = hidden_states + self.wpe(position_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len_actual = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len_actual == 1):
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len_actual)

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.ln_f(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove attention bias/masked_bias buffer keys from checkpoints.

        These keys store the causal mask as a buffer and are not
        needed since the mask is computed dynamically.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        return {
            key: value
            for key, value in weights.items()
            if not key.endswith(".attn.bias") and not key.endswith(".attn.masked_bias")
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=GPTBigCodeConfig, model_type="gpt_bigcode")
class GPTBigCodeForCausalLM(BaseCausalLMModule[GPTBigCodeModel, GPTBigCodeConfig]):
    """GPT-BigCode causal language model with LM head.

    Wraps ``GPTBigCodeModel`` and adds a linear LM head with optional
    tied embeddings. Designed for code generation with efficient
    multi-query attention.

    Attributes:
        config_class: ``GPTBigCodeConfig``.

    Example::

        >>> model = GPTBigCodeForCausalLM(GPTBigCodeConfig(vocab_size=256, n_embd=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GPTBigCodeConfig

    def __init__(self, config: GPTBigCodeConfig):
        super().__init__(
            config=config,
            base_model_class=GPTBigCodeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("GPTBigCodeForCausalLM", "GPTBigCodeModel")
