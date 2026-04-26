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

"""Gemma3Text MLX implementation (serving/inference only).

Structure:
  Gemma3TextConfig -> Gemma3TextAttention -> Gemma3TextMLP -> Gemma3TextDecoderLayer
  -> Gemma3TextModel -> Gemma3TextForCausalLM

Key features:
  - Sliding window + full attention alternation (every sliding_window_pattern-th layer is full)
  - Q/K LayerNorm (offset-based: 1 + weight) after projection, before RoPE
  - Embedding scaled by sqrt(hidden_size)
  - GELU approx activation
  - query_pre_attn_scalar for attention scaling
  - clip_residual to prevent FP16 overflow
  - 4 norms per layer (input, post_attention, pre_feedforward, post_feedforward)
"""

from __future__ import annotations

from functools import partial

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

from .gemma3_text_configuration import Gemma3TextConfig

CacheView = TransformerCacheView | PageCacheView


class Gemma3TextRMSNorm(nn.Module):
    """Gemma3-style RMSNorm with offset weights (``1 + weight``).

    Unlike standard RMSNorm where the scale is ``weight``, this variant
    uses ``1 + weight`` so that the default initialization (all-zeros
    weights) produces identity scaling.

    Attributes:
        weight: Learnable scale offset, initialized to ones.
        eps: Epsilon for numerical stability.

    Args:
        dims: Feature dimensionality.
        eps: Epsilon for numerical stability. Defaults to 1e-5.

    Example::

        >>> norm = Gemma3TextRMSNorm(64)
        >>> out = norm(mx.ones((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        """Initialize offset RMSNorm.

        Args:
            dims: Feature dimensionality.
            eps: Epsilon for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMSNorm with ``1 + weight`` scaling.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor of the same shape.
        """
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """Add residual with clamping to prevent FP16 overflow.

    When the input dtype is ``float16``, the addition is performed in
    ``float32`` and the result is clamped to the ``float16`` representable
    range before casting back. For other dtypes, this is a simple add.

    Args:
        x: First operand (residual).
        y: Second operand (sub-layer output).

    Returns:
        ``x + y``, clamped if float16.
    """
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)


class Gemma3TextAttention(nn.Module):
    """Gemma3Text attention with Q/K LayerNorm and sliding/full window support.

    Applies per-head RMSNorm (offset-based) to query and key projections
    before rotary embeddings. Each layer is either a sliding-window or
    full-attention layer based on ``sliding_window_pattern``. Sliding
    layers use ``rope_local_base_freq`` while full layers use
    ``rope_theta`` with optional scaling.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scale from ``query_pre_attn_scalar``.
        is_sliding: Whether this layer uses sliding window attention.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: Per-head offset RMSNorm for queries.
        k_norm: Per-head offset RMSNorm for keys.
        rope: Rotary embedding (local or global base frequency).
        attention_performer: Attention computation backend.

    Example::

        >>> attn = Gemma3TextAttention(
        ...     Gemma3TextConfig(hidden_size=64, num_attention_heads=4),
        ...     layer_idx=0,
        ... )
        >>> attn.is_sliding
        True
    """

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        """Initialize Gemma3Text attention layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index, used to determine
                sliding vs. full attention based on
                ``sliding_window_pattern``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = 1.0 / (config.query_pre_attn_scalar**0.5)
        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern != 0

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma3TextRMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3TextRMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)

        rope_base = config.rope_local_base_freq if self.is_sliding else config.rope_theta
        self.rope = get_rope(
            dims=self.head_dim,
            base=rope_base,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling if not self.is_sliding else None,
            max_position_embeddings=config.max_position_embeddings if not self.is_sliding else None,
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
        """Compute attention with Q/K normalization and RoPE.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache for decoding.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Output of shape ``[batch, seq_len, hidden_size]``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

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


class Gemma3TextMLP(nn.Module):
    """Gated feed-forward network with GELU approximate for Gemma3Text.

    Attributes:
        gate_proj: Gating branch projection.
        up_proj: Value branch projection.
        down_proj: Output projection back to hidden size.

    Example::

        >>> mlp = Gemma3TextMLP(Gemma3TextConfig(hidden_size=64, intermediate_size=128))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma3TextConfig):
        """Initialize Gemma3Text MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply gated MLP with GELU approximate activation.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        return self.down_proj(nn.gelu_approx(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Gemma3TextDecoderLayer(nn.Module):
    """Single Gemma3Text decoder layer with 4 norms and clip_residual.

    Uses four offset RMSNorm layers and ``clip_residual`` for safe
    FP16 residual additions. Each layer is either sliding-window or
    full attention based on ``sliding_window_pattern``.

    Attributes:
        self_attn: Gemma3Text attention with Q/K norms.
        mlp: GELU-approximate gated MLP.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm after attention output.
        pre_feedforward_layernorm: RMSNorm before MLP.
        post_feedforward_layernorm: RMSNorm after MLP output.
        is_sliding: Whether this layer uses sliding window attention.

    Example::

        >>> layer = Gemma3TextDecoderLayer(
        ...     Gemma3TextConfig(hidden_size=64), layer_idx=0
        ... )
        >>> layer.is_sliding
        True
    """

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        """Initialize Gemma3Text decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index for sliding/full selection.
        """
        super().__init__()
        self.self_attn = Gemma3TextAttention(config, layer_idx)
        self.mlp = Gemma3TextMLP(config)
        self.input_layernorm = Gemma3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = self.self_attn.is_sliding

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with clip_residual additions.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Hidden states after attention and MLP.
        """
        r = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        h = clip_residual(hidden_states, self.post_attention_layernorm(r))
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = clip_residual(h, self.post_feedforward_layernorm(r))
        return out


@register_module(task_type=TaskType.BASE_MODULE, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3TextModel(EasyMLXBaseModule):
    """Base Gemma3Text transformer with sliding/full attention alternation.

    Alternates between sliding window attention (local) and full
    attention (global) layers based on ``sliding_window_pattern``.
    Sliding layers use a separate RoPE base frequency and a windowed
    attention mask. Embeddings are scaled by ``sqrt(hidden_size)``.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: Stack of ``Gemma3TextDecoderLayer`` instances.
        norm: Final offset RMSNorm.

    Example::

        >>> model = Gemma3TextModel(Gemma3TextConfig(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        """Initialize Gemma3Text base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.norm = Gemma3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with sliding/full attention alternation.

        Builds separate masks for sliding-window and full-attention
        layers. Sliding layers also receive modified cache metadata
        with the configured window size.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata for paged attention.

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
            hidden_states = self.embed_tokens(input_ids)

        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        mask: mx.array | str | None = None
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.is_sliding for layer in self.layers):
                    sliding_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.config.sliding_window,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.is_sliding:
                layer_metadata = cache_metadata.with_sliding_window(self.config.sliding_window)
            layer_mask = sliding_mask if layer.is_sliding else mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3TextForCausalLM(BaseCausalLMModule[Gemma3TextModel, Gemma3TextConfig]):
    """Gemma3Text model with a causal language modeling head.

    Wraps ``Gemma3TextModel`` and adds a linear LM head. Supports
    tied word embeddings by default.

    Attributes:
        config_class: ``Gemma3TextConfig``.

    Example::

        >>> model = Gemma3TextForCausalLM(Gemma3TextConfig(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        """Initialize Gemma3Text causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma3TextModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Gemma3TextForCausalLM", "Gemma3TextModel")
