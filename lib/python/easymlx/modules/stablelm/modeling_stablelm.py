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

"""StableLM MLX model implementation for serving and inference.

This module provides the StableLM architecture on MLX, featuring
partial RoPE, optional per-head QK LayerNorm, optional parallel
residuals, SwiGLU MLP, and LayerNorm.
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

from .stablelm_configuration import StableLMConfig

CacheView = TransformerCacheView | PageCacheView


class LayerNormPerHead(nn.Module):
    """Per-head LayerNorm for Q/K normalization in StableLM.

    Applies an independent LayerNorm to each attention head. The weight
    parameters from all per-head norms are stacked and applied via
    ``mx.fast.layer_norm`` for efficient batch computation.

    Attributes:
        norms: List of ``nn.LayerNorm`` modules, one per head.
        eps: LayerNorm epsilon.

    Example:
        >>> ln = LayerNormPerHead(head_dim=64, num_heads=8, eps=1e-5)
        >>> out = ln(mx.zeros((1, 8, 10, 64)))
    """

    def __init__(self, head_dim: int, num_heads: int, eps: float):
        """Initialize per-head LayerNorm.

        Args:
            head_dim: Dimensionality of each attention head.
            num_heads: Number of attention heads.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.norms = [nn.LayerNorm(head_dim, eps=eps, bias=False) for _ in range(num_heads)]
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply per-head LayerNorm.

        Args:
            x: Input tensor of shape ``(..., num_heads, head_dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        w = mx.stack([n.weight for n in self.norms])
        return w * mx.fast.layer_norm(x, None, None, self.eps)


class StableLMAttention(nn.Module):
    """StableLM attention with partial RoPE and optional per-head QK LayerNorm.

    Uses Grouped Query Attention (GQA) with partial Rotary Positional
    Embeddings (only a fraction of head dimensions receive rotary encoding).
    Optionally applies per-head LayerNorm to Q and K before attention.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor (``head_dim ** -0.5``).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection (always without bias).
        rope: Partial RoPE module (rotates only a fraction of dimensions).
        qk_layernorm: Whether per-head QK LayerNorm is enabled.
        attention_performer: Attention computation backend.

    Example:
        >>> config = StableLMConfig(hidden_size=2560, qk_layernorm=True)
        >>> attn = StableLMAttention(config)
    """

    def __init__(self, config: StableLMConfig):
        """Initialize StableLM attention.

        Args:
            config: Model configuration specifying head counts, partial
                rotary factor, QK LayerNorm, and bias settings.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        rotary_dim = int(config.partial_rotary_factor * self.head_dim)
        self.rope = get_rope(
            dims=rotary_dim,
            base=config.rope_theta,
            traditional=False,
            max_position_embeddings=config.max_position_embeddings,
        )

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = LayerNormPerHead(self.head_dim, self.num_heads, eps=config.layer_norm_eps)
            self.k_layernorm = LayerNormPerHead(self.head_dim, self.num_kv_heads, eps=config.layer_norm_eps)

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
        """Compute attention with partial RoPE and optional QK LayerNorm.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask (boolean array, ``"causal"`` string, or ``None``).
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.qk_layernorm:
            queries = self.q_layernorm(queries)
            keys = self.k_layernorm(keys)

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


class StableLMMLP(nn.Module):
    """SwiGLU feed-forward network for the StableLM architecture.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``. All projections
    are bias-free.

    Attributes:
        gate_proj: Linear projection for the SiLU gate.
        up_proj: Linear projection for the element-wise product branch.
        down_proj: Linear projection back to ``hidden_size``.

    Example:
        >>> config = StableLMConfig(hidden_size=2560, intermediate_size=6912)
        >>> mlp = StableLMMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 2560)))
    """

    def __init__(self, config: StableLMConfig):
        """Initialize the SwiGLU MLP.

        Args:
            config: Model configuration specifying hidden/intermediate sizes.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class StableLMDecoderLayer(nn.Module):
    """Single StableLM decoder layer with optional parallel residual connections.

    In sequential mode (default): ``x = x + attn(norm(x))`` then
    ``x = x + mlp(norm(x))``.

    In parallel mode: ``x = x + attn(norm(x)) + mlp(norm(x))``, where
    both attention and MLP receive the same normalized input.

    Attributes:
        self_attn: Multi-head attention sub-layer.
        mlp: SwiGLU feed-forward sub-layer.
        input_layernorm: LayerNorm applied before attention (and MLP in
            parallel mode).
        use_parallel_residual: Whether parallel residual mode is active.
        post_attention_layernorm: LayerNorm applied before MLP in sequential
            mode. Not created in parallel mode.

    Example:
        >>> config = StableLMConfig(use_parallel_residual=True)
        >>> layer = StableLMDecoderLayer(config)
    """

    def __init__(self, config: StableLMConfig):
        """Initialize a StableLM decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = StableLMAttention(config)
        self.mlp = StableLMMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_parallel_residual = config.use_parallel_residual
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        h = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            h,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        if self.use_parallel_residual:
            hidden_states = hidden_states + attn_out + self.mlp(h)
        else:
            hidden_states = hidden_states + attn_out
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=StableLMConfig, model_type="stablelm")
class StableLMModel(EasyMLXBaseModule):
    """Base StableLM transformer model.

    A decoder-only transformer with partial RoPE, optional QK LayerNorm,
    optional parallel residuals, SwiGLU MLP, and LayerNorm. Uses GQA
    for efficient key/value caching.

    Attributes:
        config_class: Associated configuration class (``StableLMConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``StableLMDecoderLayer`` modules.
        norm: Final LayerNorm applied to the last hidden state.

    Example:
        >>> config = StableLMConfig(hidden_size=2560, num_hidden_layers=4)
        >>> model = StableLMModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = StableLMConfig

    def __init__(self, config: StableLMConfig):
        """Initialize the base StableLM model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [StableLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` after
            the final LayerNorm.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=StableLMConfig, model_type="stablelm")
class StableLMForCausalLM(BaseCausalLMModule[StableLMModel, StableLMConfig]):
    """StableLM causal language model with an LM head.

    Wraps ``StableLMModel`` with a linear language-model head for
    next-token prediction.

    Attributes:
        config_class: Associated configuration class (``StableLMConfig``).

    Example:
        >>> config = StableLMConfig(hidden_size=2560, num_hidden_layers=4)
        >>> model = StableLMForCausalLM(config)
    """

    config_class = StableLMConfig

    def __init__(self, config: StableLMConfig):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=StableLMModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("StableLMForCausalLM", "StableLMModel")
