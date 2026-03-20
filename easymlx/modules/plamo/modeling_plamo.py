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

"""PLaMo MLX implementation (serving/inference only).

Structure:
  PlamoConfig -> PlamoAttention -> PlamoMLP -> PlamoDecoderLayer
  -> PlamoModel -> PlamoForCausalLM

Key features:
  - Shared K/V heads: K and V are computed for ceil(num_heads/n_shared_head) heads,
    then tiled to match the full number of query heads.
  - Parallel attention + MLP: same normalized input goes to both, outputs are added.
  - SwiGLU MLP activation.
  - Single RMSNorm per layer (pre-norm).
"""

from __future__ import annotations

import math

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

from .plamo_configuration import PlamoConfig

CacheView = TransformerCacheView | PageCacheView


class PlamoAttention(nn.Module):
    """PLaMo attention with shared KV heads.

    Uses ``n_shared_head`` to determine the KV head count: every
    ``n_shared_head`` query heads share a single KV head. This is
    equivalent to grouped-query attention with automatic group sizing.

    Attributes:
        num_heads: Number of query attention heads.
        head_dim: Per-head dimensionality.
        num_kv_heads: Number of key/value heads.
        n_shared_head: Number of query heads per KV head.
        scale: Attention logit scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example:
        >>> attn = PlamoAttention(config)
        >>> out = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: PlamoConfig):
        """Initialize PLaMo attention layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = math.ceil(self.num_heads / config.n_shared_head)
        self.n_shared_head = int(config.n_shared_head)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
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
        """Compute multi-head attention with shared KV heads.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
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


class PlamoMLP(nn.Module):
    """SwiGLU feed-forward network for PLaMo.

    Attributes:
        gate_proj: Gate projection linear layer.
        up_proj: Up projection linear layer.
        down_proj: Down projection linear layer.

    Example:
        >>> mlp = PlamoMLP(config)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: PlamoConfig):
        """Initialize PLaMo MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU feed-forward transformation.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class PlamoDecoderLayer(nn.Module):
    """PLaMo decoder layer with parallel attention + MLP.

    Both attention and MLP receive the same RMSNorm-ed input, and their
    outputs are summed with the residual. Uses a single pre-norm
    (no separate post-attention norm).

    Attributes:
        self_attn: Multi-head attention module.
        mlp: SwiGLU feed-forward network.
        norm: Shared pre-norm RMSNorm.

    Example:
        >>> layer = PlamoDecoderLayer(config)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: PlamoConfig):
        """Initialize PLaMo decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = PlamoAttention(config)
        self.mlp = PlamoMLP(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer with parallel attention + MLP.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        attn_out = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        mlp_out = self.mlp(hidden_states)

        return residual + attn_out + mlp_out


@register_module(task_type=TaskType.BASE_MODULE, config=PlamoConfig, model_type="plamo")
class PlamoModel(EasyMLXBaseModule):
    """Base PLaMo transformer model without a language modeling head.

    PLaMo features parallel attention + MLP (both receive the same
    normalized input), shared KV heads via ``n_shared_head``, and SwiGLU
    activation.

    Attributes:
        config_class: Associated configuration class (``PlamoConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``PlamoDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> model = PlamoModel(config)
        >>> hidden = model(input_ids)
    """

    config_class = PlamoConfig

    def __init__(self, config: PlamoConfig):
        """Initialize the base PLaMo model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final RMSNorm.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=PlamoConfig, model_type="plamo")
class PlamoForCausalLM(BaseCausalLMModule[PlamoModel, PlamoConfig]):
    """PLaMo model with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class (``PlamoConfig``).

    Example:
        >>> model = PlamoForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = PlamoConfig

    def __init__(self, config: PlamoConfig):
        """Initialize the PLaMo causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=PlamoModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("PlamoForCausalLM", "PlamoModel")
