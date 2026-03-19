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

"""Qwen (v1) MLX model implementation for serving and inference.

This module provides the Qwen transformer architecture on MLX, including
multi-head attention with rotary embeddings, a SiLU-gated MLP, and a
causal language model wrapper. The unified ``__call__`` API at every layer
accepts both ``TransformerCacheView`` and ``PageCacheView`` for flexible serving.
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

from .qwen_configuration import QwenConfig

CacheView = TransformerCacheView | PageCacheView


class QwenAttention(nn.Module):
    """Multi-head self-attention for the Qwen model.

    Uses a fused QKV projection with bias and rotary positional embeddings.

    Attributes:
        num_attention_heads: Number of attention heads.
        scale: Scaling factor for attention logits.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.
        c_attn: Fused query/key/value linear projection.
        c_proj: Output linear projection.
    """

    def __init__(self, config: QwenConfig):
        """Initialize the Qwen attention module.

        Args:
            config: Qwen configuration.
        """
        super().__init__()
        hidden_size = config.hidden_size
        self.num_attention_heads = int(config.num_attention_heads)
        head_dim = hidden_size // self.num_attention_heads
        self.scale = head_dim**-0.5

        self.rope = get_rope(dims=head_dim, base=10000.0, traditional=False)
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

        proj_size = config.kv_channels * self.num_attention_heads
        self.c_attn = nn.Linear(hidden_size, proj_size * 3, bias=True)
        self.c_proj = nn.Linear(hidden_size, proj_size, bias=not config.no_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute multi-head attention.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``
                or ``(total_tokens, hidden_size)`` for paged mode.
            mask: Attention mask.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``.
        """
        qkv = self.c_attn(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)
        lead = hidden_states.shape[:-1]
        q = q.reshape(*lead, self.num_attention_heads, -1)
        k = k.reshape(*lead, self.num_attention_heads, -1)
        v = v.reshape(*lead, self.num_attention_heads, -1)
        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.c_proj(attn.reshape(*lead, -1))


class QwenMLP(nn.Module):
    """Feed-forward MLP for the Qwen model.

    Uses a gated SiLU activation: ``c_proj(w1(x) * silu(w2(x)))``.

    Attributes:
        w1: First linear projection.
        w2: Second linear projection (gate).
        c_proj: Output projection.
    """

    def __init__(self, config: QwenConfig):
        """Initialize the Qwen MLP.

        Args:
            config: Qwen configuration.
        """
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        self.c_proj = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=not config.no_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        return self.c_proj(a1 * nn.silu(a2))


class QwenDecoderLayer(nn.Module):
    """Single decoder layer for the Qwen model.

    Consists of RMS-normalized self-attention followed by a RMS-normalized
    MLP, each with residual connections.

    Attributes:
        ln_1: Pre-attention RMS normalization.
        attn: Multi-head self-attention module.
        ln_2: Post-attention RMS normalization.
        mlp: Feed-forward MLP module.
    """

    def __init__(self, config: QwenConfig):
        """Initialize the decoder layer.

        Args:
            config: Qwen configuration.
        """
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = QwenAttention(config)
        self.ln_2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QwenMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same shape as the input.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=QwenConfig, model_type="qwen")
class QwenModel(EasyMLXBaseModule):
    """Base Qwen transformer model.

    Consists of token embeddings, a stack of decoder layers, and a final
    RMS normalization.

    Attributes:
        config_class: The associated configuration class (``QwenConfig``).
        wte: Token embedding layer.
        layers: List of decoder layers.
        ln_f: Final RMS normalization.
    """

    config_class = QwenConfig

    def __init__(self, config: QwenConfig):
        """Initialize the base Qwen model.

        Args:
            config: Qwen configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Qwen transformer stack.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; overrides ``input_ids``
                embedding lookup when provided.
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` or
            ``(total_tokens, hidden_size)`` in paged mode.

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
            hidden_states = self.wte(input_ids)

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

        return self.ln_f(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=QwenConfig, model_type="qwen")
class QwenForCausalLM(BaseCausalLMModule[QwenModel, QwenConfig]):
    """Qwen model with a causal language modeling head.

    Wraps ``QwenModel`` and adds vocabulary projection to produce next-token
    logits. Embeddings are tied by default.

    Attributes:
        config_class: The associated configuration class (``QwenConfig``).
    """

    config_class = QwenConfig

    def __init__(self, config: QwenConfig):
        """Initialize the causal language model.

        Args:
            config: Qwen configuration.
        """
        super().__init__(
            config=config,
            base_model_class=QwenModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("QwenForCausalLM", "QwenModel")
