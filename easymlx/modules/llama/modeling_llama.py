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

"""Llama MLX implementation (serving/inference only).

Structure mirrors EasyDeL's llama:
  LlamaConfig -> LlamaAttention -> LlamaMLP -> LlamaDecoderLayer -> LlamaModel -> LlamaForCausalLM

Unified ``__call__`` at every level -- cache_view is either
``TransformerCacheView`` (standard) or ``PageCache`` (paged serving).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCache,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .llama_configuration import LlamaConfig

CacheView = TransformerCacheView | PageCache


class LlamaAttention(nn.Module):
    """Multi-head attention layer for the Llama model.

    Implements grouped-query attention (GQA) with rotary position embeddings.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
    """

    def __init__(self, config: LlamaConfig):
        """Initializes the attention layer.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

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
        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Computes multi-head attention with RoPE.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output tensor with the same leading dimensions as input.
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


class LlamaMLP(nn.Module):
    """Feed-forward network (SwiGLU) for the Llama model.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
    """

    def __init__(self, config: LlamaConfig):
        """Initializes the MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies the SwiGLU MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer for the Llama model.

    Applies pre-norm attention and MLP with residual connections.
    Optionally uses sliding window attention.

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: Attention sub-layer.
        mlp: Feed-forward sub-layer.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Pre-MLP normalization.
    """

    def __init__(self, config: LlamaConfig, *, use_sliding: bool = False):
        """Initializes a decoder layer.

        Args:
            config: Model configuration.
            use_sliding: Whether to use sliding window attention.
                Defaults to False.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = LlamaAttention(config)
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
        """Runs the decoder layer forward pass.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output hidden states tensor.
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


@register_module(task_type=TaskType.BASE_MODULE, config=LlamaConfig, model_type="llama")
class LlamaModel(EasyMLXBaseModule):
    """Base Llama transformer model.

    Features optional sliding window attention with per-layer attention
    type control.

    Attributes:
        config_class: The configuration class (``LlamaConfig``).
        embed_tokens: Token embedding layer.
        layer_types: Per-layer attention type list.
        sliding_window: Sliding window size, or None.
        layers: List of decoder layers.
        norm: Final RMS normalization.
    """

    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        """Initializes the base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window
        self.layers = [
            LlamaDecoderLayer(config, use_sliding=layer_type == "sliding_attention") for layer_type in self.layer_types
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
        """Runs the transformer forward pass with sliding window support.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            input_embeddings: Optional pre-computed embeddings.
            cache_views: Optional KV cache views.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Final hidden states after all layers and normalization.

        Raises:
            ValueError: If ``cache_views`` length mismatches layer count.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=LlamaConfig, model_type="llama")
class LlamaForCausalLM(BaseCausalLMModule[LlamaModel, LlamaConfig]):
    """Llama model with a causal language modeling head.

    Extends ``BaseCausalLMModule`` with the Llama base model. Uses tied
    word embeddings by default.

    Attributes:
        config_class: The configuration class (``LlamaConfig``).
    """

    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        """Initializes the causal LM model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=LlamaModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("LlamaForCausalLM", "LlamaModel")
