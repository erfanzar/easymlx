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

"""Cohere2 MLX model implementation for serving and inference.

Cohere2 extends Cohere with sliding window attention. Every Nth layer
(sliding_window_pattern) uses full attention; other layers use sliding window.
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

from .cohere2_configuration import Cohere2Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Cohere2MLP(nn.Module):
    """SwiGLU feed-forward network for the Cohere2 model.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: Cohere2Config):
        """Initialize the Cohere2 MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Cohere2Attention(nn.Module):
    """Multi-head attention with traditional RoPE and sliding window support.

    Uses traditional (not interleaved) RoPE layout. Sliding window
    is controlled by the ``use_sliding`` flag, which determines mask
    construction at the model level.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        use_sliding: Whether this layer uses sliding window attention.
        q_proj: Query linear projection.
        k_proj: Key linear projection.
        v_proj: Value linear projection.
        o_proj: Output linear projection.
        rope: Traditional rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = Cohere2Config(hidden_size=4096)
        >>> attn = Cohere2Attention(config, use_sliding=True)
    """

    def __init__(self, config: Cohere2Config, *, use_sliding: bool = False):
        """Initialize Cohere2 attention.

        Args:
            config: Model configuration with attention hyperparameters.
            use_sliding: Whether this layer uses sliding window attention.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5
        self.use_sliding = use_sliding

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
            scaling_config=None,
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
        """Run attention forward pass with traditional RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
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


class Cohere2DecoderLayer(nn.Module):
    """Single Cohere2 decoder layer with parallel residual and optional sliding window.

    Uses a single LayerNorm before both attention and MLP (parallel
    residual). The attention and MLP outputs are added together with
    the residual: ``output = attn(norm(x)) + mlp(norm(x)) + x``.

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: Cohere2 attention module.
        mlp: SwiGLU MLP.
        input_layernorm: LayerNorm applied before both branches.
    """

    def __init__(self, config: Cohere2Config, *, use_sliding: bool = False):
        """Initialize the Cohere2 decoder layer.

        Args:
            config: Model configuration.
            use_sliding: Whether this layer uses sliding window attention.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Cohere2Attention(config, use_sliding=use_sliding)
        self.mlp = Cohere2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the parallel-residual decoder layer forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        h = self.input_layernorm(hidden_states)
        attn_h = self.self_attn(
            h,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        ff_h = self.mlp(h)
        return attn_h + ff_h + hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Cohere2Config, model_type="cohere2")
class Cohere2Model(EasyMLXBaseModule):
    """Base Cohere2 transformer model with sliding window attention.

    Implements a decoder-only transformer with parallel residual
    connections, traditional RoPE, and a sliding window attention
    pattern where every ``sliding_window_pattern``-th layer uses full
    attention.

    Attributes:
        config_class: The configuration class (``Cohere2Config``).
        sliding_window: Sliding window size.
        sliding_window_pattern: Full attention layer interval.
        embed_tokens: Token embedding layer.
        layers: List of ``Cohere2DecoderLayer`` decoder blocks.
        norm: Final LayerNorm normalization.

    Example::

        >>> config = Cohere2Config(vocab_size=256000, hidden_size=4096)
        >>> model = Cohere2Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = Cohere2Config

    def __init__(self, config: Cohere2Config):
        """Initialize the Cohere2 base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Cohere2DecoderLayer(
                config,
                use_sliding=((i + 1) % config.sliding_window_pattern != 0),
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the transformer forward pass.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

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
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = _as_int_array(attention_mask)
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
            layer_meta = cache_metadata
            if layer.use_sliding and cache_metadata is not None:
                layer_meta = cache_metadata.with_sliding_window(self.sliding_window)
            layer_mask = sliding_mask if layer.use_sliding else mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_meta,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove rotary embedding buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Cohere2Config, model_type="cohere2")
class Cohere2ForCausalLM(BaseCausalLMModule[Cohere2Model, Cohere2Config]):
    """Cohere2 model with a causal language modeling head.

    Wraps ``Cohere2Model`` with a linear projection to vocabulary logits.
    Applies ``logit_scale`` to the output logits. Supports tied word
    embeddings.

    Attributes:
        config_class: The configuration class (``Cohere2Config``).

    Example::

        >>> config = Cohere2Config(vocab_size=256000, hidden_size=4096)
        >>> model = Cohere2ForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = Cohere2Config

    def __init__(self, config: Cohere2Config):
        """Initialize the Cohere2 causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Cohere2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits with Cohere2's logit scaling.

        Args:
            hidden_states: Final hidden states of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Scaled logits of shape ``(batch, seq_len, vocab_size)``.
        """
        logits = super().compute_lm_logits(hidden_states)
        logits = logits * self.config.logit_scale
        return logits


__all__ = ("Cohere2ForCausalLM", "Cohere2Model")
