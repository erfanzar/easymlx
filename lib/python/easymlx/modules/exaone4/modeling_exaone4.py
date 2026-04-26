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

"""Exaone4 MLX model implementation for serving and inference.

Extends Exaone with sliding window attention and Q/K RMSNorm.
Layers alternate between local (sliding window) and global (full) attention
based on the sliding_window_pattern string (e.g., "LLGLLG..." where L=local,
G=global).
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

from .exaone4_configuration import Exaone4Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an optional array-like to an ``mx.array`` of int32.

    Args:
        values: Array-like values, or ``None``.

    Returns:
        An ``mx.array`` with dtype int32, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Exaone4MLP(nn.Module):
    """SwiGLU MLP for the Exaone4 model."""

    def __init__(self, config: Exaone4Config):
        """Initialize the MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run the MLP forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Exaone4Attention(nn.Module):
    """Multi-head attention with optional Q/K RMSNorm and sliding window.

    When ``is_local`` is ``True``, this layer uses sliding window attention
    with a restricted causal mask. Both local and global layers use RoPE.
    Q/K normalization is applied when ``use_qk_norm`` is enabled in config.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        scale: Attention scaling factor.
        is_local: Whether this layer uses sliding window attention.
        use_qk_norm: Whether Q/K RMSNorm is applied.
    """

    def __init__(self, config: Exaone4Config, *, is_local: bool = False):
        """Initialize Exaone4 attention.

        Args:
            config: Model configuration.
            is_local: Whether this layer uses sliding window attention.
                Defaults to ``False``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5
        self.is_local = is_local
        self.use_rope = is_local or True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Run the attention forward pass with optional QK-norm and sliding window.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask (may be a sliding window mask).
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

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


class Exaone4DecoderLayer(nn.Module):
    """Single Exaone4 decoder layer with post-norm residual.

    Uses post-attention and post-feedforward layer norms applied to the
    residual branch before addition, rather than the pre-norm pattern.

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: Attention module (local or global).
        mlp: SwiGLU MLP module.
        post_attention_layernorm: Post-attention RMSNorm.
        post_feedforward_layernorm: Post-MLP RMSNorm.
    """

    def __init__(self, config: Exaone4Config, *, is_local: bool = False):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            is_local: Whether this layer uses sliding window attention.
                Defaults to ``False``.
        """
        super().__init__()
        self.use_sliding = is_local
        self.self_attn = Exaone4Attention(config, is_local=is_local)
        self.mlp = Exaone4MLP(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the post-norm residual decoder layer forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask (may be a sliding window mask).
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(batch, seq_len, hidden_size)``.
        """
        r = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        h = hidden_states + self.post_attention_layernorm(r)
        r = self.mlp(h)
        out = h + self.post_feedforward_layernorm(r)
        return out


def _parse_sliding_window_pattern(pattern: str | None, num_layers: int) -> list[bool]:
    """Parse the sliding_window_pattern string into per-layer is_local flags.

    The pattern is a string like "LLGLLG" where L=local (sliding), G=global.
    It repeats cyclically across layers.
    """
    if pattern is None:
        return [False] * num_layers
    result = []
    for i in range(num_layers):
        char = pattern[i % len(pattern)]
        result.append(char == "L")
    return result


@register_module(task_type=TaskType.BASE_MODULE, config=Exaone4Config, model_type="exaone4")
class Exaone4Model(EasyMLXBaseModule):
    """Base Exaone4 transformer model with sliding window attention.

    Layers alternate between local (sliding window) and global (full)
    attention based on the ``sliding_window_pattern`` string. Builds
    separate masks for local and global layers.

    Attributes:
        config_class: The configuration class (``Exaone4Config``).
        sliding_window: Window size for local attention.
        embed_tokens: Token embedding layer.
        layers: List of ``Exaone4DecoderLayer`` instances.
        norm: Final RMS normalization.

    Example::

        >>> config = Exaone4Config(sliding_window_pattern="LLGLLG")
        >>> model = Exaone4Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = Exaone4Config

    def __init__(self, config: Exaone4Config):
        """Initialize the Exaone4 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.sliding_window = config.sliding_window
        is_local_flags = _parse_sliding_window_pattern(config.sliding_window_pattern, config.num_hidden_layers)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Exaone4DecoderLayer(config, is_local=is_local_flags[i]) for i in range(config.num_hidden_layers)]
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
        """Run the transformer forward pass with sliding window support.

        Builds separate masks for local (sliding window) and global attention
        layers. Local layers receive a windowed mask; global layers receive
        the full causal mask.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings instead of ``input_ids``.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

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
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.use_sliding for layer in self.layers) and self.sliding_window is not None:
                    sliding_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.sliding_window,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_meta = cache_metadata
            if layer.use_sliding and cache_metadata is not None and self.sliding_window is not None:
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
        """Filter out rotary embedding inverse frequency keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Exaone4Config, model_type="exaone4")
class Exaone4ForCausalLM(BaseCausalLMModule[Exaone4Model, Exaone4Config]):
    """Exaone4 causal language model.

    Example::

        >>> model = Exaone4ForCausalLM(Exaone4Config())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = Exaone4Config

    def __init__(self, config: Exaone4Config):
        """Initialize the Exaone4 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Exaone4Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Exaone4ForCausalLM", "Exaone4Model")
