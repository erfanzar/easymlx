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

"""OLMo2 MLX implementation for serving and inference.

OLMo2 is Llama-like but with Q/K RMSNorm and a distinct norm pattern:
- post_attention_layernorm applied to attention output BEFORE residual add
- post_feedforward_layernorm applied to MLP output BEFORE residual add
- No pre-MLP norm (MLP takes h directly)
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

from .olmo2_configuration import Olmo2Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert values to int32 mx.array, or return None.

    Args:
        values (mx.ArrayLike | None): Input values to convert.

    Returns:
        mx.array | None: Int32 array or None if input is None.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Olmo2Attention(nn.Module):
    """OLMo2 attention with Q/K RMSNorm and RoPE.

    Applies per-head RMSNorm to queries and keys before RoPE, unlike
    OLMo1 which has no Q/K normalization.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.

    Example:
        >>> config = Olmo2Config()
        >>> attn = Olmo2Attention(config)
        >>> out = attn(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: Olmo2Config):
        """Initialize OLMo2 attention.

        Args:
            config (Olmo2Config): Model configuration.

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

        # Q/K RMSNorm applied before RoPE.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
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
        """Compute attention with Q/K RMSNorm applied per head before RoPE.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        # Apply Q/K RMSNorm per head.
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


class Olmo2MLP(nn.Module):
    """OLMo2 SwiGLU feed-forward network.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gating projection.
        up_proj: Up-projection.
        down_proj: Down-projection.

    Example:
        >>> config = Olmo2Config()
        >>> mlp = Olmo2MLP(config)
        >>> out = mlp(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: Olmo2Config):
        """Initialize OLMo2 SwiGLU MLP.

        Args:
            config (Olmo2Config): Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Olmo2DecoderLayer(nn.Module):
    """Single OLMo2 decoder layer with post-norm architecture.

    Uses a distinct norm pattern from Llama -- norms are applied to the
    attention/MLP output BEFORE the residual add::

        r = post_attention_layernorm(self_attn(input_layernorm(x)))
        h = x + r
        r = post_feedforward_layernorm(mlp(h))
        out = h + r

    Note: the MLP takes ``h`` directly (no pre-MLP norm).

    Attributes:
        self_attn: OLMo2 attention with Q/K RMSNorm.
        mlp: SwiGLU MLP module.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Post-attention RMSNorm (before residual).
        post_feedforward_layernorm: Post-MLP RMSNorm (before residual).

    Example:
        >>> config = Olmo2Config()
        >>> layer = Olmo2DecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: Olmo2Config):
        """Initialize OLMo2 decoder layer.

        Args:
            config (Olmo2Config): Model configuration.
        """
        super().__init__()
        self.self_attn = Olmo2Attention(config)
        self.mlp = Olmo2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        """Forward pass with post-norm architecture.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        # Post-attention norm applied to attn output BEFORE residual add.
        r = self.post_attention_layernorm(
            self.self_attn(
                self.input_layernorm(hidden_states),
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        )
        h = hidden_states + r

        # Post-feedforward norm applied to MLP output BEFORE residual add.
        # No pre-MLP norm: MLP takes h directly.
        r = self.post_feedforward_layernorm(self.mlp(h))
        out = h + r
        return out


@register_module(task_type=TaskType.BASE_MODULE, config=Olmo2Config, model_type="olmo2")
class Olmo2Model(EasyMLXBaseModule):
    """Base OLMo2 transformer model with Q/K RMSNorm and post-norm.

    Attributes:
        config_class: Associated configuration class (``Olmo2Config``).
        embed_tokens: Token embedding layer.
        layers: List of OLMo2 decoder layers.
        norm: Final RMSNorm.

    Example:
        >>> config = Olmo2Config()
        >>> model = Olmo2Model(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = Olmo2Config

    def __init__(self, config: Olmo2Config):
        """Initialize OLMo2 base model.

        Args:
            config (Olmo2Config): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Olmo2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the OLMo2 base model.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings.
            cache_views (list[CacheView] | None): Per-layer KV cache views.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Normalized hidden states of shape ``(B, L, D)``.

        Raises:
            ValueError: If ``cache_views`` length does not match layer count.
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
        """Remove rotary embedding buffers from upstream checkpoints.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Filtered weight dictionary.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Olmo2Config, model_type="olmo2")
class Olmo2ForCausalLM(BaseCausalLMModule[Olmo2Model, Olmo2Config]):
    """OLMo2 causal language model with LM head.

    Embeddings are tied by default (``tie_word_embeddings=True``).

    Attributes:
        config_class: Associated configuration class (``Olmo2Config``).

    Example:
        >>> config = Olmo2Config()
        >>> model = Olmo2ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = Olmo2Config

    def __init__(self, config: Olmo2Config):
        """Initialize OLMo2 causal LM.

        Args:
            config (Olmo2Config): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Olmo2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Olmo2ForCausalLM", "Olmo2Model")
