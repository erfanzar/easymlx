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

"""OLMo3 MLX implementation (serving/inference only).

Structure mirrors EasyDeL's olmo3:
  OLMo3Config -> OLMo3Attention -> OLMo3MLP -> OLMo3DecoderLayer
  -> OLMo3Model -> OLMo3ForCausalLM

OLMo3 uses post-norm architecture (norm after attention/MLP outputs),
Q/K RMSNorm, sliding window attention on alternating layers, and SwiGLU.
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

from .olmo3_configuration import OLMo3Config

CacheView = TransformerCacheView | PageCacheView


class OLMo3Attention(nn.Module):
    """OLMo3 attention with Q/K RMSNorm applied before RoPE.

    Q/K normalization is applied to the full projected dimension
    (not per-head). Sliding attention layers use unscaled RoPE,
    while full attention layers use the config's ``rope_scaling``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.

    Example:
        >>> config = OLMo3Config()
        >>> attn = OLMo3Attention(config, layer_idx=0)
        >>> out = attn(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: OLMo3Config, layer_idx: int):
        """Initialize OLMo3 attention.

        Args:
            config (OLMo3Config): Model configuration.
            layer_idx (int): Layer index to determine attention type
                (sliding vs full) from ``config.layer_types``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Q/K RMSNorm
        self.q_norm = nn.RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

        # Use standard RoPE for sliding attention layers, scaled RoPE for full attention
        if config.layer_types[layer_idx] != "full_attention":
            self.rope = get_rope(
                dims=self.head_dim,
                base=config.rope_theta,
                traditional=False,
            )
        else:
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
        """Compute attention with Q/K RMSNorm before reshape.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        lead = hidden_states.shape[:-1]
        # Apply Q/K norm before reshape
        q = self.q_norm(self.q_proj(hidden_states)).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_norm(self.k_proj(hidden_states)).reshape(*lead, self.num_kv_heads, self.head_dim)
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


class OLMo3MLP(nn.Module):
    """SwiGLU feed-forward network for OLMo3.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gating projection.
        up_proj: Up-projection.
        down_proj: Down-projection.

    Example:
        >>> config = OLMo3Config()
        >>> mlp = OLMo3MLP(config)
        >>> out = mlp(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: OLMo3Config):
        """Initialize OLMo3 SwiGLU MLP.

        Args:
            config (OLMo3Config): Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class OLMo3DecoderLayer(nn.Module):
    """Single OLMo3 decoder layer with post-norm architecture.

    Unlike standard pre-norm transformers, OLMo3 applies normalization
    AFTER the attention/MLP output, before adding the residual::

        r = post_attention_layernorm(self_attn(x))
        h = x + r
        r = post_feedforward_layernorm(mlp(h))
        out = h + r

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: OLMo3 attention module.
        mlp: SwiGLU MLP module.
        post_attention_layernorm: Post-attention RMSNorm (before residual).
        post_feedforward_layernorm: Post-MLP RMSNorm (before residual).

    Example:
        >>> config = OLMo3Config()
        >>> layer = OLMo3DecoderLayer(config, layer_idx=0)
        >>> out = layer(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: OLMo3Config, layer_idx: int):
        """Initialize OLMo3 decoder layer.

        Args:
            config (OLMo3Config): Model configuration.
            layer_idx (int): Layer index for determining attention type.
        """
        super().__init__()
        self.use_sliding = config.layer_types[layer_idx] != "full_attention"
        self.self_attn = OLMo3Attention(config, layer_idx=layer_idx)
        self.mlp = OLMo3MLP(config)
        # Post-norm: applied after attention/MLP output
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
        # Post-norm: norm(attn(x)) + x
        r = self.post_attention_layernorm(
            self.self_attn(
                hidden_states,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        )
        hidden_states = hidden_states + r
        r = self.post_feedforward_layernorm(self.mlp(hidden_states))
        hidden_states = hidden_states + r
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=OLMo3Config, model_type="olmo3")
class OLMo3Model(EasyMLXBaseModule):
    """Base OLMo3 transformer model with sliding window attention.

    Alternates between sliding window and full attention layers based on
    ``config.layer_types``. Builds separate masks for sliding and full
    attention layers.

    Attributes:
        config_class: Associated configuration class (``OLMo3Config``).
        embed_tokens: Token embedding layer.
        layers: List of OLMo3 decoder layers.
        norm: Final RMSNorm.
        sliding_window: Sliding window size.
        layer_types: Per-layer attention type list.

    Example:
        >>> config = OLMo3Config()
        >>> model = OLMo3Model(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = OLMo3Config

    def __init__(self, config: OLMo3Config):
        """Initialize OLMo3 base model.

        Args:
            config (OLMo3Config): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [OLMo3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        self.layer_types = config.layer_types

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the OLMo3 base model.

        Builds separate masks for sliding and full attention layers.
        Sliding layers use a windowed mask and adjusted cache metadata.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=OLMo3Config, model_type="olmo3")
class OLMo3ForCausalLM(BaseCausalLMModule[OLMo3Model, OLMo3Config]):
    """OLMo3 causal language model with LM head.

    Attributes:
        config_class: Associated configuration class (``OLMo3Config``).

    Example:
        >>> config = OLMo3Config()
        >>> model = OLMo3ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = OLMo3Config

    def __init__(self, config: OLMo3Config):
        """Initialize OLMo3 causal LM.

        Args:
            config (OLMo3Config): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=OLMo3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("OLMo3ForCausalLM", "OLMo3Model")
