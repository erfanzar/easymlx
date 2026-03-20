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

"""OLMo MLX implementation for serving and inference.

OLMo uses RoPE, SwiGLU MLP, and LayerNorm without affine parameters
(no learnable scale/bias). No bias in attention or MLP layers.
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

from .olmo_configuration import OlmoConfig

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


class OlmoAttention(nn.Module):
    """OLMo attention with RoPE and no bias.

    Uses multi-head attention with all heads sharing the same dimension.
    No bias in any projection layers.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.

    Example:
        >>> config = OlmoConfig()
        >>> attn = OlmoAttention(config)
        >>> out = attn(mx.zeros((1, 128, 2048)))
    """

    def __init__(self, config: OlmoConfig):
        """Initialize OLMo attention.

        Args:
            config (OlmoConfig): Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.n_heads)
        self.head_dim = int(config.d_model // config.n_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

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
        """Compute attention forward pass.

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
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)

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


class OlmoMLP(nn.Module):
    """OLMo SwiGLU feed-forward network without bias.

    Uses a fused projection ``ff_proj`` that outputs ``mlp_hidden_size``
    dimensions, which are then split in half for the gate and up paths.
    Computes ``ff_out(silu(x2) * x1)`` where ``[x1, x2] = split(ff_proj(x))``.

    Attributes:
        ff_proj: Fused gate+up projection.
        ff_out: Down-projection.

    Example:
        >>> config = OlmoConfig()
        >>> mlp = OlmoMLP(config)
        >>> out = mlp(mx.zeros((1, 128, 2048)))
    """

    def __init__(self, config: OlmoConfig):
        """Initialize OLMo SwiGLU MLP.

        Args:
            config (OlmoConfig): Model configuration.
        """
        super().__init__()
        # SwiGLU: ff_proj outputs mlp_hidden_size which is split in half
        self.ff_proj = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(config.mlp_hidden_size // 2, config.d_model, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU MLP forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(..., d_model)``.

        Returns:
            mx.array: Output of shape ``(..., d_model)``.
        """
        x1, x2 = mx.split(self.ff_proj(hidden_states), 2, axis=-1)
        return self.ff_out(nn.silu(x2) * x1)


class OlmoDecoderLayer(nn.Module):
    """Single OLMo decoder layer with non-affine LayerNorm.

    Uses ``nn.LayerNorm(affine=False)`` which has no learnable scale or
    bias parameters, unlike standard LayerNorm.

    Attributes:
        self_attn: OLMo attention module.
        mlp: SwiGLU MLP module.
        att_norm: Pre-attention non-affine LayerNorm.
        ff_norm: Pre-MLP non-affine LayerNorm.

    Example:
        >>> config = OlmoConfig()
        >>> layer = OlmoDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 2048)))
    """

    def __init__(self, config: OlmoConfig):
        """Initialize OLMo decoder layer.

        Args:
            config (OlmoConfig): Model configuration.
        """
        super().__init__()
        self.self_attn = OlmoAttention(config)
        self.mlp = OlmoMLP(config)
        # OLMo uses LayerNorm without affine parameters (no scale/bias).
        self.att_norm = nn.LayerNorm(config.d_model, affine=False)
        self.ff_norm = nn.LayerNorm(config.d_model, affine=False)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        residual = hidden_states
        hidden_states = self.att_norm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=OlmoConfig, model_type="olmo")
class OlmoModel(EasyMLXBaseModule):
    """Base OLMo transformer model with non-affine LayerNorm and SwiGLU.

    Attributes:
        config_class: Associated configuration class (``OlmoConfig``).
        embed_tokens: Token embedding layer.
        layers: List of OLMo decoder layers.
        norm: Final non-affine LayerNorm.

    Example:
        >>> config = OlmoConfig()
        >>> model = OlmoModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = OlmoConfig

    def __init__(self, config: OlmoConfig):
        """Initialize OLMo base model.

        Args:
            config (OlmoConfig): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.embedding_size, config.d_model)
        self.layers = [OlmoDecoderLayer(config) for _ in range(config.n_layers)]
        self.norm = nn.LayerNorm(config.d_model, affine=False)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the OLMo base model.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=OlmoConfig, model_type="olmo")
class OlmoForCausalLM(BaseCausalLMModule[OlmoModel, OlmoConfig]):
    """OLMo causal language model with LM head.

    Supports optional weight tying via the ``weight_tying`` config flag.

    Attributes:
        config_class: Associated configuration class (``OlmoConfig``).

    Example:
        >>> config = OlmoConfig()
        >>> model = OlmoForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = OlmoConfig

    def __init__(self, config: OlmoConfig):
        """Initialize OLMo causal LM.

        Args:
            config (OlmoConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=OlmoModel,
            tie_word_embeddings=bool(getattr(config, "weight_tying", False)),
        )


__all__ = ("OlmoForCausalLM", "OlmoModel")
