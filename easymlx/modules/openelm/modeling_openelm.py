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

"""OpenELM MLX model implementation for serving and inference.

This module provides the OpenELM architecture on MLX, featuring
per-layer variable head counts, optional Q/K RMSNorm, SwiGLU MLP
with gating, and RoPE positional embeddings.
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

from .openelm_configuration import OpenELMConfig

CacheView = TransformerCacheView | PageCacheView


def make_divisible(
    v: float | int,
    divisor: int | None = 8,
    min_value: float | int | None = None,
) -> int:
    """Ensure value is divisible by divisor (from MobileNet).

    Args:
        v: Input value.
        divisor: Divisor. Defaults to 8.
        min_value: Minimum divisor value. Defaults to divisor.

    Returns:
        Divisible integer value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class OpenELMAttention(nn.Module):
    """Multi-head attention for the OpenELM model.

    OpenELM uses per-layer variable numbers of query and KV heads, allowing
    the model to allocate more heads to later layers. Optionally applies
    RMSNorm to Q and K projections before attention computation.

    Attributes:
        head_dim: Per-head dimensionality.
        n_heads: Number of query heads for this layer.
        n_kv_heads: Number of key/value heads for this layer.
        scale: Attention logit scaling factor (``head_dim ** -0.5``).
        normalize_qk_projections: Whether Q/K RMSNorm is applied.
        qkv_proj: Fused query/key/value linear projection.
        out_proj: Output linear projection.
        q_norm: RMSNorm applied to queries (if enabled).
        k_norm: RMSNorm applied to keys (if enabled).
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example:
        >>> attn = OpenELMAttention(config, layer_idx=0)
        >>> out = attn(hidden_states, mask=mask, cache_view=cache)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int):
        """Initialize OpenELM attention layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer, used to select per-layer head
                counts from ``config.num_query_heads`` and
                ``config.num_kv_heads``.
        """
        super().__init__()
        self.head_dim = int(config.head_dim)
        self.n_heads = int(config.num_query_heads[layer_idx])
        self.n_kv_heads = int(config.num_kv_heads[layer_idx])
        self.scale = self.head_dim**-0.5

        op_size = (self.n_heads + self.n_kv_heads * 2) * self.head_dim
        self.qkv_proj = nn.Linear(config.model_dim, op_size, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, config.model_dim, bias=False)

        self.normalize_qk_projections = config.normalize_qk_projections
        if self.normalize_qk_projections:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_freq_constant,
            traditional=False,
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
        """Compute multi-head attention with optional Q/K RMSNorm and RoPE.

        Args:
            hidden_states: Input tensor of shape ``(*lead, model_dim)``.
            mask: Attention mask. Can be an ``mx.array`` broadcastable to
                ``(batch, n_heads, seq, seq)``, a string sentinel, or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata, if applicable.

        Returns:
            Output tensor of shape ``(*lead, model_dim)``.
        """
        lead = hidden_states.shape[:-1]

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(*lead, self.n_heads + self.n_kv_heads * 2, self.head_dim)

        queries = qkv[..., : self.n_heads, :]
        keys = qkv[..., self.n_heads : self.n_heads + self.n_kv_heads, :]
        values = qkv[..., self.n_heads + self.n_kv_heads :, :]

        if self.normalize_qk_projections:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            rope=self.rope,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.out_proj(attn.reshape(*lead, -1))


class OpenELMMLP(nn.Module):
    """SwiGLU feed-forward network for the OpenELM model.

    The intermediate dimension is computed per-layer using
    ``make_divisible(ffn_multiplier * model_dim, divisor=ffn_dim_divisor)``,
    allowing variable FFN widths across layers.

    Attributes:
        proj_1: Linear projection to ``2 * intermediate_dim`` (gate + up).
        proj_2: Linear projection from ``intermediate_dim`` back to
            ``model_dim``.

    Example:
        >>> mlp = OpenELMMLP(config, layer_idx=0)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int):
        """Initialize OpenELM MLP.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer, used to select the FFN
                multiplier from ``config.ffn_multipliers``.
        """
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor))

        self.proj_1 = nn.Linear(config.model_dim, 2 * intermediate_dim, bias=False)
        self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU feed-forward transformation.

        Args:
            hidden_states: Input tensor of shape ``(*lead, model_dim)``.

        Returns:
            Output tensor of shape ``(*lead, model_dim)``.
        """
        x = self.proj_1(hidden_states)
        gate, x = mx.split(x, 2, axis=-1)
        return self.proj_2(nn.silu(gate) * x)


class OpenELMDecoderLayer(nn.Module):
    """Single OpenELM transformer decoder layer.

    Applies pre-norm RMSNorm before both attention and MLP sub-layers
    with residual connections.

    Attributes:
        attn: Multi-head attention module.
        ffn: SwiGLU feed-forward network.
        attn_norm: Pre-attention RMSNorm.
        ffn_norm: Pre-FFN RMSNorm.

    Example:
        >>> layer = OpenELMDecoderLayer(config, layer_idx=0)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int):
        """Initialize OpenELM decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack.
        """
        super().__init__()
        self.attn = OpenELMAttention(config, layer_idx=layer_idx)
        self.ffn = OpenELMMLP(config, layer_idx=layer_idx)
        self.attn_norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer with pre-norm and residual connections.

        Args:
            hidden_states: Input tensor of shape
                ``(batch, seq_len, model_dim)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, model_dim)``.
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=OpenELMConfig, model_type="openelm")
class OpenELMModel(EasyMLXBaseModule):
    """Base OpenELM transformer model without a language modeling head.

    OpenELM features per-layer variable head counts, optional Q/K RMSNorm,
    and per-layer variable FFN sizes computed via ``make_divisible``.

    Attributes:
        config_class: Associated configuration class (``OpenELMConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``OpenELMDecoderLayer`` instances.
        norm: Final RMSNorm applied after all layers.

    Example:
        >>> model = OpenELMModel(config)
        >>> hidden = model(input_ids)
    """

    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        """Initialize the base OpenELM model.

        Args:
            config: Model configuration specifying architecture parameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = [OpenELMDecoderLayer(config, layer_idx=i) for i in range(config.num_transformer_layers)]
        self.norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)

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
                ``(seq_len,)``. A 1-D input is unsqueezed to add a batch
                dimension when ``cache_metadata`` is None.
            attention_mask: Optional attention mask broadcastable to
                ``(batch, 1, seq_len, seq_len)``.
            input_embeddings: Pre-computed embeddings. When provided,
                ``input_ids`` is ignored.
            cache_views: Per-layer KV cache views for autoregressive
                generation. Must have the same length as ``self.layers``.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, model_dim)`` after
            final RMSNorm.

        Raises:
            ValueError: If ``cache_views`` is provided but its length does
                not match the number of layers.
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
        """Remove precomputed rotary embedding inverse frequencies.

        These buffers are recomputed at runtime and should not be loaded
        from upstream checkpoints.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Filtered weight dictionary without rotary inv_freq entries.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=OpenELMConfig, model_type="openelm")
class OpenELMForCausalLM(BaseCausalLMModule[OpenELMModel, OpenELMConfig]):
    """OpenELM model with a causal language modeling head.

    Wraps ``OpenELMModel`` and adds an LM head for next-token prediction.
    Embedding tying is controlled by ``share_input_output_layers`` in the
    config.

    Attributes:
        config_class: Associated configuration class (``OpenELMConfig``).

    Example:
        >>> model = OpenELMForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        """Initialize the OpenELM causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=OpenELMModel,
            tie_word_embeddings=bool(getattr(config, "share_input_output_layers", True)),
        )


__all__ = ("OpenELMForCausalLM", "OpenELMModel")
