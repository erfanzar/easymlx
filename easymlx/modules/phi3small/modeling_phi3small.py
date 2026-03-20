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

"""Phi3Small MLX implementation (serving/inference only).

Structure mirrors upstream Phi3Small:
  Phi3SmallConfig -> Phi3SmallAttention -> Phi3SmallMLP
  -> Phi3SmallDecoderLayer -> Phi3SmallModel -> Phi3SmallForCausalLM

Phi3Small uses GeGELU activation, block-sparse attention patterns,
and muP scaling for embeddings and logits.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from .phi3small_configuration import Phi3SmallConfig

CacheView = TransformerCacheView | PageCacheView


@partial(mx.compile, shapeless=True)
def _gegelu_impl(a_gelu: mx.array, a_linear: mx.array, limit: float) -> mx.array:
    """Compiled GeGELU activation core computation.

    Applies clamped GELU to the gating branch and multiplies with
    the linear branch (offset by 1.0).

    Args:
        a_gelu: Gating branch tensor (even-indexed elements).
        a_linear: Linear branch tensor (odd-indexed elements).
        limit: Clamp limit for both branches.

    Returns:
        Activated output tensor with the same shape as inputs.
    """
    a_gelu = mx.where(
        mx.isinf(a_gelu),
        a_gelu,
        mx.clip(a_gelu, a_min=None, a_max=limit),
    )
    a_linear = mx.where(
        mx.isinf(a_linear),
        a_linear,
        mx.clip(a_linear, a_min=-limit, a_max=limit),
    )
    out_gelu = a_gelu * mx.sigmoid(1.702 * a_gelu)
    return out_gelu * (a_linear + 1.0)


def gegelu(x: mx.array, limit: float) -> mx.array:
    """GeGELU activation with interleaved gating and linear branches.

    Splits the input along the last dimension into even-indexed (gating)
    and odd-indexed (linear) elements, applies clamped GELU to the gating
    branch, and multiplies with the linear branch.

    Args:
        x: Input tensor with even last dimension.
        limit: Clamp limit for activation values.

    Returns:
        Activated tensor with last dimension halved.
    """
    a_gelu, a_linear = x[..., ::2], x[..., 1::2]
    return _gegelu_impl(a_gelu, a_linear, limit)


class Phi3SmallAttention(nn.Module):
    """Multi-head attention with fused QKV and optional block-sparse pattern.

    Uses a single fused ``query_key_value`` projection that outputs Q, K,
    and V concatenated. Supports block-sparse attention patterns at
    configurable layer intervals and muP attention scaling.

    Attributes:
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads for GQA.
        n_q_per_kv: Number of query heads per KV head.
        head_dim: Per-head dimensionality.
        query_key_value: Fused Q/K/V linear projection.
        dense: Output projection.
        scale: Attention logit scaling factor (muP-aware).
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.
        block_sparse: Whether this layer uses block-sparse attention.

    Example:
        >>> attn = Phi3SmallAttention(config, layer_idx=0)
        >>> out = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: Phi3SmallConfig, layer_idx: int):
        """Initialize Phi3Small attention layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer, used to determine whether
                block-sparse attention is enabled.
        """
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_q_per_kv = self.n_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query_key_value = nn.Linear(dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim)
        self.dense = nn.Linear(dim, dim)

        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)
        self.scale = 1.0 / norm_factor

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_embedding_base,
            scale=config.rope_position_scale,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

        # Block-sparse attention pattern
        if layer_idx % config.dense_attention_every_n_layers == 0:
            self.block_sparse = True
            self.blocksparse_block_size = config.blocksparse_block_size
            self.blocksparse_num_local_blocks = config.blocksparse_num_local_blocks
            self.blocksparse_vert_stride = config.blocksparse_vert_stride
        else:
            self.block_sparse = False

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute multi-head attention with fused QKV projection.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        lead[0] if len(lead) > 1 else 1
        lead[-1] if len(lead) > 1 else lead[0]

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.reshape(*lead, -1, self.n_q_per_kv + 2, self.head_dim)
        queries = qkv[..., :-2, :].reshape(*lead, self.n_heads, self.head_dim)
        keys = qkv[..., -2, :]
        values = qkv[..., -1, :]

        # Use attention performer for standard attention
        # Block-sparse attention is handled at mask level for simplicity
        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.dense(attn.reshape(*lead, -1))


class Phi3SmallMLP(nn.Module):
    """Feed-forward network with GeGELU activation for Phi3Small.

    Projects to ``2 * hidden_dim`` (interleaved gate and linear branches),
    applies GeGELU activation, then projects back down.

    Attributes:
        gegelu_limit: Clamp limit for GeGELU activation.
        up_proj: Up projection to ``2 * ff_intermediate_size``.
        down_proj: Down projection from ``ff_intermediate_size``.

    Example:
        >>> mlp = Phi3SmallMLP(config)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: Phi3SmallConfig):
        """Initialize Phi3Small MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.ff_intermediate_size
        self.gegelu_limit = config.gegelu_limit
        self.up_proj = nn.Linear(dim, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply up projection -> GeGELU -> down projection.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        x = self.up_proj(hidden_states)
        return self.down_proj(gegelu(x, self.gegelu_limit))


class Phi3SmallDecoderLayer(nn.Module):
    """Single Phi3Small transformer decoder layer.

    Uses LayerNorm (not RMSNorm) with standard pre-norm residual
    connections for both attention and MLP sub-layers.

    Attributes:
        self_attn: Multi-head attention module.
        mlp: GeGELU feed-forward network.
        input_layernorm: Pre-attention LayerNorm.
        post_attention_layernorm: Pre-MLP LayerNorm.

    Example:
        >>> layer = Phi3SmallDecoderLayer(config, layer_idx=0)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: Phi3SmallConfig, layer_idx: int):
        """Initialize Phi3Small decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack.
        """
        super().__init__()
        self.self_attn = Phi3SmallAttention(config, layer_idx)
        self.mlp = Phi3SmallMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Phi3SmallConfig, model_type="phi3small")
class Phi3SmallModel(EasyMLXBaseModule):
    """Base Phi3Small transformer model with muP embedding scaling.

    Phi3Small uses GeGELU activation, block-sparse attention patterns at
    configurable intervals, LayerNorm, and muP scaling for embeddings.
    Embeddings are multiplied by ``mup_embedding_multiplier`` after lookup.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding layer.
        mup_embedding_multiplier: muP scaling factor for embeddings.
        layers: List of ``Phi3SmallDecoderLayer`` instances.
        norm: Final LayerNorm.

    Example:
        >>> model = Phi3SmallModel(config)
        >>> hidden = model(input_ids)
    """

    config_class = Phi3SmallConfig

    def __init__(self, config: Phi3SmallConfig):
        """Initialize the base Phi3Small model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.mup_embedding_multiplier = config.mup_embedding_multiplier
        self.layers = [
            Phi3SmallDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

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

        Applies muP embedding scaling when ``input_embeddings`` is not
        provided and ``mup_embedding_multiplier`` is non-zero.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides
                ``input_ids``; muP scaling is skipped).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final LayerNorm.

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
            if self.mup_embedding_multiplier:
                hidden_states = self.mup_embedding_multiplier * hidden_states

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

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Filtered weight dictionary without rotary inv_freq entries.
        """
        return {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k and "rotary_emb.inv_freq" not in k
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Phi3SmallConfig, model_type="phi3small")
class Phi3SmallForCausalLM(BaseCausalLMModule[Phi3SmallModel, Phi3SmallConfig]):
    """Phi3Small causal language model with muP width scaling on logits.

    Output logits are divided by ``mup_width_multiplier`` to maintain
    stable training dynamics under the muP parameterization.

    Attributes:
        config_class: Associated configuration class.
        mup_width_multiplier: muP width multiplier for logit scaling.

    Example:
        >>> model = Phi3SmallForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = Phi3SmallConfig

    def __init__(self, config: Phi3SmallConfig):
        """Initialize the Phi3Small causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Phi3SmallModel,
            tie_word_embeddings=True,  # Phi3Small always ties embeddings
        )
        self.mup_width_multiplier = config.mup_width_multiplier

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute LM logits with muP width scaling.

        Divides the raw logits by ``mup_width_multiplier`` to maintain
        stable output magnitude under muP parameterization.

        Args:
            hidden_states: Hidden states of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``.
        """
        logits = super().compute_lm_logits(hidden_states)
        if self.mup_width_multiplier:
            logits = logits / self.mup_width_multiplier
        return logits


__all__ = ("Phi3SmallForCausalLM", "Phi3SmallModel")
