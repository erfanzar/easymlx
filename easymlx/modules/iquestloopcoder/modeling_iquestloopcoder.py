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

"""IQuestLoopCoder MLX model implementation for serving and inference.

IQuestLoopCoder features a two-pass loop architecture:
  - Pass 1 (global): standard transformer pass through all layers, saving KV
  - Pass 2 (mixed): each layer reuses loop-1 KV for global attention,
    computes new local (windowed) KV, and mixes via a learned sigmoid gate.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .iquestloopcoder_configuration import IQuestLoopCoderConfig

CacheView = TransformerCacheView | PageCacheView


# ---------------------------------------------------------------------------
# Compiled helpers
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """Compute SwiGLU activation: ``silu(gate) * up``.

    Args:
        gate: Gate tensor.
        up: Up-projected tensor.

    Returns:
        Element-wise product of ``silu(gate)`` and ``up``.
    """
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def _compute_gate(query: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Compute per-head sigmoid gate from query and learned weight/bias.

    Args:
        query: Query tensor of shape ``(B, H, L, D)``.
        weight: Per-head weight of shape ``(H, D)``.
        bias: Per-head bias of shape ``(H,)``.

    Returns:
        Sigmoid gate of shape ``(B, H, L, 1)``.
    """
    gate_logits = query @ weight[:, None, :].swapaxes(-1, -2)
    gate_logits = gate_logits + bias[..., None, None]
    return mx.sigmoid(gate_logits)


@partial(mx.compile, shapeless=True)
def _mix_attention(gate: mx.array, attn_global: mx.array, attn_local: mx.array) -> mx.array:
    """Mix global and local attention outputs via a learned sigmoid gate.

    Args:
        gate: Sigmoid gate of shape ``(B, H, L, 1)``.
        attn_global: Global attention output of shape ``(B, H, L, D)``.
        attn_local: Local (windowed) attention output of shape ``(B, H, L, D)``.

    Returns:
        Mixed attention output of shape ``(B, H, L, D)``.
    """
    return gate * attn_global + (1 - gate) * attn_local


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    mask: mx.array | None = None,
    scale: float,
) -> mx.array:
    """Scaled dot-product attention on ``[B, H, L, D]`` tensors.

    Args:
        queries: Query tensor of shape ``(B, H, L, D)``.
        keys: Key tensor of shape ``(B, H, S, D)``.
        values: Value tensor of shape ``(B, H, S, D)``.
        mask: Optional attention mask.
        scale: Attention scaling factor.

    Returns:
        Attention output of shape ``(B, H, L, D)``.
    """
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)


# ---------------------------------------------------------------------------
# Gate projection for loop mixing
# ---------------------------------------------------------------------------


class LoopGateProjection(nn.Module):
    """Per-head sigmoid gate for mixing global and local attention.

    Projects each query head to a scalar gate value via a learned
    weight/bias, then applies sigmoid. The gate controls the
    interpolation between global and local attention in pass 2.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Per-head dimensionality.
        weight: Learnable per-head weight of shape ``(num_heads, head_dim)``.
        bias: Learnable per-head bias of shape ``(num_heads,)``.

    Example:
        >>> gate_proj = LoopGateProjection(num_heads=8, head_dim=64)
        >>> q = mx.zeros((1, 8, 10, 64))
        >>> gate = gate_proj(q)
        >>> gate.shape
        [1, 8, 10, 1]
    """

    def __init__(self, num_heads: int, head_dim: int):
        """Initialize LoopGateProjection.

        Args:
            num_heads: Number of attention heads.
            head_dim: Per-head dimensionality.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads,))

    def __call__(self, query: mx.array) -> mx.array:
        """Compute sigmoid gate from query tensor.

        Args:
            query: Query tensor of shape ``(B, H, L, D)``.

        Returns:
            Sigmoid gate of shape ``(B, H, L, 1)``.
        """
        return _compute_gate(query, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class IQuestLoopCoderAttention(nn.Module):
    """IQuestLoopCoder attention with RoPE.

    Supports both a standard forward pass (used by ``AttentionPerformer``
    with cache) and a raw ``get_qkv`` method that returns pre-rotated
    Q/K/V in ``[B, H, L, D]`` layout for use in the two-pass loop.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = IQuestLoopCoderConfig(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> attn = IQuestLoopCoderAttention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: IQuestLoopCoderConfig):
        """Initialize IQuestLoopCoderAttention.

        Args:
            config: IQuestLoopCoder model configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
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
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

    def get_qkv(
        self,
        hidden_states: mx.array,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Project and apply RoPE, returning (Q, K, V) in ``[B, H, L, D]`` layout.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            offset: Positional offset for RoPE.

        Returns:
            Tuple of ``(queries, keys, values)`` each of shape
            ``(B, H, L, head_dim)``.
        """
        B, L = hidden_states.shape[:2]
        queries = self.q_proj(hidden_states).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(hidden_states).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(hidden_states).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        return queries, keys, values

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Standard forward pass with RoPE applied by the attention performer."""
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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class IQuestLoopCoderMLP(nn.Module):
    """SwiGLU MLP for IQuestLoopCoder.

    Attributes:
        gate_proj: Gate projection.
        down_proj: Down projection.
        up_proj: Up projection.

    Example:
        >>> config = IQuestLoopCoderConfig(hidden_size=256, intermediate_size=512)
        >>> mlp = IQuestLoopCoderMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: IQuestLoopCoderConfig):
        """Initialize IQuestLoopCoderMLP.

        Args:
            config: IQuestLoopCoder model configuration.
        """
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=config.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=config.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(_silu_mul(self.gate_proj(hidden_states), self.up_proj(hidden_states)))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class IQuestLoopCoderDecoderLayer(nn.Module):
    """Single IQuestLoopCoder decoder layer (used in both loop passes).

    Contains attention, MLP, and layer norms but no ``__call__`` method.
    Forward logic is handled by ``IQuestLoopCoderModel`` which orchestrates
    the two-pass loop.

    Attributes:
        self_attn: Attention sub-layer.
        mlp: SwiGLU MLP sub-layer.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = IQuestLoopCoderConfig(hidden_size=256, num_attention_heads=8)
        >>> layer = IQuestLoopCoderDecoderLayer(config)
    """

    def __init__(self, config: IQuestLoopCoderConfig):
        """Initialize IQuestLoopCoderDecoderLayer.

        Args:
            config: IQuestLoopCoder model configuration.
        """
        super().__init__()
        self.self_attn = IQuestLoopCoderAttention(config)
        self.mlp = IQuestLoopCoderMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.BASE_MODULE, config=IQuestLoopCoderConfig, model_type="iquestloopcoder")
class IQuestLoopCoderModel(EasyMLXBaseModule):
    """Base IQuestLoopCoder transformer model with two-pass loop architecture.

    Implements a two-pass loop over the same decoder layers:
      - Pass 1 (global): Standard transformer pass, collecting KV pairs.
      - Pass 2 (mixed): Reuses pass-1 KV for global attention, computes
        new local (windowed) KV, and mixes via per-head sigmoid gates.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding layer.
        layers: List of ``IQuestLoopCoderDecoderLayer`` instances.
        norm: Final RMSNorm.
        gate_projections: Per-layer sigmoid gate projections.
        loop_num: Number of loop passes (always 2).
        loop_window_size: Sliding window size for local attention.

    Example:
        >>> config = IQuestLoopCoderConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4)
        >>> model = IQuestLoopCoderModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = IQuestLoopCoderConfig

    def __init__(self, config: IQuestLoopCoderConfig):
        """Initialize IQuestLoopCoderModel.

        Args:
            config: IQuestLoopCoder model configuration.

        Raises:
            AssertionError: If ``config.loop_num`` is not 2.
        """
        super().__init__(config)
        assert config.loop_num == 2, f"Only loop_num=2 is supported, got {config.loop_num}"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [IQuestLoopCoderDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_projections = [
            LoopGateProjection(config.num_attention_heads, config.head_dim) for _ in range(config.num_hidden_layers)
        ]
        self.loop_num = config.loop_num
        self.loop_window_size = config.loop_window_size

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the two-pass loop forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Not used in the loop architecture (reserved for API
                compatibility).
            cache_metadata: Paged-attention metadata (reserved).

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        if input_embeddings is not None:
            h = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            h = self.embed_tokens(input_ids)

        B, L = h.shape[:2]

        # Build masks
        mask: mx.array | None = None
        window_mask: mx.array | None = None
        attention_mask_arr = _as_int_array(attention_mask)
        if h.ndim == 3:
            mask = build_attention_mask(attention_mask_arr, batch_size=B, seq_len=L)
            window_mask = build_attention_mask(
                attention_mask_arr,
                batch_size=B,
                seq_len=L,
                window_size=self.loop_window_size,
            )

        # Pass 1: Global attention, collect KV pairs
        loop1_kv: list[tuple[mx.array, mx.array]] = []
        for layer in self.layers:
            h_norm = layer.input_layernorm(h)
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset=0)
            loop1_kv.append((k1, v1))

            out = _sdpa(q1, k1, v1, mask=mask, scale=layer.self_attn.scale)
            r = layer.self_attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        # Pass 2: Mixed global/local attention
        for layer, gate_proj, (k1, v1) in zip(self.layers, self.gate_projections, loop1_kv, strict=False):
            h_norm = layer.input_layernorm(h)
            q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset=0)
            gate = gate_proj(q2)

            attn_global = _sdpa(q2, k1, v1, mask=mask, scale=layer.self_attn.scale)
            attn_local = _sdpa(q2, k2, v2, mask=window_mask, scale=layer.self_attn.scale)

            mixed = _mix_attention(gate, attn_global, attn_local)
            r = layer.self_attn.o_proj(mixed.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        return self.norm(h)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove rotary embedding inverse-frequency buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Filtered weight dictionary.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.CAUSAL_LM, config=IQuestLoopCoderConfig, model_type="iquestloopcoder")
class IQuestLoopCoderForCausalLM(BaseCausalLMModule[IQuestLoopCoderModel, IQuestLoopCoderConfig]):
    """IQuestLoopCoder transformer with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = IQuestLoopCoderConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4)
        >>> model = IQuestLoopCoderForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = IQuestLoopCoderConfig

    def __init__(self, config: IQuestLoopCoderConfig):
        """Initialize IQuestLoopCoderForCausalLM.

        Args:
            config: IQuestLoopCoder model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=IQuestLoopCoderModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("IQuestLoopCoderForCausalLM", "IQuestLoopCoderModel")
