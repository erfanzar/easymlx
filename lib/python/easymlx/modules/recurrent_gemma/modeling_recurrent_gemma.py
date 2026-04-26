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

"""RecurrentGemma (Griffin) MLX model implementation for serving and inference.

Hybrid architecture alternating between local sliding-window attention
and Real-Gated Linear Recurrent Unit (RG-LRU) recurrent blocks.
Embeddings are scaled by sqrt(hidden_size) and logits are soft-capped.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .recurrent_gemma_configuration import RecurrentGemmaConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values or None.

    Returns:
        An ``mx.array`` of dtype ``int32``, or None if input is None.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class RecurrentGemmaRMSNorm(nn.Module):
    """RMSNorm with weight offset of 1.0 (Gemma convention).

    The effective weight is ``1.0 + self.weight``, matching the Gemma
    initialization convention where weights start at zero but the
    normalization acts as identity.

    Attributes:
        weight: Learnable weight parameter initialized to ones.
        eps: Epsilon for numerical stability.

    Example:
        >>> norm = RecurrentGemmaRMSNorm(2560)
        >>> out = norm(hidden_states)
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        """Initialize RecurrentGemma RMSNorm.

        Args:
            dims: Number of features.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMSNorm with weight offset.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def _rnn_scan(x: mx.array, a: mx.array, h0: mx.array | None) -> tuple[mx.array, mx.array]:
    """Linear recurrence: h_t = a_t * h_{t-1} + x_t.

    Args:
        x: Input tensor (B, T, D).
        a: Decay tensor, same shape as x or broadcastable.
        h0: Initial hidden state (B, D) or None.

    Returns:
        Tuple of (outputs (B, T, D), final hidden state (B, D)).
    """
    if x.shape[1] == 1:
        if h0 is None:
            return x, x[:, 0]
        y = a * h0[:, None] + x
        return y, y[:, -1]

    if h0 is not None:
        h_t = h0
    else:
        B, _, D = x.shape
        h_t = mx.zeros((B, D), dtype=x.dtype)

    y = mx.zeros_like(x)
    for t in range(x.shape[1]):
        h_t = a[:, t] * h_t + x[:, t]
        y[:, t] = h_t

    return y, h_t


class RecurrentGemmaConv1d(nn.Module):
    """Depthwise 1D convolution for temporal mixing in recurrent blocks.

    Applies a depthwise convolution with zero-padding on the left side
    to maintain causal ordering.

    Attributes:
        weight: Convolution kernel of shape ``(channels, kernel_size, 1)``.
        bias: Bias parameter of shape ``(channels,)``.

    Example:
        >>> conv = RecurrentGemmaConv1d(2560, kernel_size=4)
        >>> out, cache = conv(x)
    """

    def __init__(self, channels: int, kernel_size: int):
        """Initialize RecurrentGemma Conv1d.

        Args:
            channels: Number of input/output channels (depthwise).
            kernel_size: Temporal kernel size.
        """
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Apply depthwise conv1d with causal zero-padding.

        Args:
            x: Input tensor of shape ``(batch, seq_len, channels)``.

        Returns:
            Tuple of (output of shape ``(batch, seq_len, channels)``,
            cache state of shape ``(batch, kernel_size - 1, channels)``
            for incremental decoding).
        """
        _B, _L, _C = x.shape
        groups, K, _ = self.weight.shape

        x_padded = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        y = mx.conv_general(x_padded, self.weight, groups=groups)
        y = y + self.bias
        return y, x_padded[:, -K + 1 :, :]


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit (RG-LRU) layer.

    Uses sigmoid-gated decay with learned recurrent parameters for
    stable long-range recurrence. The decay rate is computed as
    ``exp(-8 * gate_a * softplus(recurrent_param))``.

    Attributes:
        width: Total width of the recurrent state.
        num_heads: Number of independent recurrent heads.
        head_dim: Per-head dimensionality.
        recurrent_param: Learned parameter controlling decay rate.
        input_gate_weight: Per-head weight for the input gate.
        input_gate_bias: Per-head bias for the input gate.
        recurrent_gate_weight: Per-head weight for the recurrent gate.
        recurrent_gate_bias: Per-head bias for the recurrent gate.

    Example:
        >>> rglru = RGLRU(width=2560, num_heads=10)
        >>> out, state = rglru(x)
    """

    def __init__(self, width: int, num_heads: int):
        """Initialize RG-LRU.

        Args:
            width: Total width of the recurrent state.
            num_heads: Number of independent recurrent heads.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = self.width // self.num_heads

        self.recurrent_param = mx.zeros((self.width,))

        self.input_gate_weight = mx.zeros((self.num_heads, self.head_dim, self.head_dim))
        self.input_gate_bias = mx.zeros((self.num_heads, self.head_dim))

        self.recurrent_gate_weight = mx.zeros((self.num_heads, self.head_dim, self.head_dim))
        self.recurrent_gate_bias = mx.zeros((self.num_heads, self.head_dim))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Apply RG-LRU recurrence.

        Args:
            x: Input tensor of shape ``(batch, seq_len, width)``.

        Returns:
            Tuple of (output of shape ``(batch, seq_len, width)``,
            final recurrent state of shape ``(batch, width)``).
        """
        B, L, _ = x.shape

        def apply_block_linear(h: mx.array, w: mx.array, b: mx.array) -> mx.array:
            h = h.reshape((B, L, self.num_heads, self.head_dim))
            h = (h.swapaxes(1, 2) @ w).swapaxes(1, 2) + b
            return mx.sigmoid(h.flatten(2, 3))

        gate_x = apply_block_linear(x, self.input_gate_weight, self.input_gate_bias)
        gate_a = apply_block_linear(x, self.recurrent_gate_weight, self.recurrent_gate_bias)

        log_a = -8.0 * gate_a * nn.softplus(self.recurrent_param)
        a = mx.exp(log_a)
        a_square = mx.exp(2 * log_a)

        gated_x = x * gate_x
        multiplier = mx.sqrt(1 - a_square)
        multiplier = multiplier.at[:, 0, :].add(1.0 - multiplier[:, 0, :])
        normalized_x = gated_x * multiplier.astype(x.dtype)

        y, last_h = _rnn_scan(x=normalized_x, a=a, h0=None)
        return y, last_h


class RecurrentBlock(nn.Module):
    """Recurrent temporal block combining Conv1D and RG-LRU.

    Applies linear_x -> conv1d -> RG-LRU for the recurrent branch,
    and linear_y -> GELU for the gating branch. The two are multiplied
    element-wise and projected back.

    Attributes:
        width: Input/output width.
        num_heads: Number of recurrent heads.
        lru_width: Width of the recurrent layer.
        conv1d_temporal_width: Temporal kernel size.
        linear_y: Gating branch projection.
        linear_x: Recurrent branch projection.
        linear_out: Output projection.
        conv_1d: Causal depthwise conv1d.
        rg_lru: RG-LRU recurrence module.

    Example:
        >>> block = RecurrentBlock(2560, 10, conv1d_temporal_width=4)
        >>> out = block(x)
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
    ):
        """Initialize recurrent temporal block.

        Args:
            width: Input/output dimensionality.
            num_heads: Number of recurrent heads.
            lru_width: Width of the RG-LRU. Defaults to ``width``.
            conv1d_temporal_width: Temporal conv1d kernel size.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width

        self.linear_y = nn.Linear(width, self.lru_width)
        self.linear_x = nn.Linear(width, self.lru_width)
        self.linear_out = nn.Linear(self.lru_width, width)
        self.conv_1d = RecurrentGemmaConv1d(
            channels=self.lru_width,
            kernel_size=self.conv1d_temporal_width,
        )
        self.rg_lru = RGLRU(
            width=self.lru_width,
            num_heads=self.num_heads,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply recurrent temporal block.

        Args:
            x: Input tensor of shape ``(batch, seq_len, width)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, width)``.
        """
        y = self.linear_y(x)
        y = nn.gelu_approx(y)

        x_branch = self.linear_x(x)
        x_branch, _conv_cache = self.conv_1d(x_branch)
        x_branch, _rnn_state = self.rg_lru(x_branch)

        x_branch = x_branch * y
        return self.linear_out(x_branch)


class LocalAttentionBlock(nn.Module):
    """Local sliding-window attention block for RecurrentGemma.

    Uses multi-query attention (K and V have a single head) with partial
    RoPE (half of head_dim).

    Attributes:
        width: Input/output dimensionality.
        num_heads: Number of query attention heads.
        head_dim: Per-head dimensionality.
        scale: Attention logit scaling factor.
        q_proj: Query projection.
        k_proj: Key projection (single head).
        v_proj: Value projection (single head).
        o_proj: Output projection.
        rope: Partial rotary positional embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> attn = LocalAttentionBlock(config)
        >>> out = attn(x, mask=mask)
    """

    def __init__(self, config: RecurrentGemmaConfig):
        """Initialize local attention block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.width, self.width, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.width, self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.width, self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.width, self.width, bias=True)

        self.rope = get_rope(
            dims=self.head_dim // 2,
            base=config.rope_theta,
            traditional=False,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale,
            attn_mechanism=getattr(config, "attn_mechanism", None),
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute local sliding-window attention.

        Args:
            x: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = x.shape[:-1]
        queries = self.q_proj(x).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(*lead, 1, self.head_dim)
        values = self.v_proj(x).reshape(*lead, 1, self.head_dim)

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


class RecurrentGemmaMLP(nn.Module):
    """Feed-forward block with GELU-approximate gating.

    Uses a gated architecture: ``down(gelu_approx(gate) * up)``.

    Attributes:
        up_proj: Up projection.
        gate_proj: Gate projection.
        down_proj: Down projection.

    Example:
        >>> mlp = RecurrentGemmaMLP(2560, 7680)
        >>> out = mlp(x)
    """

    def __init__(self, width: int, expanded_width: int):
        """Initialize RecurrentGemma MLP.

        Args:
            width: Input/output dimensionality.
            expanded_width: Total intermediate size (halved for gate/up).
        """
        super().__init__()
        self.up_proj = nn.Linear(width, expanded_width // 2)
        self.gate_proj = nn.Linear(width, expanded_width // 2)
        self.down_proj = nn.Linear(expanded_width // 2, width)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply gated GELU feed-forward.

        Args:
            x: Input tensor of shape ``(*lead, width)``.

        Returns:
            Output tensor of shape ``(*lead, width)``.
        """
        gate = self.gate_proj(x)
        x = self.up_proj(x)
        return self.down_proj(nn.gelu_approx(gate) * x)


class RecurrentGemmaResidualBlock(nn.Module):
    """Residual block wrapping either attention or recurrent temporal block.

    Each block contains a temporal sub-layer (attention or recurrent) and
    a channel (MLP) sub-layer, both with pre-norm and residual connections.

    Attributes:
        temporal_block_type: Either ``"attention"`` or ``"recurrent"``.
        temporal_pre_norm: Pre-norm for the temporal sub-layer.
        temporal_block: The temporal sub-layer module.
        channel_pre_norm: Pre-norm for the MLP sub-layer.
        mlp_block: The MLP sub-layer.

    Example:
        >>> block = RecurrentGemmaResidualBlock(config, "attention")
        >>> out = block(x, mask=mask)
    """

    def __init__(self, config: RecurrentGemmaConfig, block_type: str):
        """Initialize RecurrentGemma residual block.

        Args:
            config: Model configuration.
            block_type: Either ``"attention"`` or ``"recurrent"``.
        """
        super().__init__()
        self.temporal_block_type = block_type

        self.temporal_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if block_type == "recurrent":
            self.temporal_block = RecurrentBlock(
                width=config.hidden_size,
                num_heads=config.num_attention_heads,
                lru_width=config.lru_width,
                conv1d_temporal_width=config.conv1d_width,
            )
        else:
            self.temporal_block = LocalAttentionBlock(config)

        self.channel_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = RecurrentGemmaMLP(
            width=config.hidden_size,
            expanded_width=config.intermediate_size,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the residual block with temporal + channel sub-layers.

        For recurrent blocks, mask/cache arguments are ignored. For
        attention blocks, they are forwarded to the attention module.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask (only used for attention blocks).
            cache_view: KV cache view (only used for attention blocks).
            cache_metadata: Paged-attention metadata (only for attention).

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        normed = self.temporal_pre_norm(x)

        if self.temporal_block_type == "recurrent":
            h = self.temporal_block(normed)
        else:
            h = self.temporal_block(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )

        residual = h + x
        out = self.channel_pre_norm(residual)
        out = self.mlp_block(out) + residual
        return out


@register_module(
    task_type=TaskType.BASE_MODULE,
    config=RecurrentGemmaConfig,
    model_type="recurrent_gemma",
)
class RecurrentGemmaModel(EasyMLXBaseModule):
    """Base RecurrentGemma (Griffin) model (no LM head).

    Attention blocks use standard KV caches; recurrent blocks use
    internal RG-LRU state. Embeddings are scaled by sqrt(hidden_size).
    """

    config_class = RecurrentGemmaConfig

    def __init__(self, config: RecurrentGemmaConfig):
        """Initialize the base RecurrentGemma model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.scale_by_sqrt_dim = config.embeddings_scale_by_sqrt_dim

        self.layers = [
            RecurrentGemmaResidualBlock(
                config,
                config.block_types[i % len(config.block_types)],
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._attn_layer_indices = [i for i, layer in enumerate(self.layers) if layer.temporal_block_type == "attention"]

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through all residual blocks.

        Cache views are routed only to attention layers; recurrent layers
        manage their own internal state. Embeddings are optionally scaled
        by ``sqrt(hidden_size)``.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: KV cache views for attention layers only.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final RMSNorm.
        """
        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        if self.scale_by_sqrt_dim:
            hidden_states = hidden_states * math.sqrt(hidden_states.shape[-1])

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(
                    attention_mask_arr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            is_attn = layer.temporal_block_type == "attention"
            hidden_states = layer(
                hidden_states,
                mask=mask if is_attn else None,
                cache_view=layer_cache,
                cache_metadata=cache_metadata if is_attn else None,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose conv1d weights from upstream format.

        Converts conv1d weights from ``(channels, 1, kernel)`` to
        ``(channels, kernel, 1)`` layout expected by MLX.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Sanitized weight dictionary with transposed conv1d weights.
        """
        sanitized = {}
        for k, v in weights.items():
            if "conv_1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                sanitized[k] = v.moveaxis(2, 1)
            else:
                sanitized[k] = v
        return sanitized


@register_module(
    task_type=TaskType.CAUSAL_LM,
    config=RecurrentGemmaConfig,
    model_type="recurrent_gemma",
)
class RecurrentGemmaForCausalLM(BaseCausalLMModule[RecurrentGemmaModel, RecurrentGemmaConfig]):
    """RecurrentGemma causal language model with logit soft capping."""

    config_class = RecurrentGemmaConfig

    def __init__(self, config: RecurrentGemmaConfig):
        """Initialize the RecurrentGemma causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=RecurrentGemmaModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            logit_cap=config.logits_soft_cap,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights and auto-detect embedding tying.

        Delegates conv1d transposition to the base model. If no separate
        ``lm_head.weight`` is found, enables embedding tying and removes
        the standalone lm_head module.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Sanitized weight dictionary.
        """
        sanitized = self.base_model.sanitize(weights)
        if "lm_head.weight" not in sanitized:
            self._tie_word_embeddings = True
            if hasattr(self, "lm_head"):
                delattr(self, "lm_head")
        return sanitized


__all__ = ("RecurrentGemmaForCausalLM", "RecurrentGemmaModel")
