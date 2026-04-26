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

"""Qwen3-Next MLX model implementation for serving and inference.

This module provides the Qwen3-Next hybrid architecture on MLX, combining
full softmax attention with linear attention layers, partial rotary
embeddings, gated normalization, MoE routing with shared experts, and a
causal language model wrapper.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule
from easymlx.operations import OperationMetadata
from easymlx.operations.kernels.gated_delta_rule import GatedDeltaRuleOp

from .qwen3_next_configuration import Qwen3NextConfig

CacheView = TransformerCacheView | PageCacheView


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Look up an activation function by name.

    Args:
        name: Case-insensitive activation name. Supported values are
            ``"silu"``, ``"swish"``, and ``"gelu"``.

    Returns:
        The corresponding MLX activation callable.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


def _resolve_module_activation_dtype(module: nn.Module, fallback: mx.Dtype) -> mx.Dtype:
    """Return a real floating activation dtype for ``module``.

    Quantized MLX linears store packed weights in integer dtypes, so using
    ``module.weight.dtype`` for activations is incorrect. This function
    inspects common floating-point attributes (bias, scales, weight) and
    returns the first valid floating dtype found, falling back to the
    provided runtime dtype.

    Args:
        module: The MLX module to inspect.
        fallback: Fallback dtype if no floating-point parameter is found.

    Returns:
        A floating-point ``mx.Dtype`` suitable for activation computation.
    """

    for attr in ("bias", "biases", "scales", "weight"):
        value = getattr(module, attr, None)
        dtype = getattr(value, "dtype", None)
        if dtype is not None and mx.issubdtype(dtype, mx.floating):
            return dtype
    if mx.issubdtype(fallback, mx.floating):
        return fallback
    return mx.float16


def sanitize_qwen3_next_projection_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Sanitize Qwen3-Next weights for HuggingFace to MLX conversion.

    Performs two transformations:
    1. Transposes ``conv1d.weight`` tensors from HuggingFace layout
       ``(out, in/groups, kernel)`` to MLX layout ``(out, kernel, in/groups)``.
    2. Drops multi-token prediction (``mtp.*``) head weights.

    Args:
        weights: Raw weight dictionary from a HuggingFace checkpoint.

    Returns:
        Sanitized weight dictionary with transposed conv1d weights and
        dropped mtp head weights.
    """
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith("mtp."):
            continue
        if key.endswith("conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
            value = value.transpose(0, 2, 1)
        out[key] = value
    return out


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm with ``(1 + weight)`` scaling.

    Unlike standard RMSNorm which multiplies by ``weight``, this variant
    uses ``(1 + weight)`` to allow zero-initialized weights to produce
    identity behavior.

    Attributes:
        weight: Learnable scale parameter initialized to zeros.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize the RMSNorm layer.

        Args:
            hidden_size: Dimensionality of the input features.
            eps: Epsilon for numerical stability in the variance computation.
        """
        super().__init__()
        self.weight = mx.zeros((hidden_size,))
        self.eps = float(eps)
        self._scale_cache: mx.array | None = None
        self._scale_cache_dtype: mx.Dtype | None = None
        self._scale_cache_weight: mx.array | None = None

    def _runtime_scale(self, dtype: mx.Dtype) -> mx.array:
        if self._scale_cache is None or self._scale_cache_dtype != dtype or self._scale_cache_weight is not self.weight:
            self._scale_cache = self.weight.astype(dtype) + mx.array(1.0, dtype=dtype)
            self._scale_cache_dtype = dtype
            self._scale_cache_weight = self.weight
        return self._scale_cache

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply RMSNorm with ``(1 + weight)`` scaling.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Normalized tensor of the same shape and original dtype.
        """
        return mx.fast.rms_norm(hidden_states, self._runtime_scale(hidden_states.dtype), self.eps)


class Qwen3NextRMSNormGated(nn.Module):
    """Gated RMSNorm for linear attention outputs.

    Applies standard RMSNorm to the hidden states, then multiplies by a
    SiLU-activated gate signal.

    Attributes:
        weight: Learnable scale parameter initialized to ones.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize the gated RMSNorm layer.

        Args:
            hidden_size: Dimensionality of the input features.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = float(eps)
        self._weight_cache: mx.array | None = None
        self._weight_cache_dtype: mx.Dtype | None = None
        self._weight_cache_source: mx.array | None = None

    def _runtime_weight(self, dtype: mx.Dtype) -> mx.array:
        if (
            self._weight_cache is None
            or self._weight_cache_dtype != dtype
            or self._weight_cache_source is not self.weight
        ):
            self._weight_cache = self.weight.astype(dtype)
            self._weight_cache_dtype = dtype
            self._weight_cache_source = self.weight
        return self._weight_cache

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        """Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            gate: Gate tensor of the same shape, activated with SiLU.

        Returns:
            Gated normalized tensor of the same shape and original dtype.
        """
        hidden_states = mx.fast.rms_norm(hidden_states, self._runtime_weight(hidden_states.dtype), self.eps)
        return hidden_states * nn.silu(gate.astype(hidden_states.dtype))


class _PartialRoPE(nn.Module):
    """Apply rotary embeddings only to the leading ``rotary_dim`` dimensions.

    When ``partial_rotary_factor`` is less than 1.0, only the first
    ``rotary_dim`` dimensions of each head receive RoPE; the remaining
    dimensions pass through unchanged.

    Attributes:
        rope: The underlying rotary embedding module.
        rotary_dim: Number of leading dimensions to apply RoPE to.
    """

    def __init__(self, rope: nn.Module, rotary_dim: int):
        """Initialize the partial RoPE wrapper.

        Args:
            rope: A rotary embedding module that operates on tensors of
                shape ``(..., rotary_dim)``.
            rotary_dim: Number of leading dimensions to apply RoPE to.
        """
        super().__init__()
        self.rope = rope
        self.rotary_dim = int(rotary_dim)

    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array:
        """Apply partial rotary embeddings.

        Args:
            x: Input tensor of shape ``(..., head_dim)``.
            offset: Position offset for the rotary embedding computation.

        Returns:
            Tensor of the same shape with RoPE applied to the first
            ``rotary_dim`` dimensions.
        """
        return self.rope(x, offset=offset)


class Qwen3NextFullAttention(nn.Module):
    """Full softmax attention with partial RoPE and gated output for Qwen3-Next.

    Projects queries to double width (Q + gate), applies QK normalization,
    partial rotary embeddings, standard scaled dot-product attention, and
    sigmoid gating on the output.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
        rotary_dim: Number of head dimensions receiving RoPE.
        q_proj: Query projection (2x width for Q + gate).
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: RMSNorm for queries.
        k_norm: RMSNorm for keys.
        rope: Partial rotary positional embedding module.
        attention_performer: Attention computation backend.
    """

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the full attention module.

        Args:
            config: Qwen3-Next configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5
        self.rotary_dim = min(self.head_dim, int(config.rotary_dim))

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = None
        if self.rotary_dim > 0:
            self.rope = _PartialRoPE(
                get_rope(
                    dims=self.rotary_dim,
                    base=config.rope_theta,
                    traditional=False,
                    scaling_config=config.rope_scaling,
                    max_position_embeddings=config.max_position_embeddings,
                ),
                rotary_dim=self.rotary_dim,
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
        """Compute full attention with partial RoPE and sigmoid gating.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Gated attention output.
        """
        lead = hidden_states.shape[:-1]
        q_proj = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim * 2)
        q, gate = mx.split(q_proj, 2, axis=-1)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = self.attention_performer(
            q,
            k,
            v,
            rope=self.rope,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        attn = attn * mx.sigmoid(gate)
        return self.o_proj(attn.reshape(*lead, -1))


def _shift_conv_state_left(conv_state: mx.array, new_value: mx.array) -> mx.array:
    """Shift a rolling convolution buffer left and append a new token.

    Args:
        conv_state: Rolling buffer of shape ``(batch, channels, kernel_size)``.
        new_value: New token values of shape ``(batch, channels)``.

    Returns:
        Updated buffer of the same shape with the oldest entry dropped and
        ``new_value`` appended at the right.
    """
    return mx.concatenate([conv_state[:, :, 1:], new_value[:, :, None]], axis=-1)


def _manual_depthwise_conv(conv_state: mx.array, kernel: mx.array) -> mx.array:
    """Compute a single-step depthwise convolution from a cached state buffer.

    Performs ``silu(sum(state * kernel, axis=-1))`` for efficient decode-time
    convolution without requiring a full Conv1d forward pass.

    Args:
        conv_state: Cached convolution state of shape
            ``(batch, channels, kernel_size)``.
        kernel: Convolution kernel of shape ``(channels, kernel_size)``.

    Returns:
        Convolution output of shape ``(batch, channels)`` after SiLU activation.
    """
    return nn.silu(mx.sum(conv_state * kernel[None, :, :], axis=-1))


def _decode_depthwise_conv_state_update(
    conv_state: mx.array,
    new_value: mx.array,
    kernel: mx.array,
) -> tuple[mx.array, mx.array]:
    """Shift decode convolution state and compute the new depthwise output."""
    new_state = _shift_conv_state_left(conv_state, new_value)
    return _manual_depthwise_conv(new_state, kernel), new_state


def _slot_ids_match_state_rows(slot_ids: tuple[int, ...], state_rows: int) -> bool:
    """Return whether ragged slot ids already address every state row in order."""
    if len(slot_ids) != int(state_rows):
        return False
    return all(slot_id == idx for idx, slot_id in enumerate(slot_ids))


class Qwen3NextLinearAttention(nn.Module):
    """Linear attention module using the Gated Delta Rule (GDR).

    Instead of softmax attention, this module uses a recurrent linear
    attention mechanism with:
    - Causal depthwise 1-D convolution for local context
    - Gated delta rule recurrence for global context
    - Learnable decay (A_log) and time bias (dt_bias)

    Attributes:
        config: Qwen3-Next configuration.
        num_k_heads: Number of key heads.
        num_v_heads: Number of value heads.
        head_k_dim: Key head dimensionality.
        head_v_dim: Value head dimensionality.
        key_dim: Total key dimensionality.
        value_dim: Total value dimensionality.
        conv_dim: Dimensionality for the convolution input.
    """

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the linear attention module.

        Args:
            config: Qwen3-Next configuration.
        """
        super().__init__()
        self.config = config
        self.num_k_heads = int(config.linear_num_key_heads)
        self.num_v_heads = int(config.linear_num_value_heads)
        self.head_k_dim = int(config.linear_key_head_dim)
        self.head_v_dim = int(config.linear_value_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.expand_ratio = self.num_v_heads // self.num_k_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.d_conv = int(getattr(config, "linear_conv_kernel_dim", 4))

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.gdr_op = GatedDeltaRuleOp()
        self._decode_query_scale = float(self.head_k_dim**-0.5)

        conv_cls = getattr(nn, "Conv1d", None)
        self.conv1d = None
        if conv_cls is not None:
            self.conv1d = conv_cls(
                self.conv_dim,
                self.conv_dim,
                kernel_size=self.d_conv,
                groups=self.conv_dim,
                bias=False,
            )
        else:
            self._decode_identity_conv_kernel = mx.ones((self.conv_dim, self.d_conv), dtype=mx.float32)

        self.A_log = mx.zeros((self.num_v_heads,))
        self.dt_bias = mx.ones((self.num_v_heads,))

        self._conv_state: mx.array | None = None
        self._recurrent_state: mx.array | None = None
        self._decode_decay_base: mx.array | None = None
        self._decode_dt_bias: mx.array | None = None
        self._decode_conv_kernel: mx.array | None = None
        self._decode_out_proj_input_dtype: mx.Dtype | None = None

    @staticmethod
    def _compute_decay_base(decay_log: mx.array) -> mx.array:
        """Return the recurrent decay base in fp32 without widening stored weights."""
        decay_base = -mx.exp(decay_log)
        if decay_base.dtype != mx.float32:
            decay_base = decay_base.astype(mx.float32)
        return decay_base

    @staticmethod
    def _compute_log_decay(alpha_raw: mx.array, dt_bias: mx.array, decay_base: mx.array) -> mx.array:
        """Compute log-space decay from native-dtype activations and fp32 decay coefficients."""
        if dt_bias.dtype != alpha_raw.dtype:
            dt_bias = dt_bias.astype(alpha_raw.dtype)
        alpha_biased = alpha_raw + dt_bias
        if alpha_biased.dtype != mx.float32:
            alpha_biased = alpha_biased.astype(mx.float32)
        return decay_base * nn.softplus(alpha_biased)

    def _refresh_decode_cache(self) -> None:
        """Refresh decode-only tensors that stay constant within a generation.

        Pre-computes the derived fp32 decay base while keeping the learnable
        dt bias and convolution kernel in the decode activation dtype so the
        single-step convolution does not upcast cached activations.
        """
        activation_dtype = _resolve_module_activation_dtype(self.in_proj_qkv, mx.float16)
        alpha_dtype = _resolve_module_activation_dtype(self.in_proj_a, activation_dtype)

        self.gdr_op.metadata = OperationMetadata(runtime_dtype=activation_dtype)
        self._decode_decay_base = self._compute_decay_base(self.A_log)
        self._decode_dt_bias = self.dt_bias
        self._decode_out_proj_input_dtype = _resolve_module_activation_dtype(self.out_proj, activation_dtype)
        if self._decode_dt_bias.dtype != alpha_dtype:
            self._decode_dt_bias = self._decode_dt_bias.astype(alpha_dtype)
        if self.conv1d is None:
            self._decode_conv_kernel = self._decode_identity_conv_kernel
            if self._decode_conv_kernel.dtype != activation_dtype:
                self._decode_conv_kernel = self._decode_conv_kernel.astype(activation_dtype)
            return
        kernel = self.conv1d.weight.squeeze(-1)
        if kernel.dtype != activation_dtype:
            kernel = kernel.astype(activation_dtype)
        self._decode_conv_kernel = kernel

    def _ensure_decode_cache(self) -> tuple[mx.array, mx.array, mx.array]:
        """Return cached decode invariants, refreshing them if needed.

        Returns:
            A tuple of ``(decay_base, dt_bias, conv_kernel)`` arrays where
            ``decay_base`` is fp32 and the cached bias/kernel stay in the
            decode activation dtype for single-step convolution.
        """
        if self._decode_decay_base is None or self._decode_dt_bias is None or self._decode_conv_kernel is None:
            self._refresh_decode_cache()
        assert self._decode_decay_base is not None
        assert self._decode_dt_bias is not None
        assert self._decode_conv_kernel is not None
        return self._decode_decay_base, self._decode_dt_bias, self._decode_conv_kernel

    def reset_state(self, batch_size: int = 1):
        """Reset convolution and recurrent states for a new generation.

        Initializes the rolling convolution buffer and the recurrent state
        matrix to zeros, and refreshes the decode cache invariants.

        Args:
            batch_size: Batch size for the new generation session.
        """
        conv_dtype = _resolve_module_activation_dtype(self.in_proj_qkv, mx.float16)
        self._conv_state = mx.zeros(
            (batch_size, self.conv_dim, self.d_conv),
            dtype=conv_dtype,
        )
        self._recurrent_state = mx.zeros(
            (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=conv_dtype,
        )
        self._refresh_decode_cache()

    def _ensure_decode_state_batch_size(self, batch_size: int) -> None:
        """Ensure cached decode state has at least ``batch_size`` rows."""
        batch_size = max(int(batch_size), 1)
        if self._conv_state is None or self._recurrent_state is None:
            self.reset_state(batch_size=batch_size)
            return

        current_batch = int(self._conv_state.shape[0])
        if current_batch >= batch_size:
            self._ensure_decode_cache()
            return

        conv_state = mx.zeros(
            (batch_size, self.conv_dim, self.d_conv),
            dtype=self._conv_state.dtype,
        )
        recurrent_state = mx.zeros(
            (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            dtype=self._recurrent_state.dtype,
        )
        conv_indices = mx.arange(current_batch, dtype=mx.int32).reshape(current_batch, 1, 1)
        recurrent_indices = mx.arange(current_batch, dtype=mx.int32).reshape(current_batch, 1, 1, 1)
        self._conv_state = mx.put_along_axis(conv_state, conv_indices, self._conv_state, axis=0)
        self._recurrent_state = mx.put_along_axis(
            recurrent_state,
            recurrent_indices,
            self._recurrent_state,
            axis=0,
        )
        self._ensure_decode_cache()

    def get_decode_state(self) -> dict[str, mx.array]:
        if self._conv_state is None or self._recurrent_state is None:
            self.reset_state(batch_size=1)
        self._ensure_decode_cache()
        assert self._conv_state is not None
        assert self._recurrent_state is not None
        return {
            "conv_state": self._conv_state,
            "recurrent_state": self._recurrent_state,
        }

    def set_decode_state(self, state: dict[str, mx.array]) -> None:
        self._conv_state = state["conv_state"]
        self._recurrent_state = state["recurrent_state"]

    def decode_step_with_state(
        self,
        hidden_states: mx.array,
        *,
        decode_state: dict[str, mx.array],
        cache_metadata: PageMetadata | None = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        is_paged = hidden_states.ndim == 2
        ragged_slot_ids: tuple[int, ...] | None = None
        full_conv_state: mx.array | None = None
        full_recurrent_state: mx.array | None = None
        conv_state = decode_state["conv_state"]
        recurrent_state = decode_state["recurrent_state"]
        if is_paged and cache_metadata is not None and bool(getattr(cache_metadata, "is_single_token_decode", False)):
            raw_slot_ids = getattr(cache_metadata, "slot_ids", None)
            ragged_slot_ids = (
                tuple(int(slot_id) for slot_id in raw_slot_ids)
                if raw_slot_ids is not None
                else tuple(range(int(hidden_states.shape[0])))
            )
            if ragged_slot_ids:
                if not _slot_ids_match_state_rows(ragged_slot_ids, int(conv_state.shape[0])):
                    slot_indices = mx.array(ragged_slot_ids, dtype=mx.int32)
                    full_conv_state = conv_state
                    full_recurrent_state = recurrent_state
                    conv_state = mx.take(full_conv_state, slot_indices, axis=0)
                    recurrent_state = mx.take(full_recurrent_state, slot_indices, axis=0)
                hidden_states = hidden_states[:, None, :]
            else:
                hidden_states = hidden_states[None]
        elif is_paged:
            hidden_states = hidden_states[None]

        batch_size, seq_len, _ = hidden_states.shape

        decode_decay_base, decode_dt_bias, decode_conv_kernel = self._ensure_decode_cache()

        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        beta_raw = self.in_proj_b(hidden_states)
        alpha_raw = self.in_proj_a(hidden_states)

        if seq_len == 1:
            conv_x = qkv[:, 0, :]
            conv_out_2d, conv_state = _decode_depthwise_conv_state_update(conv_state, conv_x, decode_conv_kernel)
            conv_out = conv_out_2d[:, None, :]
            decay = None
        elif self.conv1d is not None:
            prefix = conv_state[:, :, 1:].transpose(0, 2, 1)
            conv_source = mx.concatenate([prefix, qkv], axis=1)
            conv_out = nn.silu(self.conv1d(conv_source))
            conv_state = conv_source[:, -self.d_conv :, :].transpose(0, 2, 1)
            decay = self._compute_log_decay(alpha_raw, decode_dt_bias, decode_decay_base)
            if conv_out.dtype != mx.float32:
                conv_out = conv_out.astype(mx.float32)
        else:
            conv_out = qkv
            conv_state = conv_state
            decay = self._compute_log_decay(alpha_raw, decode_dt_bias, decode_decay_base)
            if conv_out.dtype != mx.float32:
                conv_out = conv_out.astype(mx.float32)

        query = conv_out[:, :, : self.key_dim].reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = conv_out[:, :, self.key_dim : self.key_dim * 2].reshape(
            batch_size, seq_len, self.num_k_heads, self.head_k_dim
        )
        value = conv_out[:, :, self.key_dim * 2 :].reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        if self.expand_ratio > 1 and not isinstance(self.gdr_op, GatedDeltaRuleOp):
            query = mx.repeat(query, self.expand_ratio, axis=2)
            key = mx.repeat(key, self.expand_ratio, axis=2)

        if seq_len == 1 and hasattr(self.gdr_op, "step_decode_with_alpha"):
            output, recurrent_state = self.gdr_op.step_decode_with_alpha(
                query=query,
                key=key,
                value=value,
                beta=beta_raw,
                alpha=alpha_raw,
                dt_bias=decode_dt_bias,
                decay_base=decode_decay_base,
                recurrent_state=recurrent_state,
                use_qk_l2norm=True,
                query_scale=self._decode_query_scale,
                beta_is_logits=True,
                prefer_metal=True,
            )
        elif seq_len > 1:
            beta = mx.sigmoid(beta_raw)
            gdr_output = self.gdr_op(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=True,
                query_scale=self._decode_query_scale,
                decay_is_log=True,
                prefer_metal=True,
            )
            output = gdr_output.attention_outputs
            recurrent_state = gdr_output.recurrent_state
        else:
            beta = mx.sigmoid(beta_raw)
            output, recurrent_state = self.gdr_op.step_decode(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=True,
                query_scale=self._decode_query_scale,
                decay_is_log=True,
                prefer_metal=seq_len == 1,
            )

        output = self.norm(output, z)
        out_proj_input_dtype = self._decode_out_proj_input_dtype or _resolve_module_activation_dtype(
            self.out_proj,
            hidden_states.dtype,
        )
        if output.dtype != out_proj_input_dtype:
            output = output.astype(out_proj_input_dtype)
        output = self.out_proj(output.reshape(batch_size, seq_len, -1))
        if ragged_slot_ids is not None and full_conv_state is not None and full_recurrent_state is not None:
            conv_indices = mx.array(ragged_slot_ids, dtype=mx.int32).reshape(len(ragged_slot_ids), 1, 1)
            recurrent_indices = mx.array(ragged_slot_ids, dtype=mx.int32).reshape(len(ragged_slot_ids), 1, 1, 1)
            conv_state = mx.put_along_axis(full_conv_state, conv_indices, conv_state, axis=0)
            recurrent_state = mx.put_along_axis(full_recurrent_state, recurrent_indices, recurrent_state, axis=0)
            output = output[:, 0, :]
        elif is_paged:
            output = output[0]
        return output, {
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
        }

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute linear attention via Gated Delta Rule.

        Args:
            hidden_states: Input tensor of shape ``(T, hidden)`` (paged) or
                ``(batch, seq, hidden)``.
            mask: Attention mask (used to zero padding tokens).
            cache_view: Cache view (used for offset tracking only).
            cache_metadata: Paged cache metadata.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``.
        """
        if mask is not None and not isinstance(mask, str):
            if mask.ndim == 4:
                token_mask = mask[:, 0, 0, :]
            else:
                token_mask = mask
            hidden_states = hidden_states * token_mask[:, :, None].astype(hidden_states.dtype)

        is_paged = hidden_states.ndim == 2
        ragged_slot_ids: tuple[int, ...] | None = None
        full_conv_state: mx.array | None = None
        full_recurrent_state: mx.array | None = None
        if is_paged and cache_metadata is not None and bool(getattr(cache_metadata, "is_single_token_decode", False)):
            raw_slot_ids = getattr(cache_metadata, "slot_ids", None)
            ragged_slot_ids = (
                tuple(int(slot_id) for slot_id in raw_slot_ids)
                if raw_slot_ids is not None
                else tuple(range(int(hidden_states.shape[0])))
            )
            if ragged_slot_ids:
                self._ensure_decode_state_batch_size(max(ragged_slot_ids) + 1)
                assert self._conv_state is not None
                assert self._recurrent_state is not None
                if not _slot_ids_match_state_rows(ragged_slot_ids, int(self._conv_state.shape[0])):
                    full_conv_state = self._conv_state
                    full_recurrent_state = self._recurrent_state
                    slot_indices = mx.array(ragged_slot_ids, dtype=mx.int32)
                    self._conv_state = mx.take(full_conv_state, slot_indices, axis=0)
                    self._recurrent_state = mx.take(full_recurrent_state, slot_indices, axis=0)
                hidden_states = hidden_states[:, None, :]
            else:
                hidden_states = hidden_states[None]
        elif is_paged:
            hidden_states = hidden_states[None]

        batch_size, seq_len, _ = hidden_states.shape
        is_decode = seq_len == 1 and self._recurrent_state is not None

        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        beta_raw = self.in_proj_b(hidden_states)
        alpha_raw = self.in_proj_a(hidden_states)

        conv_input = qkv

        if is_decode and self._conv_state is not None:
            decode_decay_base, decode_dt_bias, decode_conv_kernel = self._ensure_decode_cache()
            conv_x = conv_input[:, 0, :]
            conv_out_2d, self._conv_state = _decode_depthwise_conv_state_update(
                self._conv_state,
                conv_x,
                decode_conv_kernel,
            )
            conv_out = conv_out_2d[:, None, :]
        elif self.conv1d is not None:
            decay = self._compute_log_decay(
                alpha_raw,
                self.dt_bias,
                self._compute_decay_base(self.A_log),
            )
            if self._conv_state is not None:
                prefix = self._conv_state[:, :, 1:].transpose(0, 2, 1)
                conv_source = mx.concatenate([prefix, conv_input], axis=1)
            else:
                conv_source = mx.pad(conv_input, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
            conv_out = nn.silu(self.conv1d(conv_source))
            self._conv_state = conv_source[:, -self.d_conv :, :].transpose(0, 2, 1)
        else:
            decay = self._compute_log_decay(
                alpha_raw,
                self.dt_bias,
                self._compute_decay_base(self.A_log),
            )
            conv_out = conv_input

        if not is_decode and conv_out.dtype != mx.float32:
            conv_out = conv_out.astype(mx.float32)

        query = conv_out[:, :, : self.key_dim].reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = conv_out[:, :, self.key_dim : self.key_dim * 2].reshape(
            batch_size, seq_len, self.num_k_heads, self.head_k_dim
        )
        value = conv_out[:, :, self.key_dim * 2 :].reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        if self.expand_ratio > 1 and not isinstance(self.gdr_op, GatedDeltaRuleOp):
            query = mx.repeat(query, self.expand_ratio, axis=2)
            key = mx.repeat(key, self.expand_ratio, axis=2)

        if is_decode:
            if hasattr(self.gdr_op, "step_decode_with_alpha"):
                output, self._recurrent_state = self.gdr_op.step_decode_with_alpha(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta_raw,
                    alpha=alpha_raw,
                    dt_bias=decode_dt_bias,
                    decay_base=decode_decay_base,
                    recurrent_state=self._recurrent_state,
                    use_qk_l2norm=True,
                    query_scale=self._decode_query_scale,
                    beta_is_logits=True,
                    prefer_metal=True,
                )
            else:
                beta = mx.sigmoid(beta_raw)
                decay = self._compute_log_decay(alpha_raw, decode_dt_bias, decode_decay_base)
                output, self._recurrent_state = self.gdr_op.step_decode(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    recurrent_state=self._recurrent_state,
                    use_qk_l2norm=True,
                    query_scale=self._decode_query_scale,
                    decay_is_log=True,
                    prefer_metal=True,
                )
        else:
            beta = mx.sigmoid(beta_raw)
            gdr_output = self.gdr_op(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=self._recurrent_state,
                use_qk_l2norm=True,
                query_scale=self._decode_query_scale,
                decay_is_log=True,
                prefer_metal=True,
            )
            output = gdr_output.attention_outputs
            self._recurrent_state = gdr_output.recurrent_state

        output = self.norm(output, z)
        out_proj_input_dtype = (
            self._decode_out_proj_input_dtype
            if is_decode and self._decode_out_proj_input_dtype is not None
            else _resolve_module_activation_dtype(self.out_proj, hidden_states.dtype)
        )
        if output.dtype != out_proj_input_dtype:
            output = output.astype(out_proj_input_dtype)
        output = output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)

        if ragged_slot_ids is not None and full_conv_state is not None and full_recurrent_state is not None:
            assert self._conv_state is not None
            assert self._recurrent_state is not None
            conv_indices = mx.array(ragged_slot_ids, dtype=mx.int32).reshape(len(ragged_slot_ids), 1, 1)
            recurrent_indices = mx.array(ragged_slot_ids, dtype=mx.int32).reshape(len(ragged_slot_ids), 1, 1, 1)
            self._conv_state = mx.put_along_axis(full_conv_state, conv_indices, self._conv_state, axis=0)
            self._recurrent_state = mx.put_along_axis(
                full_recurrent_state,
                recurrent_indices,
                self._recurrent_state,
                axis=0,
            )
            output = output[:, 0, :]
        elif is_paged:
            output = output[0]

        return output


class Qwen3NextMLP(nn.Module):
    """SiLU-gated feed-forward MLP for Qwen3-Next.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
        act_fn: Activation function.
    """

    def __init__(self, config: Qwen3NextConfig, *, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Qwen3-Next configuration.
            intermediate_size: Override for the intermediate dimensionality.
                Defaults to ``config.intermediate_size`` when ``None``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3NextSparseBlock(nn.Module):
    """MoE block with shared expert for Qwen3-Next.

    Routes tokens through top-k experts and adds a sigmoid-gated shared
    expert output.

    Attributes:
        router: Top-k expert routing module.
        experts: SwitchGLU expert bank.
        shared_expert: Dense MLP shared across all tokens.
        shared_expert_gate: Sigmoid gate for the shared expert output.
    """

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the MoE sparse block.

        Args:
            config: Qwen3-Next configuration.
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
            norm_topk_prob=config.norm_topk_prob,
        )
        self.experts = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)
        self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and combine with shared expert.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Combined output tensor of the same shape.
        """
        inds, scores = self.router(hidden_states)
        out = self.experts(hidden_states, inds)
        out = (out * scores[..., None]).sum(axis=-2).astype(out.dtype)
        shared_out = self.shared_expert(hidden_states)
        shared_gate = mx.sigmoid(self.shared_expert_gate(hidden_states))
        return out + (shared_out * shared_gate)


class Qwen3NextDecoderLayer(nn.Module):
    """Single decoder layer for Qwen3-Next.

    Uses either full softmax attention or linear attention depending on the
    layer index, combined with either a dense MLP or MoE block.

    Attributes:
        use_full_attention: Whether this layer uses full softmax attention.
        use_moe: Whether this layer uses MoE routing.
        self_attn: Attention module (full or linear).
        mlp: Dense MLP or MoE sparse block.
        input_layernorm: Pre-attention RMSNorm with ``(1+w)`` scaling.
        post_attention_layernorm: Post-attention RMSNorm with ``(1+w)`` scaling.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        """Initialize the decoder layer.

        Args:
            config: Qwen3-Next configuration.
            layer_idx: Zero-based index of this layer.
        """
        super().__init__()
        self.use_full_attention = config.is_full_attention_layer(layer_idx)
        self.use_moe = config.is_moe_layer(layer_idx)
        if self.use_full_attention:
            self.self_attn = Qwen3NextFullAttention(config)
        else:
            self.linear_attn = Qwen3NextLinearAttention(config)

        self.mlp = Qwen3NextSparseBlock(config) if self.use_moe else Qwen3NextMLP(config)
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same shape as the input.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn = self.self_attn if self.use_full_attention else self.linear_attn
        hidden_states = residual + attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states

    def decode_step_with_state(
        self,
        hidden_states: mx.array,
        *,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
        decode_state: dict[str, mx.array] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.use_full_attention:
            hidden_states = residual + self.self_attn(
                hidden_states,
                mask=None,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
            new_decode_state = None
        else:
            if decode_state is None:
                raise ValueError("linear-attention decode requires explicit decode_state.")
            attn_out, new_decode_state = self.linear_attn.decode_step_with_state(
                hidden_states,
                decode_state=decode_state,
                cache_metadata=cache_metadata,
            )
            hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, new_decode_state


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextModel(EasyMLXBaseModule):
    """Base Qwen3-Next hybrid transformer model.

    Combines full softmax and linear attention layers with MoE routing.

    Attributes:
        config_class: The associated configuration class (``Qwen3NextConfig``).
        embed_tokens: Token embedding layer.
        layers: List of decoder layers (mixed full/linear attention).
        norm: Final RMSNorm with ``(1+w)`` scaling.
    """

    config_class = Qwen3NextConfig
    supports_multitoken_decode_state = True

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the base Qwen3-Next model.

        Args:
            config: Qwen3-Next configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _activation_dtype(self, fallback: mx.Dtype) -> mx.Dtype:
        try:
            dtype = self.config.mlx_dtype
        except Exception:
            dtype = fallback
        return dtype if mx.issubdtype(dtype, mx.floating) else fallback

    def get_decode_state(self) -> dict[str, list[dict[str, mx.array]]]:
        return {
            "linear_layers": [
                layer.linear_attn.get_decode_state() for layer in self.layers if not layer.use_full_attention
            ]
        }

    def set_decode_state(self, state: dict[str, list[dict[str, mx.array]]]) -> None:
        linear_states = state["linear_layers"]
        state_idx = 0
        for layer in self.layers:
            if layer.use_full_attention:
                continue
            layer.linear_attn.set_decode_state(linear_states[state_idx])
            state_idx += 1

    def decode_step_with_state(
        self,
        input_ids: mx.ArrayLike,
        *,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        decode_state: dict[str, list[dict[str, mx.array]]],
    ) -> tuple[mx.array, dict[str, list[dict[str, mx.array]]]]:
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        input_ids = mx.array(input_ids, dtype=mx.int32)
        hidden_states = self.embed_tokens(input_ids)
        activation_dtype = self._activation_dtype(hidden_states.dtype)
        if hidden_states.dtype != activation_dtype:
            hidden_states = hidden_states.astype(activation_dtype)

        linear_states = decode_state["linear_layers"]
        new_linear_states: list[dict[str, mx.array]] = []
        state_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_state = None
            if not layer.use_full_attention:
                layer_state = linear_states[state_idx]
                state_idx += 1
            hidden_states, new_layer_state = layer.decode_step_with_state(
                hidden_states,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
                decode_state=layer_state,
            )
            if new_layer_state is not None:
                new_linear_states.append(new_layer_state)

        return self.norm(hidden_states), {"linear_layers": new_linear_states}

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Qwen3-Next transformer stack.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states from the transformer stack.

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
        activation_dtype = self._activation_dtype(hidden_states.dtype)
        if hidden_states.dtype != activation_dtype:
            hidden_states = hidden_states.astype(activation_dtype)

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


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextForCausalLM(BaseCausalLMModule[Qwen3NextModel, Qwen3NextConfig]):
    """Qwen3-Next model with a causal language modeling head.

    Wraps ``Qwen3NextModel`` and adds vocabulary projection to produce
    next-token logits. Embeddings are tied by default.

    Attributes:
        config_class: The associated configuration class (``Qwen3NextConfig``).
    """

    config_class = Qwen3NextConfig
    supports_multitoken_decode_state = True

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the causal language model.

        Args:
            config: Qwen3-Next configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3NextModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("Qwen3NextForCausalLM", "Qwen3NextModel")
