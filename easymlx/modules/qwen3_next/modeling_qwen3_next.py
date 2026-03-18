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
    PageCache,
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
from easymlx.operations.kernels.gated_delta_rule import GatedDeltaRuleOp

from .qwen3_next_configuration import Qwen3NextConfig

CacheView = TransformerCacheView | PageCache


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

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply RMSNorm with ``(1 + weight)`` scaling.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Normalized tensor of the same shape and original dtype.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states / mx.sqrt(variance + self.eps)
        hidden_states = hidden_states * (1.0 + self.weight)
        return hidden_states.astype(input_dtype)


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

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        """Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            gate: Gate tensor of the same shape, activated with SiLU.

        Returns:
            Gated normalized tensor of the same shape and original dtype.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states / mx.sqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight.astype(hidden_states.dtype)
        hidden_states = hidden_states * nn.silu(gate.astype(mx.float32))
        return hidden_states.astype(input_dtype)


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
            self.rope = get_rope(
                dims=self.rotary_dim,
                base=config.rope_theta,
                traditional=False,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )

        self.attention_performer = AttentionPerformer(scale=self.scale)

    def _apply_partial_rope_blhd(self, tensor: mx.array, *, offset: int = 0) -> mx.array:
        """Apply partial RoPE on ``[B, L, H, D]`` layout.

        Transposes internally to ``[B, H, L, D]`` for the rope module, applies
        RoPE to the first ``rotary_dim`` dimensions, and transposes back.

        Args:
            tensor: Input tensor of shape ``(batch, seq_len, heads, head_dim)``.
            offset: Position offset for the rotary embeddings.

        Returns:
            Tensor with partial RoPE applied, same shape as input.
        """
        if self.rope is None or self.rotary_dim <= 0:
            return tensor
        t = tensor.transpose(0, 2, 1, 3)
        head_rot = t[..., : self.rotary_dim]
        head_pass = t[..., self.rotary_dim :]
        head_rot = self.rope(head_rot, offset=offset)
        t = mx.concatenate([head_rot, head_pass], axis=-1)
        return t.transpose(0, 2, 1, 3)

    def _apply_partial_rope_thd(self, tensor: mx.array, *, offset: int = 0) -> mx.array:
        """Apply partial RoPE on ``[T, H, D]`` layout.

        Adds a batch dimension, transposes for the rope module, applies RoPE,
        and squeezes back to 3-D.

        Args:
            tensor: Input tensor of shape ``(total_tokens, heads, head_dim)``.
            offset: Position offset for the rotary embeddings.

        Returns:
            Tensor with partial RoPE applied, same shape as input.
        """
        if self.rope is None or self.rotary_dim <= 0:
            return tensor
        t = tensor[None].transpose(0, 2, 1, 3)
        head_rot = t[..., : self.rotary_dim]
        head_pass = t[..., self.rotary_dim :]
        head_rot = self.rope(head_rot, offset=offset)
        t = mx.concatenate([head_rot, head_pass], axis=-1)
        return t.transpose(0, 2, 1, 3).squeeze(0)

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

        offset = cache_view.offset if cache_view is not None else 0
        if q.ndim == 4:
            q = self._apply_partial_rope_blhd(q, offset=offset)
            k = self._apply_partial_rope_blhd(k, offset=offset)
        else:
            q = self._apply_partial_rope_thd(q, offset=offset)
            k = self._apply_partial_rope_thd(k, offset=offset)

        attn = self.attention_performer(
            q,
            k,
            v,
            rope=None,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        attn = attn * mx.sigmoid(gate)
        return self.o_proj(attn.reshape(*lead, -1))


def _shift_conv_state_left(conv_state: mx.array, new_value: mx.array) -> mx.array:
    """Shift conv rolling buffer left, append new token at the right."""
    return mx.concatenate([conv_state[:, :, 1:], new_value[:, :, None]], axis=-1)


def _manual_depthwise_conv(conv_state: mx.array, kernel: mx.array) -> mx.array:
    """Single-step depthwise conv from cached state: sum(state * kernel, axis=-1) then silu."""
    return nn.silu(mx.sum(conv_state.astype(mx.float32) * kernel[None, :, :].astype(mx.float32), axis=-1))


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
        self.conv_dim = self.key_dim * 2 + self.num_k_heads * self.head_v_dim
        self.d_conv = int(getattr(config, "linear_conv_kernel_dim", 4))
        self.expand_ratio = self.num_v_heads // self.num_k_heads

        self.conv_value_dim = self.num_k_heads * self.head_v_dim
        qkv_dim = self.key_dim * 2 + self.conv_value_dim
        self.in_proj_qkv = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.conv_value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_k_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_k_heads, bias=False)
        self.out_proj = nn.Linear(self.conv_value_dim, config.hidden_size, bias=False)
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.gdr_op = GatedDeltaRuleOp()

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

        self.A_log = mx.zeros((self.num_k_heads,))
        self.dt_bias = mx.ones((self.num_k_heads,))

        self._conv_state: mx.array | None = None
        self._recurrent_state: mx.array | None = None

    def reset_state(self, batch_size: int = 1):
        """Reset conv and recurrent states for a new generation."""
        self._conv_state = mx.zeros((batch_size, self.conv_dim, self.d_conv))
        self._recurrent_state = mx.zeros(
            (batch_size, self.num_k_heads, self.head_k_dim, self.head_v_dim),
        )

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
        if is_paged:
            hidden_states = hidden_states[None]

        batch_size, seq_len, _ = hidden_states.shape
        is_decode = seq_len == 1 and self._recurrent_state is not None

        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_k_heads, self.head_v_dim)
        beta_raw = self.in_proj_b(hidden_states)
        alpha_raw = self.in_proj_a(hidden_states)

        conv_input = qkv

        A = -mx.exp(self.A_log.astype(mx.float32))
        alpha_biased = alpha_raw.astype(mx.float32) + self.dt_bias.astype(mx.float32)
        decay = A * nn.softplus(alpha_biased)
        beta = mx.sigmoid(beta_raw)

        if is_decode and self._conv_state is not None:
            conv_x = conv_input[:, 0, :]
            self._conv_state = _shift_conv_state_left(self._conv_state, conv_x)
            kernel = self.conv1d.weight.squeeze(-1) if self.conv1d is not None else mx.ones((self.conv_dim, self.d_conv))
            conv_out = _manual_depthwise_conv(self._conv_state, kernel)[:, None, :]
        elif self.conv1d is not None:
            padded = mx.pad(conv_input, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
            conv_out = nn.silu(self.conv1d(padded))
            if seq_len >= self.d_conv:
                self._conv_state = conv_input[:, -self.d_conv :, :].transpose(0, 2, 1)
            else:
                self._conv_state = mx.zeros((batch_size, self.conv_dim, self.d_conv))
                self._conv_state = mx.concatenate(
                    [
                        self._conv_state[:, :, seq_len:],
                        conv_input.transpose(0, 2, 1),
                    ],
                    axis=-1,
                )
        else:
            conv_out = conv_input

        query = conv_out[:, :, : self.key_dim].reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = conv_out[:, :, self.key_dim : self.key_dim * 2].reshape(
            batch_size, seq_len, self.num_k_heads, self.head_k_dim
        )
        value = conv_out[:, :, self.key_dim * 2 :].reshape(batch_size, seq_len, self.num_k_heads, self.head_v_dim)

        gdr_output = self.gdr_op(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=self._recurrent_state,
            use_qk_l2norm=True,
            query_scale=self.head_k_dim**-0.5,
            decay_is_log=True,
            prefer_metal=True,
        )
        output = gdr_output.attention_outputs.astype(self.out_proj.weight.dtype)
        self._recurrent_state = gdr_output.recurrent_state

        output = output.reshape(batch_size * seq_len, self.num_k_heads, self.head_v_dim)
        gate = z.reshape(batch_size * seq_len, self.num_k_heads, self.head_v_dim)
        output = self.norm(output, gate)
        output = output.reshape(batch_size, seq_len, -1)
        output = self.out_proj(output)

        if is_paged:
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

    def __init__(self, config: Qwen3NextConfig):
        """Initialize the base Qwen3-Next model.

        Args:
            config: Qwen3-Next configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
