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

"""Falcon-H1 MLX model implementation for serving and inference.

This is a hybrid Attention + Mamba2 + MLP architecture. Each decoder layer
contains an attention block, a Mamba SSM block, and an MLP, with MuP-style
scaling multipliers applied throughout.

Attention layers use the standard EasyMLX attention infrastructure with
``cache_views``. Mamba state is managed internally within each mixer block.
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

from .falcon_h1_configuration import FalconH1Config

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


@partial(mx.compile, shapeless=True)
def _swiglu(gate, x):
    """Compute SiLU-gated linear unit: ``SiLU(gate) * x``.

    Compiled with ``mx.compile`` for performance.

    Args:
        gate: Gate tensor.
        x: Input tensor.

    Returns:
        Element-wise ``SiLU(gate) * x``.
    """
    return nn.silu(gate) * x


def _compute_mup_vector(config: FalconH1Config) -> mx.array:
    """Build the per-component MuP scaling vector for the SSM in_proj.

    Creates a concatenated vector of MuP multipliers for each component of
    the SSM input projection: [x, z, B, C, dt], where each component's
    multiplier is broadcast to its corresponding dimension.

    Args:
        config: Falcon-H1 model configuration.

    Returns:
        A 1-D ``mx.array`` of per-component multipliers.
    """
    intermediate_size = config.mamba_d_ssm
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    sizes = [
        intermediate_size,
        intermediate_size,
        groups_time_state_size,
        groups_time_state_size,
        num_heads,
    ]
    return mx.concatenate(
        [mx.broadcast_to(mx.array(m), (s,)) for s, m in zip(sizes, config.ssm_multipliers, strict=False)]
    )


@mx.compile
def _compute_dt(dt, dt_bias, time_step_limit):
    """Compute the discretized time step from raw dt logits.

    Applies softplus activation with bias, then clips to the time step range.

    Args:
        dt: Raw time step logits.
        dt_bias: Per-head bias added before activation.
        time_step_limit: Tuple of ``(min, max)`` clip bounds.

    Returns:
        Discretized time steps.
    """
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def _segsum(x, mask=None):
    """Compute segmented cumulative sum for SSD attention decay.

    Args:
        x: Input tensor of shape ``(..., seq_len)``.
        mask: Optional boolean mask for segment boundaries.

    Returns:
        Segmented cumulative sum of shape ``(..., seq_len, seq_len)``.
    """
    seq_len = x.shape[-1]
    if mask is not None:
        mask = mx.expand_dims(mask, 1)
        x = x * mask
    x = mx.repeat(x[..., None], seq_len, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    if mask is not None:
        x_segsum = mx.where(mask[..., None, :] * mask[..., None], x_segsum, -float("inf"))
    return x_segsum


def _ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state=None,
    time_step_limit=(0.001, 100.0),
    mask=None,
    lengths=None,
    step: int = 256,
):
    """SSD-SSM forward pass using chunk-based surrogate attention.

    Processes the input in chunks of ``step`` positions, computing a
    surrogate attention matrix from the SSM dynamics (CB decay product)
    within each chunk, and propagating state across chunks.

    Args:
        x: Input hidden states of shape ``(batch, seq_len, heads, head_dim)``.
        A_log: Log of SSM decay parameter A of shape ``(heads,)``.
        B: SSM input matrix of shape ``(batch, seq_len, groups, state_dim)``.
        C: SSM output matrix of shape ``(batch, seq_len, groups, state_dim)``.
        D: SSM skip connection of shape ``(heads,)``.
        dt: Raw time step logits of shape ``(batch, seq_len, heads)``.
        dt_bias: Per-head time step bias of shape ``(heads,)``.
        state: Optional initial SSM state.
        time_step_limit: Tuple of ``(min, max)`` for time step clipping.
        mask: Optional boolean mask of shape ``(batch, seq_len)``.
        lengths: Optional per-sequence lengths for variable-length batching.
        step: Chunk size for processing. Defaults to ``256``.

    Returns:
        A tuple of ``(y, state)`` where ``y`` is the output of shape
        ``(batch, seq_len, heads, head_dim)`` and ``state`` is the final
        SSM state.
    """
    b, sl, h, dh = x.shape
    _, _, g, d = B.shape

    dt = _compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, sl, h, 1) * x

    def _step_fn(dtx, dtA, B, C, state, mask):
        s = dtx.shape[1]
        B_t = mx.transpose(B, (0, 2, 3, 1))
        CB = mx.swapaxes(C, 1, 2) @ B_t
        CB = mx.repeat(CB, repeats, axis=1)
        decay = mx.exp(_segsum(dtA.swapaxes(1, 2), mask=mask))
        surrogate_attention_matrix = mx.tril(CB * decay, 0)
        y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        if lengths is not None:
            pos = mx.maximum(mx.minimum(lengths, step) - 1, 0)
            pos = mx.expand_dims(pos, (1, 2, 3))
            decay = mx.take_along_axis(decay, pos, axis=2)
        else:
            decay = decay[:, :, -1:, :]

        decay = decay.transpose(0, 3, 1, 2)
        B_rep = mx.repeat(B_t, h // g, axis=1).swapaxes(2, 3)
        dtxdecay = dtx * decay
        dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)
        next_state = dtxdecay @ B_rep

        if state is not None:
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
            C_r = C.reshape(b, s, g, 1, d, 1)
            y_prev = (state.reshape((b, 1, g, repeats, dh, d)) @ C_r).squeeze(-1).flatten(2, 3)
            y += exp_dtA_cumsum[..., None] * y_prev
        if lengths is not None and state is not None:
            next_state = mx.where(mx.expand_dims(lengths < 0, (1, 2, 3)), state, next_state)
        return y, next_state

    ys = []
    for i in range(0, sl, step):
        y, state = _step_fn(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
            None if mask is None else mask[..., i : i + step],
        )
        if lengths is not None:
            lengths = lengths - step
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1)
    return y, state


def _ssm_update(
    hidden_states,
    A_log,
    B,
    C,
    D,
    dt,
    dt_bias,
    state=None,
    time_step_limit=(0.001, 100.0),
    mask=None,
    lengths=None,
):
    """Dispatch the SSM update to the chunk-based implementation.

    Args:
        hidden_states: Input of shape ``(batch, seq_len, heads, head_dim)``.
        A_log: Log of SSM decay parameter.
        B: SSM input matrix.
        C: SSM output matrix.
        D: SSM skip connection.
        dt: Time step logits.
        dt_bias: Time step bias.
        state: Optional initial SSM state.
        time_step_limit: Time step clip bounds.
        mask: Optional boolean mask.
        lengths: Optional per-sequence lengths.

    Returns:
        A tuple of ``(output, state)``.
    """
    return _ssm_attn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        time_step_limit,
        mask=mask,
        lengths=lengths,
    )


class FalconH1RMSNormGated(nn.Module):
    """Gated RMSNorm for the Falcon-H1 Mamba mixer.

    Applies RMSNorm to the hidden states and optionally gates them with a
    SiLU activation. The order of norm and gating is controlled by
    ``norm_before_gate``: when ``True``, norm is applied first and then the
    result is gated; when ``False``, gating is applied first.

    Attributes:
        weight: Learnable RMSNorm scale parameters.
        variance_epsilon: Epsilon for numerical stability.
        n_groups: Number of normalization groups.
        norm_before_gate: Whether to apply norm before gating.

    Example::

        >>> norm = FalconH1RMSNormGated(1536, eps=1e-5)
        >>> output = norm(hidden_states, gate=gate)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, n_groups: int = 1, norm_before_gate: bool = True):
        """Initialize FalconH1RMSNormGated.

        Args:
            hidden_size: Feature dimension to normalize.
            eps: Epsilon for RMSNorm. Defaults to ``1e-6``.
            n_groups: Number of normalization groups. Defaults to ``1``.
            norm_before_gate: Whether to apply norm before the SiLU gate.
                Defaults to ``True``.
        """
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states, gate=None):
        """Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor to normalize.
            gate: Optional gate tensor for SiLU gating. If ``None``,
                only RMSNorm is applied without gating.

        Returns:
            Normalized (and optionally gated) tensor.
        """
        if not self.norm_before_gate and gate is not None:
            hidden_states = _swiglu(gate, hidden_states)
        hidden_states = mx.fast.rms_norm(hidden_states, self.weight, self.variance_epsilon)
        if self.norm_before_gate and gate is not None:
            hidden_states = _swiglu(gate, hidden_states)
        return hidden_states


class FalconH1Attention(nn.Module):
    """Standard attention block with RoPE, used within each hybrid layer.

    Provides the attention component of the hybrid Attention+Mamba+MLP
    architecture. Uses the EasyMLX ``AttentionPerformer`` infrastructure
    with standard KV caching.

    Attributes:
        hidden_size: Hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        scale: Attention scaling factor.

    Example::

        >>> attn = FalconH1Attention(config)
        >>> output = attn(hidden_states, mask=mask, cache_view=cache)
    """

    def __init__(self, config: FalconH1Config):
        """Initialize Falcon-H1 attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale,
            attn_mechanism=getattr(config, "attn_mechanism", None),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the attention forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
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


class FalconH1Mixer(nn.Module):
    """Mamba2 SSM mixer block.

    Implements a Mamba2-style selective state space model (SSM) with:

    - An input projection that produces gate, conv_input, and dt components.
    - A depthwise 1-D convolution followed by SiLU activation.
    - SSM dynamics using chunk-based surrogate attention (SSD).
    - Optional gated RMSNorm or SiLU gating before output projection.

    SSM state is managed internally (no external cache needed). The mixer
    runs in parallel with the attention block within each hybrid layer.

    Attributes:
        num_heads: Number of SSM heads.
        hidden_size: Model hidden dimension.
        ssm_state_size: SSM state dimension per group.
        conv_kernel_size: Convolution kernel size.
        intermediate_size: SSM intermediate (expanded) dimension.
        n_groups: Number of SSM groups.
        head_dim: Per-head dimension.
        chunk_size: Chunk size for SSD processing.
        A_log: Log of the SSM decay parameter A.
        D: SSM skip connection parameter.
        dt_bias: Per-head time step bias.

    Example::

        >>> mixer = FalconH1Mixer(config)
        >>> output = mixer(hidden_states)
    """

    def __init__(self, config: FalconH1Config):
        """Initialize the Mamba2 mixer.

        Args:
            config: Model configuration with Mamba hyperparameters.
        """
        super().__init__()
        self.num_heads = config.mamba_n_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_d_ssm
        self.use_conv_bias = config.mamba_conv_bias
        self.layer_norm_epsilon = config.rms_norm_eps
        self.groups_time_state_size = config.mamba_n_groups * self.ssm_state_size
        self.n_groups = config.mamba_n_groups
        self.head_dim = config.mamba_d_head
        self.chunk_size = config.mamba_chunk_size

        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.mamba_proj_bias,
        )

        self.dt_bias = mx.ones(self.num_heads)
        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)

        self.mamba_rms_norm = config.mamba_rms_norm
        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )

        self.D = mx.ones(self.num_heads)
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.projectors_bias,
        )

    def _conv(self, conv_input: mx.array, mask: mx.array | None = None) -> mx.array:
        """Apply causal depthwise convolution with SiLU activation.

        Args:
            conv_input: Input of shape ``(batch, seq_len, conv_dim)``.
            mask: Optional boolean mask to zero out padded positions.

        Returns:
            Convolution output of shape ``(batch, seq_len, conv_dim)``.
        """
        if mask is not None:
            conv_input = mx.where(mask[..., None], conv_input, 0)
        padded_input = mx.pad(conv_input, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Run the SSM forward pass.

        Args:
            hidden_states: SSM input of shape ``(batch, seq_len, intermediate_size)``.
            B: SSM input matrix of shape ``(batch, seq_len, groups * state_dim)``.
            C: SSM output matrix of shape ``(batch, seq_len, groups * state_dim)``.
            dt: Time step logits of shape ``(batch, seq_len, num_heads)``.
            mask: Optional boolean mask.

        Returns:
            SSM output of shape ``(batch, seq_len, intermediate_size)``.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        y, _ = _ssm_update(
            hidden_states,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            state=None,
            time_step_limit=self.time_step_limit,
            mask=mask,
        )
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(self, input_states: mx.array, mask: mx.array | None = None) -> mx.array:
        """Run the full Mamba mixer forward pass.

        Projects input, applies convolution, runs SSM, applies gating
        (either via FalconH1RMSNormGated or SiLU), and projects output.

        Args:
            input_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional boolean mask for padding.

        Returns:
            Output of shape ``(batch, seq_len, hidden_size)``.
        """
        projected_states = self.in_proj(input_states)

        gate, conv_input, dt = mx.split(
            projected_states,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )

        conv_output = self._conv(conv_input, mask)

        hidden_states, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        y = self._ssm(hidden_states, B, C, dt, mask=mask)

        if self.mamba_rms_norm:
            y = self.norm(y, gate)
        else:
            y = _swiglu(gate, y)

        return self.out_proj(y)


class FalconH1MLP(nn.Module):
    """SwiGLU MLP for Falcon-H1 hybrid layers.

    Part of the Attention+Mamba+MLP triplet in each decoder layer.
    """

    def __init__(self, config: FalconH1Config):
        """Initialize the MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass.

        Args:
            x: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class FalconH1DecoderLayer(nn.Module):
    """Single Falcon-H1 decoder layer: Attention + Mamba + MLP.

    Each layer applies the input through three parallel/sequential blocks:

    1. **Mamba SSM** (mixer) -- processes the normalized hidden states through
       a selective state space model. No KV cache needed.
    2. **Attention** -- standard multi-head attention with RoPE and KV cache.
    3. **MLP** -- SwiGLU feed-forward network.

    The Mamba and attention outputs are added to the residual, then the MLP
    output is added in a separate residual connection.

    Attributes:
        self_attn: Attention module with KV caching.
        mamba: Mamba2 SSM mixer module.
        feed_forward: SwiGLU MLP module.
        input_layernorm: Pre-attention/mamba RMSNorm.
        pre_ff_layernorm: Pre-MLP RMSNorm.
    """

    def __init__(self, config: FalconH1Config):
        """Initialize a hybrid decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = FalconH1Attention(config)
        self.mamba = FalconH1Mixer(config)
        self.feed_forward = FalconH1MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the hybrid decoder layer forward pass.

        Applies Mamba SSM and attention in parallel on the normalized input,
        sums both with the residual, then applies the MLP with a second
        residual connection.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view for the attention block.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        mamba_h = self.mamba(h)

        attn_h = self.self_attn(
            h,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        h = residual + mamba_h + attn_h

        residual = h
        h = self.pre_ff_layernorm(h)
        h = self.feed_forward(h)
        return residual + h


@register_module(task_type=TaskType.BASE_MODULE, config=FalconH1Config, model_type="falcon_h1")
class FalconH1Model(EasyMLXBaseModule):
    """Base Falcon-H1 hybrid Attention+Mamba+MLP model.

    Each decoder layer contains three blocks (attention, Mamba SSM, MLP) that
    work together. The attention block uses standard KV caching; the Mamba
    SSM manages its state internally. MuP scaling multipliers are applied
    during weight sanitization to scale different components appropriately.

    Attributes:
        config_class: The configuration class (``FalconH1Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``FalconH1DecoderLayer`` instances.
        final_layernorm: Final RMS normalization.

    Example::

        >>> config = FalconH1Config()
        >>> model = FalconH1Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = FalconH1Config

    def __init__(self, config: FalconH1Config):
        """Initialize the Falcon-H1 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self._mup_vector = _compute_mup_vector(config)
        self.layers = [FalconH1DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the hybrid transformer forward pass.

        Passes input through all decoder layers, each of which contains
        an attention block, a Mamba SSM block, and an MLP. The attention
        blocks use ``cache_views`` for KV caching; Mamba blocks manage
        their state internally.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings instead of ``input_ids``.
            cache_views: Per-layer KV cache views for the attention blocks.
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
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.final_layernorm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Apply MuP multipliers and transpose conv1d weights from upstream.

        Performs the following transformations (only when needed, detected by
        checking conv1d weight shape):

        1. **Embedding scaling**: Multiplies ``embed_tokens.weight`` by
           ``embedding_multiplier``.
        2. **LM head scaling**: Multiplies ``lm_head.weight`` by
           ``lm_head_multiplier``.
        3. **Attention scaling**: Multiplies Q/K projections by
           ``attention_in_multiplier``, O projection by ``attention_out_multiplier``.
        4. **SSM scaling**: Multiplies ``in_proj.weight`` by ``ssm_in_multiplier``
           times the per-component MuP vector. Multiplies ``out_proj.weight``
           by ``ssm_out_multiplier``.
        5. **MLP scaling**: Multiplies ``gate_proj`` and ``down_proj`` by
           the respective ``mlp_multipliers``.
        6. **Conv1d transpose**: Transposes conv1d weights from upstream
           ``(out, kernel, in)`` to MLX ``(out, in, kernel)`` layout.
        7. **Rotary filter**: Removes ``rotary_emb.inv_freq`` keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary with MuP scaling applied.
        """

        conv_key = None
        for k in weights:
            if "conv1d.weight" in k:
                conv_key = k
                break

        needs_sanitize = False
        if conv_key is not None:
            c1d = weights[conv_key]
            if c1d.shape[-1] > c1d.shape[1]:
                needs_sanitize = True

        if not needs_sanitize:
            return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        config = self.config
        sanitized = {}
        for name, param in weights.items():
            if "rotary_emb.inv_freq" in name:
                continue
            if name.endswith("embed_tokens.weight"):
                param = param * config.embedding_multiplier
            elif name.endswith("lm_head.weight"):
                param = param * config.lm_head_multiplier
            elif name.endswith("q_proj.weight") or name.endswith("k_proj.weight"):
                param = param * config.attention_in_multiplier
            elif name.endswith("key_proj.weight"):
                param = param * config.attention_in_multiplier * config.key_multiplier
            elif name.endswith("o_proj.weight"):
                param = param * config.attention_out_multiplier
            elif name.endswith("out_proj.weight"):
                param = param * config.ssm_out_multiplier
            elif name.endswith("gate_proj.weight"):
                param = param * config.mlp_multipliers[0]
            elif name.endswith("down_proj.weight"):
                param = param * config.mlp_multipliers[1]
            elif name.endswith("in_proj.weight"):
                param = param * (config.ssm_in_multiplier * self._mup_vector.astype(param.dtype)[:, None])
            elif "conv1d.weight" in name:
                param = param.transpose(0, 2, 1)
            sanitized[name] = param
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=FalconH1Config, model_type="falcon_h1")
class FalconH1ForCausalLM(BaseCausalLMModule[FalconH1Model, FalconH1Config]):
    """Falcon-H1 causal language model.

    Wraps ``FalconH1Model`` with a language modeling head. When embeddings
    are tied, applies a ``lm_head_multiplier / embedding_multiplier``
    correction to the logits via ``compute_lm_logits``.

    Example::

        >>> model = FalconH1ForCausalLM(FalconH1Config())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = FalconH1Config

    def __init__(self, config: FalconH1Config):
        """Initialize the Falcon-H1 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=FalconH1Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits with MuP scaling correction.

        When word embeddings are tied, the MuP ``embedding_multiplier`` was
        already applied to the embedding weights during sanitization. This
        method corrects by scaling the logits by
        ``lm_head_multiplier / embedding_multiplier``.

        Args:
            hidden_states: Final hidden states of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        logits = super().compute_lm_logits(hidden_states)
        if self._tie_word_embeddings:
            config = self.config
            logits = logits * (config.lm_head_multiplier / config.embedding_multiplier)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Apply CausalLM sanitization followed by base model MuP scaling.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """
        weights = super().sanitize(weights)
        return self.base_model.sanitize(weights)


__all__ = ("FalconH1ForCausalLM", "FalconH1Model")
