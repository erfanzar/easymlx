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

"""Gated Delta Rule (GDR) linear attention implementation for EasyMLX.

This module provides the GatedDeltaRuleOp operation, a linear attention
mechanism used in hybrid transformer architectures (e.g. Qwen3Next).
The gated delta rule combines:

1. Causal convolution for local context
2. Gated linear attention with delta rule updates
3. Learnable decay for forgetting previous state

Key characteristics:

- Linear complexity O(N) in sequence length (vs O(N^2) for standard attention)
- Maintains recurrent state for efficient inference
- Supports chunked computation for efficient prefill

The algorithm:

    Prefill (chunked):
        - Process sequence in chunks for parallelism
        - Intra-chunk: parallel computation within each chunk
        - Inter-chunk: sequential state propagation

    Decode (recurrent):
        - Single-step state update, O(1) per token
        - h_t = decay * h_{t-1} + beta_t * (v_t outer k_t)
        - o_t = h_t @ q_t

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/
      src/transformers/models/qwen3_next/
"""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

import mlx.core as mx

from .._attention_outputs import AttentionOutput
from .._base_operation import BaseOperation, OperationRegistry
from ..requirements import ExecutionMode, OperationRequirements


@dataclass(slots=True)
class GatedDeltaRuleOutput(AttentionOutput):
    """Output container for GatedDeltaRule operation.

    Extends ``AttentionOutput`` with recurrent state fields needed for
    hybrid attention models.

    Attributes:
        attention_outputs: Output tensor
            ``[batch, seq_len, num_heads, head_dim]``.
        attention_weights: Always ``None`` for linear attention (no explicit
            weight matrix).
        conv_state: Updated convolution state
            ``[batch, d_inner, d_conv]`` or ``None``.
        recurrent_state: Updated recurrent state
            ``[batch, num_heads, head_dim, d_state]`` or ``None``.
    """

    conv_state: mx.array | None = None
    recurrent_state: mx.array | None = None


def _l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    """L2-normalize ``x`` along ``axis``.

    Args:
        x: Input array.
        axis: Axis along which to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2-normalized array with the same shape as ``x``.
    """
    norm = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def _recurrent_step(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array | None,
    recurrent_state: mx.array | None,
    use_qk_l2norm: bool = True,
) -> tuple[mx.array, mx.array]:
    """Single recurrent step for decode (seq_len == 1).

    Computes the gated delta rule update for a single token:
        h_t = decay * h_{t-1} + beta_t * (v_t outer k_t)
        o_t = h_t @ q_t

    Args:
        query: Query tensor ``[batch, 1, num_heads, head_dim]``.
        key: Key tensor ``[batch, 1, num_heads, head_dim]``.
        value: Value tensor ``[batch, 1, num_heads, d_state]``.
        beta: Gating tensor ``[batch, 1, num_heads]`` (scalar per head)
            or ``[batch, 1, num_heads, head_dim]``.
        decay: Optional decay factors ``[num_heads, head_dim]`` or ``None``.
        recurrent_state: Previous recurrent state
            ``[batch, num_heads, head_dim, d_state]`` or ``None``.
        use_qk_l2norm: Whether to L2-normalize query and key.

    Returns:
        Tuple of (output ``[batch, 1, num_heads, d_state]``,
        new_state ``[batch, num_heads, head_dim, d_state]``).
    """
    batch, _, num_heads, head_dim = query.shape
    d_state = value.shape[-1]

    # Squeeze the length-1 sequence dim: [B, H, D]
    q = query[:, 0]  # [B, H, head_dim]
    k = key[:, 0]  # [B, H, head_dim]
    v = value[:, 0]  # [B, H, d_state]

    if use_qk_l2norm:
        q = _l2_normalize(q, axis=-1)
        k = _l2_normalize(k, axis=-1)

    # beta: [B, 1, H, ...] -> [B, H] or [B, H, head_dim]
    b = beta[:, 0]
    if b.ndim == 2:
        # scalar per head: [B, H] -> [B, H, 1, 1] for broadcasting
        b = b[:, :, None, None]
    elif b.ndim == 3:
        # per-dim: [B, H, head_dim] -> [B, H, head_dim, 1]
        b = b[:, :, :, None]

    # Outer product: v_t (x) k_t -> [B, H, head_dim, d_state]
    # k: [B, H, head_dim] -> [..., head_dim, 1]
    # v: [B, H, d_state]  -> [..., 1, d_state]
    outer = k[:, :, :, None] * v[:, :, None, :]  # [B, H, head_dim, d_state]

    # Initialize state if needed
    if recurrent_state is None:
        recurrent_state = mx.zeros(
            (batch, num_heads, head_dim, d_state),
            dtype=query.dtype,
        )

    # State update: h_t = decay * h_{t-1} + beta * outer
    new_state = recurrent_state
    if decay is not None:
        # decay: [H, head_dim] -> [1, H, head_dim, 1]
        d = decay[None, :, :, None]
        new_state = d * new_state
    new_state = new_state + b * outer

    # Output: o_t = h_t @ q_t -> sum over head_dim
    # new_state: [B, H, head_dim, d_state], q: [B, H, head_dim]
    # -> einsum "bhkd,bhk->bhd"
    output = mx.sum(new_state * q[:, :, :, None], axis=2)  # [B, H, d_state]

    # Restore sequence dim: [B, 1, H, d_state]
    output = output[:, None, :, :]

    return output, new_state


def _chunked_forward(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array | None,
    recurrent_state: mx.array | None,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Chunked forward pass for prefill (seq_len > 1).

    Processes the full sequence by iterating over chunks. Within each
    chunk, a sequential scan applies the gated delta rule so that state
    propagates correctly across the chunk boundary.

    Args:
        query: Query tensor ``[batch, seq_len, num_heads, head_dim]``.
        key: Key tensor ``[batch, seq_len, num_heads, head_dim]``.
        value: Value tensor ``[batch, seq_len, num_heads, d_state]``.
        beta: Gating tensor ``[batch, seq_len, num_heads]`` or
            ``[batch, seq_len, num_heads, head_dim]``.
        decay: Optional decay factors ``[num_heads, head_dim]`` or ``None``.
        recurrent_state: Optional initial state
            ``[batch, num_heads, head_dim, d_state]``.
        use_qk_l2norm: Whether to L2-normalize query and key.
        chunk_size: Number of tokens per chunk (default 64).

    Returns:
        Tuple of (output ``[batch, seq_len, num_heads, d_state]``,
        final_state ``[batch, num_heads, head_dim, d_state]``).
    """
    batch, seq_len, num_heads, head_dim = query.shape
    d_state = value.shape[-1]

    if use_qk_l2norm:
        query = _l2_normalize(query, axis=-1)
        key = _l2_normalize(key, axis=-1)

    # Prepare beta shape
    if beta.ndim == 3:
        # [B, T, H] -> [B, T, H, 1, 1] for broadcasting with outer product
        beta_expanded = beta[:, :, :, None, None]
    else:
        # [B, T, H, head_dim] -> [B, T, H, head_dim, 1]
        beta_expanded = beta[:, :, :, :, None]

    # Prepare decay
    if decay is not None:
        # [H, head_dim] -> [1, H, head_dim, 1]
        decay_bcast = decay[None, :, :, None]
    else:
        decay_bcast = None

    # Initialize state
    if recurrent_state is None:
        state = mx.zeros(
            (batch, num_heads, head_dim, d_state),
            dtype=query.dtype,
        )
    else:
        state = recurrent_state

    # Collect outputs for all chunks
    output_chunks: list[mx.array] = []

    num_chunks = math.ceil(seq_len / chunk_size)
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_len = end - start

        q_c = query[:, start:end]  # [B, chunk_len, H, head_dim]
        k_c = key[:, start:end]
        v_c = value[:, start:end]
        b_c = beta_expanded[:, start:end]

        # Sequential scan within the chunk
        chunk_outputs: list[mx.array] = []
        for t in range(chunk_len):
            k_t = k_c[:, t]  # [B, H, head_dim]
            v_t = v_c[:, t]  # [B, H, d_state]
            q_t = q_c[:, t]  # [B, H, head_dim]

            # beta for this timestep
            if beta.ndim == 3:
                b_t = b_c[:, t]  # [B, H, 1, 1]
            else:
                b_t = b_c[:, t]  # [B, H, head_dim, 1]

            # Outer product: [B, H, head_dim, d_state]
            outer = k_t[:, :, :, None] * v_t[:, :, None, :]

            # State update
            if decay_bcast is not None:
                state = decay_bcast * state
            state = state + b_t * outer

            # Output: sum over head_dim
            o_t = mx.sum(state * q_t[:, :, :, None], axis=2)  # [B, H, d_state]
            chunk_outputs.append(o_t)

        # Stack chunk outputs: [B, chunk_len, H, d_state]
        chunk_out = mx.stack(chunk_outputs, axis=1)
        output_chunks.append(chunk_out)

    # Concatenate all chunks: [B, seq_len, H, d_state]
    output = mx.concatenate(output_chunks, axis=1)

    return output, state


@OperationRegistry.register
class GatedDeltaRuleOp(BaseOperation):
    """Gated Delta Rule linear attention operation.

    Implements the gated delta rule mechanism for efficient linear
    attention on Apple Silicon via MLX:

    - **Prefill** (chunked): Processes full sequences with sequential
      scan within chunks.
    - **Decode** (recurrent): Single-step state update for O(1)
      per-token generation.

    The state update rule is::

        h_t = decay * h_{t-1} + beta_t * (v_t outer k_t)
        o_t = h_t @ q_t

    Where:
        - ``beta_t`` is a learned gating signal.
        - ``decay`` is an optional forgetting factor.
        - ``v_t outer k_t`` is the outer product.

    Registered under the names ``"gated_delta_rule"`` and ``"gdr"``.

    Example::

        >>> from easymlx.operations.kernels.gated_delta_rule import (
        ...     GatedDeltaRuleOp,
        ... )
        >>> op = GatedDeltaRuleOp()
        >>> output = op(
        ...     query=query,
        ...     key=key,
        ...     value=value,
        ...     beta=beta,
        ...     decay=decay,
        ... )
    """

    @classmethod
    def get_impl_name(cls) -> tuple[str, ...]:
        """Return the registered names of this operation.

        Returns:
            Tuple ``("gated_delta_rule", "gdr")``.
        """
        return ("gated_delta_rule", "gdr")

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Return requirements for GatedDeltaRuleOp.

        The GDR mechanism is a recurrent/linear attention variant that
        needs recurrent state management but does not need KV caches or
        attention masks.

        Args:
            mode: The execution mode (prefill, decode, or mixed).

        Returns:
            An ``OperationRequirements`` instance.
        """
        return OperationRequirements.default("gated_delta_rule")

    def forward_native(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        decay: mx.array | None = None,
        conv_state: mx.array | None = None,
        recurrent_state: mx.array | None = None,
        use_qk_l2norm: bool = True,
        chunk_size: int = 64,
        **kwargs: tp.Any,
    ) -> GatedDeltaRuleOutput:
        """Forward pass for gated delta rule attention.

        Automatically selects recurrent mode (decode) when
        ``seq_len == 1`` and chunked mode (prefill) otherwise.

        Args:
            query: Query tensor ``[batch, seq_len, num_heads, head_dim]``.
            key: Key tensor ``[batch, seq_len, num_heads, head_dim]``.
            value: Value tensor ``[batch, seq_len, num_heads, d_state]``.
            beta: Gating tensor. Shape is either
                ``[batch, seq_len, num_heads]`` (scalar gate per head) or
                ``[batch, seq_len, num_heads, head_dim]`` (per-dim gate).
            decay: Optional decay factors ``[num_heads, head_dim]``.
            conv_state: Optional convolution state passed through for
                external conv management ``[batch, d_inner, d_conv]``.
            recurrent_state: Optional recurrent state
                ``[batch, num_heads, head_dim, d_state]``.
            use_qk_l2norm: Whether to L2-normalize query and key
                (default ``True``).
            chunk_size: Chunk size for prefill mode (default 64).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ``GatedDeltaRuleOutput`` containing attention outputs and
            updated states.
        """
        runtime_dtype = mx.float32
        if self.metadata is not None and self.metadata.runtime_dtype is not None:
            runtime_dtype = self.metadata.runtime_dtype

        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)
        beta = beta.astype(runtime_dtype)

        # Squeeze trailing singleton on beta if present
        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta[..., 0]

        if decay is not None:
            decay = decay.astype(runtime_dtype)
            if decay.ndim == 4 and decay.shape[-1] == 1:
                decay = decay[..., 0]

        if recurrent_state is not None:
            recurrent_state = recurrent_state.astype(runtime_dtype)

        seq_len = query.shape[1]
        is_inference = seq_len == 1

        if is_inference:
            outputs, new_recurrent_state = _recurrent_step(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=use_qk_l2norm,
            )
        else:
            outputs, new_recurrent_state = _chunked_forward(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=use_qk_l2norm,
                chunk_size=chunk_size,
            )

        return GatedDeltaRuleOutput(
            attention_outputs=outputs,
            attention_weights=None,
            conv_state=conv_state,
            recurrent_state=new_recurrent_state,
        )

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        decay: mx.array | None = None,
        conv_state: mx.array | None = None,
        recurrent_state: mx.array | None = None,
        use_qk_l2norm: bool = True,
        chunk_size: int = 64,
        **kwargs: tp.Any,
    ) -> GatedDeltaRuleOutput:
        """Execute the gated delta rule operation.

        Delegates to ``forward_native``.

        Args:
            query: Query tensor ``[batch, seq_len, num_heads, head_dim]``.
            key: Key tensor ``[batch, seq_len, num_heads, head_dim]``.
            value: Value tensor ``[batch, seq_len, num_heads, d_state]``.
            beta: Gating tensor (scalar or per-dim).
            decay: Optional decay factors.
            conv_state: Optional convolution state (passed through).
            recurrent_state: Optional recurrent state.
            use_qk_l2norm: Whether to L2-normalize query and key.
            chunk_size: Chunk size for prefill mode.
            **kwargs: Additional keyword arguments forwarded.

        Returns:
            ``GatedDeltaRuleOutput`` with attention outputs and updated
            states.
        """
        return self.forward_native(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            chunk_size=chunk_size,
            **kwargs,
        )


__all__ = (
    "GatedDeltaRuleOp",
    "GatedDeltaRuleOutput",
)
