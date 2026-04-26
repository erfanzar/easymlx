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
        - s_t = decay_t * s_{t-1}
        - delta_t = (v_t - <s_t, k_t>) * beta_t
        - s_t = s_t + (k_t outer delta_t)
        - o_t = s_t @ q_t

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/
      src/transformers/models/qwen3_next/
"""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass
from functools import lru_cache

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


_MAX_GDR_METAL_HEAD_DIM = 128

_GDR_METAL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""

_GDR_RECURRENT_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int batch = query_shape[0];
    int num_query_heads = query_shape[1];
    int num_heads = value_shape[1];
    int head_dim = query_shape[2];
    int d_state = value_shape[2];
    if (head_dim <= 0 || d_state <= 0) {
        return;
    }
    if ((int)batch_idx >= batch || (int)head_idx >= num_heads) {
        return;
    }
    int expand_ratio = max(num_heads / max(num_query_heads, 1), 1);
    int query_head_idx = min((int)head_idx / expand_ratio, num_query_heads - 1);

    threadgroup float key_shared[128];
    threadgroup float query_shared[128];
    threadgroup float beta_shared;
    threadgroup float decay_shared;

    int q_base = ((int)batch_idx * num_query_heads + query_head_idx) * head_dim;
    int v_base = ((int)batch_idx * num_heads + (int)head_idx) * d_state;
    int state_base = (((int)batch_idx * num_heads + (int)head_idx) * head_dim) * d_state;
    int beta_base = ((int)batch_idx * num_heads + (int)head_idx);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        key_shared[d] = (float)key[q_base + d];
        query_shared[d] = (float)query[q_base + d];
    }
    if (tid == 0) {
        beta_shared = (float)beta[beta_base];
        decay_shared = (float)decay_scale[beta_base];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid >= d_state) {
        return;
    }

    int value_idx = (int)tid;
    float kv_dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float scaled_state = (float)recurrent_state[state_base + d * d_state + value_idx] * decay_shared;
        kv_dot += scaled_state * key_shared[d];
    }

    float delta = ((float)value[v_base + value_idx] - kv_dot) * beta_shared;
    float out_val = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        int state_idx = state_base + d * d_state + value_idx;
        float new_state_val = (float)recurrent_state[state_idx] * decay_shared + key_shared[d] * delta;
        new_recurrent_state[state_idx] = new_state_val;
        out_val += new_state_val * query_shared[d];
    }
    output[v_base + value_idx] = out_val;
"""

_GDR_RECURRENT_FUSED_LOGDECAY_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int batch = query_shape[0];
    int num_query_heads = query_shape[1];
    int num_heads = value_shape[1];
    int head_dim = query_shape[2];
    int d_state = value_shape[2];
    if (head_dim <= 0 || d_state <= 0) {
        return;
    }
    if ((int)batch_idx >= batch || (int)head_idx >= num_heads) {
        return;
    }
    int expand_ratio = max(num_heads / max(num_query_heads, 1), 1);
    int query_head_idx = min((int)head_idx / expand_ratio, num_query_heads - 1);

    threadgroup float key_shared[128];
    threadgroup float query_shared[128];
    threadgroup float query_norm_sums[128];
    threadgroup float key_norm_sums[128];
    threadgroup float beta_shared;
    threadgroup float decay_shared;
    threadgroup float query_scale_shared;
    threadgroup float key_scale_shared;

    int q_base = ((int)batch_idx * num_query_heads + query_head_idx) * head_dim;
    int v_base = ((int)batch_idx * num_heads + (int)head_idx) * d_state;
    int state_base = (((int)batch_idx * num_heads + (int)head_idx) * head_dim) * d_state;
    int beta_base = ((int)batch_idx * num_heads + (int)head_idx);

    float query_sq = 0.0f;
    float key_sq = 0.0f;
    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        float q_val = (float)query[q_base + d];
        float k_val = (float)key[q_base + d];
        query_shared[d] = q_val;
        key_shared[d] = k_val;
        query_sq += q_val * q_val;
        key_sq += k_val * k_val;
    }
    query_norm_sums[tid] = query_sq;
    key_norm_sums[tid] = key_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            query_norm_sums[tid] += query_norm_sums[tid + stride];
            key_norm_sums[tid] += key_norm_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        beta_shared = (float)beta[beta_base];
        decay_shared = exp((float)log_decay[beta_base]);
        query_scale_shared = (float)query_scale[0] * rsqrt(query_norm_sums[0] + 1e-6f);
        key_scale_shared = rsqrt(key_norm_sums[0] + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        query_shared[d] *= query_scale_shared;
        key_shared[d] *= key_scale_shared;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid >= d_state) {
        return;
    }

    int value_idx = (int)tid;
    float kv_dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float scaled_state = (float)recurrent_state[state_base + d * d_state + value_idx] * decay_shared;
        kv_dot += scaled_state * key_shared[d];
    }

    float delta = ((float)value[v_base + value_idx] - kv_dot) * beta_shared;
    float out_val = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        int state_idx = state_base + d * d_state + value_idx;
        float new_state_val = (float)recurrent_state[state_idx] * decay_shared + key_shared[d] * delta;
        new_recurrent_state[state_idx] = new_state_val;
        out_val += new_state_val * query_shared[d];
    }
    output[v_base + value_idx] = out_val;
"""

_GDR_RECURRENT_FUSED_ALPHA_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int batch = query_shape[0];
    int num_query_heads = query_shape[1];
    int num_heads = value_shape[1];
    int head_dim = query_shape[2];
    int d_state = value_shape[2];
    if (head_dim <= 0 || d_state <= 0) {
        return;
    }
    if ((int)batch_idx >= batch || (int)head_idx >= num_heads) {
        return;
    }
    int expand_ratio = max(num_heads / max(num_query_heads, 1), 1);
    int query_head_idx = min((int)head_idx / expand_ratio, num_query_heads - 1);

    threadgroup float key_shared[128];
    threadgroup float query_shared[128];
    threadgroup float query_norm_sums[128];
    threadgroup float key_norm_sums[128];
    threadgroup float beta_shared;
    threadgroup float decay_shared;
    threadgroup float query_scale_shared;
    threadgroup float key_scale_shared;

    int q_base = ((int)batch_idx * num_query_heads + query_head_idx) * head_dim;
    int v_base = ((int)batch_idx * num_heads + (int)head_idx) * d_state;
    int state_base = (((int)batch_idx * num_heads + (int)head_idx) * head_dim) * d_state;
    int beta_base = ((int)batch_idx * num_heads + (int)head_idx);

    float query_sq = 0.0f;
    float key_sq = 0.0f;
    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        float q_val = (float)query[q_base + d];
        float k_val = (float)key[q_base + d];
        query_shared[d] = q_val;
        key_shared[d] = k_val;
        query_sq += q_val * q_val;
        key_sq += k_val * k_val;
    }
    query_norm_sums[tid] = query_sq;
    key_norm_sums[tid] = key_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            query_norm_sums[tid] += query_norm_sums[tid + stride];
            key_norm_sums[tid] += key_norm_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float alpha_val = (float)alpha[beta_base] + (float)dt_bias[(int)head_idx];
        float softplus_val = alpha_val > 20.0f ? alpha_val : log(1.0f + exp(alpha_val));
        beta_shared = (float)beta[beta_base];
        decay_shared = exp((float)decay_base[(int)head_idx] * softplus_val);
        query_scale_shared = (float)query_scale[0] * rsqrt(query_norm_sums[0] + 1e-6f);
        key_scale_shared = rsqrt(key_norm_sums[0] + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
        query_shared[d] *= query_scale_shared;
        key_shared[d] *= key_scale_shared;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid >= d_state) {
        return;
    }

    int value_idx = (int)tid;
    float kv_dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float scaled_state = (float)recurrent_state[state_base + d * d_state + value_idx] * decay_shared;
        kv_dot += scaled_state * key_shared[d];
    }

    float delta = ((float)value[v_base + value_idx] - kv_dot) * beta_shared;
    float out_val = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        int state_idx = state_base + d * d_state + value_idx;
        float new_state_val = (float)recurrent_state[state_idx] * decay_shared + key_shared[d] * delta;
        new_recurrent_state[state_idx] = new_state_val;
        out_val += new_state_val * query_shared[d];
    }
    output[v_base + value_idx] = out_val;
"""

_GDR_RECURRENT_FUSED_ALPHA_BETA_METAL_SOURCE = _GDR_RECURRENT_FUSED_ALPHA_METAL_SOURCE.replace(
    "        beta_shared = (float)beta[beta_base];",
    "        float beta_raw = (float)beta[beta_base];\n        beta_shared = 1.0f / (1.0f + exp(-beta_raw));",
)

_GDR_CHUNKED_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int batch = query_shape[0];
    int seq_len = query_shape[1];
    int num_heads = query_shape[2];
    int head_dim = query_shape[3];
    int d_state = value_shape[3];
    if (head_dim <= 0 || d_state <= 0 || seq_len <= 0) {
        return;
    }
    if ((int)batch_idx >= batch || (int)head_idx >= num_heads) {
        return;
    }

    threadgroup float q_shared[128];
    threadgroup float k_shared[128];

    bool active_lane = (int)tid < d_state;
    int state_base = (((int)batch_idx * num_heads + (int)head_idx) * head_dim) * d_state;
    if (active_lane) {
        int lane = (int)tid;
        for (int d = 0; d < head_dim; ++d) {
            int state_idx = state_base + d * d_state + lane;
            new_recurrent_state[state_idx] = (float)recurrent_state[state_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < seq_len; ++t) {
        int qkv_base = (((int)batch_idx * seq_len + t) * num_heads + (int)head_idx);
        int q_base = qkv_base * head_dim;
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            q_shared[d] = (float)query[q_base + d];
            k_shared[d] = (float)key[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_lane) {
            int lane = (int)tid;
            int value_idx = qkv_base * d_state + lane;
            int beta_idx = ((int)batch_idx * seq_len + t) * num_heads + (int)head_idx;
            float decay = (float)decay_scale[beta_idx];
            float beta_t = (float)beta[beta_idx];

            float kv_dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int state_idx = state_base + d * d_state + lane;
                float scaled_state = (float)new_recurrent_state[state_idx] * decay;
                kv_dot += scaled_state * k_shared[d];
            }

            float delta = ((float)value[value_idx] - kv_dot) * beta_t;
            float out_val = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int state_idx = state_base + d * d_state + lane;
                float new_state_val = (float)new_recurrent_state[state_idx] * decay + k_shared[d] * delta;
                new_recurrent_state[state_idx] = new_state_val;
                out_val += new_state_val * q_shared[d];
            }
            output[value_idx] = out_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_GDR_CHUNKED_GROUPED_METAL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y;
    uint head_idx = threadgroup_position_in_grid.z;
    uint tg_size = threads_per_threadgroup.x;

    int batch = query_shape[0];
    int seq_len = query_shape[1];
    int num_query_heads = query_shape[2];
    int num_heads = value_shape[2];
    int head_dim = query_shape[3];
    int d_state = value_shape[3];
    if (head_dim <= 0 || d_state <= 0 || seq_len <= 0) {
        return;
    }
    if ((int)batch_idx >= batch || (int)head_idx >= num_heads) {
        return;
    }
    int expand_ratio = max(num_heads / max(num_query_heads, 1), 1);
    int query_head_idx = min((int)head_idx / expand_ratio, num_query_heads - 1);

    threadgroup float q_shared[128];
    threadgroup float k_shared[128];

    bool active_lane = (int)tid < d_state;
    int state_base = (((int)batch_idx * num_heads + (int)head_idx) * head_dim) * d_state;
    if (active_lane) {
        int lane = (int)tid;
        for (int d = 0; d < head_dim; ++d) {
            int state_idx = state_base + d * d_state + lane;
            new_recurrent_state[state_idx] = (float)recurrent_state[state_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < seq_len; ++t) {
        int q_base = (((int)batch_idx * seq_len + t) * num_query_heads + query_head_idx) * head_dim;
        for (int d = (int)tid; d < head_dim; d += (int)tg_size) {
            q_shared[d] = (float)query[q_base + d];
            k_shared[d] = (float)key[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_lane) {
            int lane = (int)tid;
            int value_idx = (((int)batch_idx * seq_len + t) * num_heads + (int)head_idx) * d_state + lane;
            int beta_idx = ((int)batch_idx * seq_len + t) * num_heads + (int)head_idx;
            float decay = (float)decay_scale[beta_idx];
            float beta_t = (float)beta[beta_idx];

            float kv_dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int state_idx = state_base + d * d_state + lane;
                float scaled_state = (float)new_recurrent_state[state_idx] * decay;
                kv_dot += scaled_state * k_shared[d];
            }

            float delta = ((float)value[value_idx] - kv_dot) * beta_t;
            float out_val = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int state_idx = state_base + d * d_state + lane;
                float new_state_val = (float)new_recurrent_state[state_idx] * decay + k_shared[d] * delta;
                new_recurrent_state[state_idx] = new_state_val;
                out_val += new_state_val * q_shared[d];
            }
            output[value_idx] = out_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _next_power_of_two(value: int) -> int:
    """Return the next power of two greater than or equal to ``value``."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


_GDR_RECURRENT_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_recurrent",
    input_names=["query", "key", "value", "beta", "decay_scale", "recurrent_state"],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_RECURRENT_METAL_SOURCE,
)

_GDR_RECURRENT_FUSED_LOGDECAY_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_recurrent_fused_logdecay",
    input_names=["query", "key", "value", "beta", "log_decay", "query_scale", "recurrent_state"],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_RECURRENT_FUSED_LOGDECAY_METAL_SOURCE,
)

_GDR_RECURRENT_FUSED_ALPHA_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_recurrent_fused_alpha",
    input_names=[
        "query",
        "key",
        "value",
        "beta",
        "alpha",
        "dt_bias",
        "decay_base",
        "query_scale",
        "recurrent_state",
    ],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_RECURRENT_FUSED_ALPHA_METAL_SOURCE,
)

_GDR_RECURRENT_FUSED_ALPHA_BETA_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_recurrent_fused_alpha_beta",
    input_names=[
        "query",
        "key",
        "value",
        "beta",
        "alpha",
        "dt_bias",
        "decay_base",
        "query_scale",
        "recurrent_state",
    ],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_RECURRENT_FUSED_ALPHA_BETA_METAL_SOURCE,
)

_GDR_CHUNKED_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_chunked",
    input_names=["query", "key", "value", "beta", "decay_scale", "recurrent_state"],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_CHUNKED_METAL_SOURCE,
)

_GDR_CHUNKED_GROUPED_METAL_KERNEL = mx.fast.metal_kernel(
    name="gated_delta_rule_chunked_grouped",
    input_names=["query", "key", "value", "beta", "decay_scale", "recurrent_state"],
    output_names=["output", "new_recurrent_state"],
    header=_GDR_METAL_HEADER,
    source=_GDR_CHUNKED_GROUPED_METAL_SOURCE,
)


def _l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-6) -> mx.array:
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


@lru_cache(maxsize=32)
def _float32_scalar_ptr(value: float) -> mx.array:
    """Return a cached 1-element float32 MLX array."""
    return mx.array([float(value)], dtype=mx.float32)


def _normalize_query_key(
    query: mx.array,
    key: mx.array,
    *,
    use_qk_l2norm: bool,
    query_scale: float | None,
    cast_to_float32: bool,
) -> tuple[mx.array, mx.array]:
    """Normalize and scale query/key tensors in float32."""
    if cast_to_float32:
        query = query.astype(mx.float32)
        key = key.astype(mx.float32)
    if use_qk_l2norm:
        query = _l2_normalize(query, axis=-1)
        key = _l2_normalize(key, axis=-1)
    if query_scale is not None:
        query = query * float(query_scale)
    return query, key


def _apply_beta(delta: mx.array, beta: mx.array) -> mx.array:
    """Apply beta gating to the delta update."""
    beta = beta.astype(mx.float32)
    if beta.ndim == 2:
        return delta * beta[:, :, None]
    if beta.ndim == 3 and beta.shape[-1] == delta.shape[-1]:
        return delta * beta
    raise ValueError("beta must be shaped [batch, heads] or [batch, heads, value_dim].")


def _decay_scale_to_state(
    decay: mx.array | None,
    state: mx.array,
    *,
    decay_is_log: bool,
) -> mx.array | None:
    """Broadcast decay to recurrent-state shape."""
    if decay is None:
        return None

    decay = decay.astype(mx.float32)
    if decay_is_log:
        decay = mx.exp(decay)

    if decay.ndim == 1 and decay.shape[0] == state.shape[1]:
        return decay[None, :, None, None]
    if decay.ndim == 2:
        if decay.shape == (state.shape[0], state.shape[1]):
            return decay[:, :, None, None]
        if decay.shape == (state.shape[1], state.shape[2]):
            return decay[None, :, :, None]
    if decay.ndim == 3 and decay.shape == state.shape[:3]:
        return decay[:, :, :, None]
    raise ValueError("Unsupported decay shape for gated delta rule state update.")


def _scalar_decay_scale_for_metal(
    decay: mx.array | None,
    *,
    batch_size: int,
    num_heads: int,
    decay_is_log: bool,
) -> mx.array | None:
    """Return [batch, heads] decay scales when the recurrent Metal path can use them."""
    if decay is None:
        return mx.ones((batch_size, num_heads), dtype=mx.float32)

    decay = decay.astype(mx.float32)
    if decay_is_log:
        decay = mx.exp(decay)

    if decay.ndim == 1 and decay.shape[0] == num_heads:
        return mx.broadcast_to(decay[None, :], (batch_size, num_heads))
    if decay.ndim == 2 and decay.shape == (batch_size, num_heads):
        return decay
    return None


def _scalar_log_decay_for_metal(
    decay: mx.array | None,
    *,
    batch_size: int,
    num_heads: int,
) -> mx.array | None:
    """Return raw log-decay terms for the fused recurrent Metal path."""
    if decay is None:
        return mx.zeros((batch_size, num_heads), dtype=mx.float32)

    decay = decay.astype(mx.float32)
    if decay.ndim == 1 and decay.shape[0] == num_heads:
        return mx.broadcast_to(decay[None, :], (batch_size, num_heads))
    if decay.ndim == 2 and decay.shape == (batch_size, num_heads):
        return decay
    return None


def _scalar_token_decay_scale_for_metal(
    decay: mx.array | None,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    decay_is_log: bool,
) -> mx.array | None:
    """Return ``[batch, seq_len, heads]`` decay scales for the chunked Metal path."""
    if decay is None:
        return mx.ones((batch_size, seq_len, num_heads), dtype=mx.float32)

    decay = decay.astype(mx.float32)
    if decay_is_log:
        decay = mx.exp(decay)

    if decay.ndim == 1 and decay.shape[0] == num_heads:
        return mx.broadcast_to(decay[None, None, :], (batch_size, seq_len, num_heads))
    if decay.ndim == 2:
        if decay.shape == (batch_size, num_heads):
            return mx.broadcast_to(decay[:, None, :], (batch_size, seq_len, num_heads))
        if decay.shape == (seq_len, num_heads):
            return mx.broadcast_to(decay[None, :, :], (batch_size, seq_len, num_heads))
    if decay.ndim == 3 and decay.shape == (batch_size, seq_len, num_heads):
        return decay
    return None


def _gdr_recurrent_step_metal(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay_scale: mx.array,
    recurrent_state: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
    state_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal recurrent gated-delta update for normalized float32 inputs."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for GDR Metal kernel.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    kernel = _GDR_RECURRENT_METAL_KERNEL
    output_shape = (query.shape[0], value.shape[1], value.shape[2])
    output, new_state = kernel(
        inputs=[query, key, value, beta, decay_scale, recurrent_state],
        grid=(tg_size, int(query.shape[0]), int(value.shape[1])),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[output_shape, recurrent_state.shape],
        output_dtypes=[output_dtype or value.dtype, state_dtype or recurrent_state.dtype],
    )
    return output[:, None, :, :], new_state


def _gdr_recurrent_step_metal_fused_logdecay(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    log_decay: mx.array,
    query_scale: float,
    recurrent_state: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
    state_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal recurrent gated-delta update with fused q/k norm and log-decay exp."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for fused GDR Metal kernel.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    output_shape = (query.shape[0], value.shape[1], value.shape[2])
    output, new_state = _GDR_RECURRENT_FUSED_LOGDECAY_METAL_KERNEL(
        inputs=[query, key, value, beta, log_decay, _float32_scalar_ptr(float(query_scale)), recurrent_state],
        grid=(tg_size, int(query.shape[0]), int(value.shape[1])),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[output_shape, recurrent_state.shape],
        output_dtypes=[output_dtype or value.dtype, state_dtype or recurrent_state.dtype],
    )
    return output[:, None, :, :], new_state


def _gdr_recurrent_step_metal_fused_alpha(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    alpha: mx.array,
    dt_bias: mx.array,
    decay_base: mx.array,
    query_scale: float,
    recurrent_state: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
    state_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal recurrent GDR update with fused Qwen alpha-to-decay transform."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for fused-alpha GDR Metal kernel.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    output_shape = (query.shape[0], value.shape[1], value.shape[2])
    output, new_state = _GDR_RECURRENT_FUSED_ALPHA_METAL_KERNEL(
        inputs=[
            query,
            key,
            value,
            beta,
            alpha,
            dt_bias,
            decay_base,
            _float32_scalar_ptr(float(query_scale)),
            recurrent_state,
        ],
        grid=(tg_size, int(query.shape[0]), int(value.shape[1])),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[output_shape, recurrent_state.shape],
        output_dtypes=[output_dtype or value.dtype, state_dtype or recurrent_state.dtype],
    )
    return output[:, None, :, :], new_state


def _gdr_recurrent_step_metal_fused_alpha_beta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    alpha: mx.array,
    dt_bias: mx.array,
    decay_base: mx.array,
    query_scale: float,
    recurrent_state: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
    state_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Metal recurrent GDR update fusing beta sigmoid and alpha decay."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for fused-alpha-beta GDR Metal kernel.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    output_shape = (query.shape[0], value.shape[1], value.shape[2])
    output, new_state = _GDR_RECURRENT_FUSED_ALPHA_BETA_METAL_KERNEL(
        inputs=[
            query,
            key,
            value,
            beta,
            alpha,
            dt_bias,
            decay_base,
            _float32_scalar_ptr(float(query_scale)),
            recurrent_state,
        ],
        grid=(tg_size, int(query.shape[0]), int(value.shape[1])),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[output_shape, recurrent_state.shape],
        output_dtypes=[output_dtype or value.dtype, state_dtype or recurrent_state.dtype],
    )
    return output[:, None, :, :], new_state


def _gdr_chunked_forward_metal(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay_scale: mx.array,
    recurrent_state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Metal chunked gated-delta update for normalized float32 inputs."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for chunked GDR Metal kernel.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    kernel = _GDR_CHUNKED_METAL_KERNEL
    output_shape = value.shape
    output, new_state = kernel(
        inputs=[query, key, value, beta, decay_scale, recurrent_state],
        grid=(tg_size, int(query.shape[0]), int(query.shape[2])),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[output_shape, recurrent_state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return output, new_state


def _gdr_chunked_forward_metal_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay_scale: mx.array,
    recurrent_state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Metal chunked gated-delta update with grouped Q/K heads."""
    head_dim = int(query.shape[-1])
    d_state = int(value.shape[-1])
    query_heads = int(query.shape[2])
    value_heads = int(value.shape[2])
    if head_dim <= 0 or head_dim > _MAX_GDR_METAL_HEAD_DIM or d_state <= 0 or d_state > _MAX_GDR_METAL_HEAD_DIM:
        raise ValueError("Unsupported head or state dimension for grouped chunked GDR Metal kernel.")
    if query_heads <= 0 or value_heads % query_heads != 0:
        raise ValueError("GDR value heads must be divisible by query heads.")

    tg_size = min(_MAX_GDR_METAL_HEAD_DIM, _next_power_of_two(max(head_dim, d_state)))
    output, new_state = _GDR_CHUNKED_GROUPED_METAL_KERNEL(
        inputs=[query, key, value, beta, decay_scale, recurrent_state],
        grid=(tg_size, int(query.shape[0]), value_heads),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[value.shape, recurrent_state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return output, new_state


def _recurrent_step(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array | None,
    recurrent_state: mx.array | None,
    use_qk_l2norm: bool = True,
    query_scale: float | None = None,
    decay_is_log: bool = False,
    prefer_metal: bool = True,
) -> tuple[mx.array, mx.array]:
    """Single recurrent step for decode (seq_len == 1).

    Computes the gated delta rule update for a single token:
        s_t = decay_t * s_{t-1}
        delta_t = (v_t - <s_t, k_t>) * beta_t
        s_t = s_t + k_t outer delta_t
        o_t = s_t @ q_t

    Args:
        query: Query tensor ``[batch, 1, num_heads, head_dim]``.
        key: Key tensor ``[batch, 1, num_heads, head_dim]``.
        value: Value tensor ``[batch, 1, num_heads, d_state]``.
        beta: Gating tensor ``[batch, 1, num_heads]`` or
            ``[batch, 1, num_heads, d_state]``.
        decay: Optional decay terms. Supports multiplicative decay or
            per-token log-decay depending on ``decay_is_log``.
        recurrent_state: Previous recurrent state
            ``[batch, num_heads, head_dim, d_state]`` or ``None``.
        use_qk_l2norm: Whether to L2-normalize query and key.
        query_scale: Optional scale applied after query normalization.
        decay_is_log: Whether ``decay`` should be exponentiated first.
        prefer_metal: Whether to try the recurrent Metal fast path.

    Returns:
        Tuple of (output ``[batch, 1, num_heads, d_state]``,
        new_state ``[batch, num_heads, head_dim, d_state]``).
    """
    batch, _, _query_heads, head_dim = query.shape
    value_heads = value.shape[2]
    d_state = value.shape[-1]

    if recurrent_state is None:
        recurrent_state = mx.zeros(
            (batch, value_heads, head_dim, d_state),
            dtype=mx.float32,
        )

    q_raw = query[:, 0]
    k_raw = key[:, 0]
    v_raw = value[:, 0]
    b_raw = beta[:, 0]
    decay_t = None
    if decay is not None:
        decay_t = decay[:, 0] if decay.ndim >= 3 else decay

    if prefer_metal and b_raw.ndim == 2:
        if use_qk_l2norm and query_scale is not None and decay_is_log:
            log_decay = _scalar_log_decay_for_metal(
                decay_t,
                batch_size=batch,
                num_heads=value_heads,
            )
            if log_decay is not None:
                try:
                    return _gdr_recurrent_step_metal_fused_logdecay(
                        query=q_raw,
                        key=k_raw,
                        value=v_raw,
                        beta=b_raw,
                        log_decay=log_decay,
                        query_scale=float(query_scale),
                        recurrent_state=recurrent_state,
                        output_dtype=value.dtype,
                        state_dtype=recurrent_state.dtype,
                    )
                except (RuntimeError, ValueError):
                    pass

        q, k = _normalize_query_key(
            q_raw,
            k_raw,
            use_qk_l2norm=use_qk_l2norm,
            query_scale=query_scale,
            cast_to_float32=True,
        )
        decay_scale = _scalar_decay_scale_for_metal(
            decay_t,
            batch_size=batch,
            num_heads=value_heads,
            decay_is_log=decay_is_log,
        )
        if decay_scale is not None:
            try:
                return _gdr_recurrent_step_metal(
                    query=q,
                    key=k,
                    value=v_raw,
                    beta=b_raw,
                    decay_scale=decay_scale,
                    recurrent_state=recurrent_state,
                    output_dtype=value.dtype,
                    state_dtype=recurrent_state.dtype,
                )
            except (RuntimeError, ValueError):
                pass
    else:
        q, k = _normalize_query_key(
            q_raw,
            k_raw,
            use_qk_l2norm=use_qk_l2norm,
            query_scale=query_scale,
            cast_to_float32=True,
        )

    v = v_raw.astype(mx.float32)
    b = b_raw.astype(mx.float32)
    state = recurrent_state.astype(mx.float32)
    if q.shape[1] != value_heads:
        if value_heads % q.shape[1] != 0:
            raise ValueError("GDR value heads must be divisible by query heads.")
        expand_ratio = value_heads // q.shape[1]
        q = mx.repeat(q, expand_ratio, axis=1)
        k = mx.repeat(k, expand_ratio, axis=1)
    decay_scale_state = _decay_scale_to_state(decay_t, state, decay_is_log=decay_is_log)
    scaled_state = state if decay_scale_state is None else state * decay_scale_state
    kv_dot = mx.sum(scaled_state * k[:, :, :, None], axis=-2)
    delta = _apply_beta(v - kv_dot, b)
    new_state = scaled_state + k[:, :, :, None] * delta[:, :, None, :]
    output = mx.sum(new_state * q[:, :, :, None], axis=-2)
    return output[:, None, :, :], new_state


def _chunked_forward(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array | None,
    recurrent_state: mx.array | None,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
    query_scale: float | None = None,
    decay_is_log: bool = False,
    prefer_metal: bool = True,
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
            ``[batch, seq_len, num_heads, d_state]``.
        decay: Optional decay terms. Supports multiplicative decay or
            per-token log-decay depending on ``decay_is_log``.
        recurrent_state: Optional initial state
            ``[batch, num_heads, head_dim, d_state]``.
        use_qk_l2norm: Whether to L2-normalize query and key.
        chunk_size: Number of tokens per chunk (default 64).
        query_scale: Optional scale applied after query normalization.
        decay_is_log: Whether ``decay`` should be exponentiated first.

    Returns:
        Tuple of (output ``[batch, seq_len, num_heads, d_state]``,
        final_state ``[batch, num_heads, head_dim, d_state]``).
    """
    batch, seq_len, query_heads, head_dim = query.shape
    value_heads = value.shape[2]
    d_state = value.shape[-1]

    query, key = _normalize_query_key(
        query,
        key,
        use_qk_l2norm=use_qk_l2norm,
        query_scale=query_scale,
        cast_to_float32=False,
    )

    if recurrent_state is None:
        state = mx.zeros(
            (batch, value_heads, head_dim, d_state),
            dtype=mx.float32,
        )
    else:
        state = recurrent_state.astype(mx.float32)

    if prefer_metal and beta.ndim == 3:
        decay_scale = _scalar_token_decay_scale_for_metal(
            decay,
            batch_size=batch,
            seq_len=seq_len,
            num_heads=value_heads,
            decay_is_log=decay_is_log,
        )
        if decay_scale is not None:
            try:
                if query_heads == value_heads:
                    return _gdr_chunked_forward_metal(
                        query=query.astype(mx.float32),
                        key=key.astype(mx.float32),
                        value=value.astype(mx.float32),
                        beta=beta.astype(mx.float32),
                        decay_scale=decay_scale,
                        recurrent_state=state,
                    )
                return _gdr_chunked_forward_metal_grouped(
                    query=query.astype(mx.float32),
                    key=key.astype(mx.float32),
                    value=value.astype(mx.float32),
                    beta=beta.astype(mx.float32),
                    decay_scale=decay_scale,
                    recurrent_state=state,
                )
            except (RuntimeError, ValueError):
                pass

    if query_heads != value_heads:
        if value_heads % query_heads != 0:
            raise ValueError("GDR value heads must be divisible by query heads.")
        expand_ratio = value_heads // query_heads
        query = mx.repeat(query, expand_ratio, axis=2)
        key = mx.repeat(key, expand_ratio, axis=2)

    output_chunks: list[mx.array] = []
    num_chunks = math.ceil(seq_len / chunk_size)
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_len = end - start

        q_c = query[:, start:end]
        k_c = key[:, start:end]
        v_c = value[:, start:end]
        chunk_outputs: list[mx.array] = []
        for t in range(chunk_len):
            q_t = q_c[:, t].astype(mx.float32)
            k_t = k_c[:, t].astype(mx.float32)
            v_t = v_c[:, t].astype(mx.float32)
            b_t = beta[:, start + t].astype(mx.float32)
            d_t = None
            if decay is not None:
                d_t = decay[:, start + t] if decay.ndim >= 3 else decay

            decay_scale_state = _decay_scale_to_state(d_t, state, decay_is_log=decay_is_log)
            scaled_state = state if decay_scale_state is None else state * decay_scale_state
            kv_dot = mx.sum(scaled_state * k_t[:, :, :, None], axis=-2)
            delta = _apply_beta(v_t - kv_dot, b_t)
            state = scaled_state + k_t[:, :, :, None] * delta[:, :, None, :]
            o_t = mx.sum(state * q_t[:, :, :, None], axis=-2)
            chunk_outputs.append(o_t)

        chunk_out = mx.stack(chunk_outputs, axis=1)
        output_chunks.append(chunk_out)

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

        s_t = decay_t * s_{t-1}
        delta_t = (v_t - <s_t, k_t>) * beta_t
        s_t = s_t + (k_t outer delta_t)
        o_t = s_t @ q_t

    Where:
        - ``beta_t`` is a learned gating signal.
        - ``decay_t`` is an optional forgetting factor.
        - ``k_t outer delta_t`` is the rank-1 state update.

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

    Args:
        metadata: Optional operation metadata for runtime configuration.
        use_metal: Whether to prefer Metal GPU kernels for the gated
            delta rule computation. Defaults to ``True``.
    """

    def __init__(
        self,
        metadata: tp.Any | None = None,
        *,
        use_metal: bool = True,
    ):
        super().__init__(metadata=metadata)
        self.use_metal = bool(use_metal)

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
        query_scale: float | None = None,
        decay_is_log: bool = False,
        prefer_metal: bool | None = None,
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
                ``[batch, seq_len, num_heads, d_state]`` (per-value gate).
            decay: Optional decay terms. Supports per-token scalar decay
                for Qwen-style GDR or multiplicative broadcast decay.
            conv_state: Optional convolution state passed through for
                external conv management ``[batch, d_inner, d_conv]``.
            recurrent_state: Optional recurrent state
                ``[batch, num_heads, head_dim, d_state]``.
            use_qk_l2norm: Whether to L2-normalize query and key
                (default ``True``).
            chunk_size: Chunk size for prefill mode (default 64).
            query_scale: Optional scale applied after query
                normalization.
            decay_is_log: Whether ``decay`` should be exponentiated before
                applying it to the recurrent state.
            prefer_metal: Whether to try the recurrent Metal fast path for
                decode.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            ``GatedDeltaRuleOutput`` containing attention outputs and
            updated states.
        """
        if prefer_metal is None:
            prefer_metal = self.use_metal

        runtime_dtype = mx.float32
        if self.metadata is not None and self.metadata.runtime_dtype is not None:
            runtime_dtype = self.metadata.runtime_dtype

        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)
        beta = beta.astype(runtime_dtype)

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
            outputs, new_recurrent_state = self._step_decode_prepared(
                query=query,
                key=key,
                value=value,
                beta=beta,
                decay=decay,
                recurrent_state=recurrent_state,
                use_qk_l2norm=use_qk_l2norm,
                query_scale=query_scale,
                decay_is_log=decay_is_log,
                prefer_metal=prefer_metal,
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
                query_scale=query_scale,
                decay_is_log=decay_is_log,
                prefer_metal=prefer_metal,
            )

        return GatedDeltaRuleOutput(
            attention_outputs=outputs,
            attention_weights=None,
            conv_state=conv_state,
            recurrent_state=new_recurrent_state,
        )

    @staticmethod
    def _step_decode_prepared(
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        decay: mx.array | None,
        recurrent_state: mx.array | None,
        use_qk_l2norm: bool,
        query_scale: float | None,
        decay_is_log: bool,
        prefer_metal: bool,
    ) -> tuple[mx.array, mx.array]:
        """Run the recurrent decode math assuming inputs are already coerced."""
        if int(query.shape[1]) != 1:
            raise ValueError("step_decode expects seq_len == 1.")
        return _recurrent_step(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            query_scale=query_scale,
            decay_is_log=decay_is_log,
            prefer_metal=prefer_metal,
        )

    def step_decode(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        decay: mx.array | None = None,
        recurrent_state: mx.array | None = None,
        use_qk_l2norm: bool = True,
        query_scale: float | None = None,
        decay_is_log: bool = False,
        prefer_metal: bool | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run the recurrent single-token decode step without output wrapping.

        Args:
            query: Query tensor ``[batch, 1, num_heads, head_dim]``.
            key: Key tensor ``[batch, 1, num_heads, head_dim]``.
            value: Value tensor ``[batch, 1, num_heads, d_state]``.
            beta: Gating tensor ``[batch, 1, num_heads]`` or
                ``[batch, 1, num_heads, d_state]``.
            decay: Optional decay factors for the recurrent state.
            recurrent_state: Previous recurrent state.
            use_qk_l2norm: Whether to L2-normalize query and key.
            query_scale: Optional query scale after normalization.
            decay_is_log: Whether ``decay`` should be exponentiated first.
            prefer_metal: Whether to prefer the Metal recurrent kernel.

        Returns:
            Tuple ``(outputs, recurrent_state)`` for the decode step.
        """
        if prefer_metal is None:
            prefer_metal = self.use_metal

        runtime_dtype = mx.float32
        if self.metadata is not None and self.metadata.runtime_dtype is not None:
            runtime_dtype = self.metadata.runtime_dtype

        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)
        beta = beta.astype(runtime_dtype)

        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta[..., 0]

        if decay is not None:
            decay = decay.astype(runtime_dtype)
            if decay.ndim == 4 and decay.shape[-1] == 1:
                decay = decay[..., 0]

        if recurrent_state is not None:
            recurrent_state = recurrent_state.astype(runtime_dtype)

        return self._step_decode_prepared(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            query_scale=query_scale,
            decay_is_log=decay_is_log,
            prefer_metal=prefer_metal,
        )

    def step_decode_with_alpha(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        alpha: mx.array,
        dt_bias: mx.array,
        decay_base: mx.array,
        recurrent_state: mx.array | None = None,
        use_qk_l2norm: bool = True,
        query_scale: float | None = None,
        beta_is_logits: bool = False,
        prefer_metal: bool | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run Qwen-style recurrent decode with alpha-to-decay fused when possible."""
        if prefer_metal is None:
            prefer_metal = self.use_metal

        runtime_dtype = mx.float32
        if self.metadata is not None and self.metadata.runtime_dtype is not None:
            runtime_dtype = self.metadata.runtime_dtype

        query = query.astype(runtime_dtype)
        key = key.astype(runtime_dtype)
        value = value.astype(runtime_dtype)
        beta = beta.astype(runtime_dtype)
        alpha = alpha.astype(runtime_dtype)
        dt_bias = dt_bias.astype(runtime_dtype)
        decay_base = decay_base.astype(mx.float32)

        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta[..., 0]
        if alpha.ndim == 4 and alpha.shape[-1] == 1:
            alpha = alpha[..., 0]

        batch, _, _, head_dim = query.shape
        value_heads = value.shape[2]
        d_state = value.shape[-1]
        if recurrent_state is None:
            recurrent_state = mx.zeros(
                (batch, value_heads, head_dim, d_state),
                dtype=runtime_dtype,
            )
        else:
            recurrent_state = recurrent_state.astype(runtime_dtype)

        if (
            prefer_metal
            and use_qk_l2norm
            and query_scale is not None
            and int(query.shape[1]) == 1
            and beta.ndim == 3
            and alpha.ndim == 3
        ):
            try:
                kernel = (
                    _gdr_recurrent_step_metal_fused_alpha_beta
                    if beta_is_logits
                    else _gdr_recurrent_step_metal_fused_alpha
                )
                return kernel(
                    query=query[:, 0],
                    key=key[:, 0],
                    value=value[:, 0],
                    beta=beta[:, 0],
                    alpha=alpha[:, 0],
                    dt_bias=dt_bias,
                    decay_base=decay_base,
                    query_scale=float(query_scale),
                    recurrent_state=recurrent_state,
                    output_dtype=value.dtype,
                    state_dtype=recurrent_state.dtype,
                )
            except (RuntimeError, ValueError):
                pass

        if beta_is_logits:
            beta = mx.sigmoid(beta)
        alpha_biased = alpha + dt_bias[None, None, :]
        decay = decay_base * mx.log(1.0 + mx.exp(alpha_biased.astype(mx.float32)))
        return self.step_decode(
            query=query,
            key=key,
            value=value,
            beta=beta,
            decay=decay,
            recurrent_state=recurrent_state,
            use_qk_l2norm=use_qk_l2norm,
            query_scale=query_scale,
            decay_is_log=True,
            prefer_metal=prefer_metal,
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
        query_scale: float | None = None,
        decay_is_log: bool = False,
        prefer_metal: bool | None = None,
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
            query_scale: Optional scale applied after query
                normalization.
            decay_is_log: Whether ``decay`` should be exponentiated before
                applying it to the recurrent state.
            prefer_metal: Whether to try the recurrent Metal fast path for
                decode.
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
            query_scale=query_scale,
            decay_is_log=decay_is_log,
            prefer_metal=prefer_metal,
            **kwargs,
        )


__all__ = (
    "GatedDeltaRuleOp",
    "GatedDeltaRuleOutput",
)
