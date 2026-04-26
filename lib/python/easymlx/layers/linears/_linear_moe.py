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

"""Switch-style MoE layers adapted for MLX (serving-only).

Provides expert-routed linear layers and feed-forward blocks using MLX's
``gather_mm`` and ``gather_qmm`` primitives for efficient sparse
computation on Apple Silicon.
"""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Sort tokens by expert index for contiguous expert processing.

    Rearranges the input tokens so that tokens assigned to the same
    expert are contiguous in memory, enabling more efficient batched
    computation.

    Args:
        x: Input tensor of shape ``[..., M, D]``.
        indices: Expert assignment indices of shape ``[..., M]``.

    Returns:
        A tuple of ``(sorted_x, sorted_indices, inverse_order)`` where
        *inverse_order* can be used with :func:`_scatter_unsort` to
        restore the original ordering.
    """
    *_, m = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // m], indices[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape: tuple[int, ...] | None = None) -> mx.array:
    """Restore the original token ordering after expert processing.

    Args:
        x: The sorted output tensor.
        inv_order: The inverse permutation from :func:`_gather_sort`.
        shape: If provided, unflatten the first dimension back to this
            shape.

    Returns:
        The tensor with tokens restored to their original order.
    """
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    """Quantized expert-routed linear layer.

    Stores expert weights in quantized format and uses
    ``mx.gather_qmm`` for efficient inference. This layer is typically
    created via :meth:`SwitchLinear.to_quantized`.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        num_experts: Number of experts.
        bias: Whether to include a bias term.
        group_size: Quantization group size.
        bits: Number of quantization bits.
        mode: Quantization mode (e.g. ``"affine"``).
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        """Initialize the quantized switch linear layer.

        Args:
            input_dims: Input feature dimension per token.
            output_dims: Output feature dimension per token.
            num_experts: Number of experts in the layer.
            bias: If ``True``, add a learnable bias per expert.
            group_size: Group size for quantization.
            bits: Number of bits for quantization.
            mode: Quantization mode string.
        """
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.freeze()

    @property
    def input_dims(self) -> int:
        """The input feature dimension.

        Returns:
            The input dimension inferred from the quantization scales.
        """
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self) -> int:
        """The output feature dimension.

        Returns:
            The output dimension from the weight shape.
        """
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        """The number of experts.

        Returns:
            The expert count from the weight shape.
        """
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        """Apply the quantized expert linear transformation.

        Args:
            x: Input tensor of shape ``[..., input_dims]``.
            indices: Expert indices of shape ``[...]`` selecting which
                expert to apply to each input.
            sorted_indices: If ``True``, the indices are already sorted
                by expert, enabling kernel optimizations.

        Returns:
            Output tensor of shape ``[..., output_dims]``.
        """
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    """Full-precision expert-routed linear layer.

    Each expert has its own weight matrix. The forward pass uses
    ``mx.gather_mm`` to efficiently dispatch tokens to their assigned
    experts.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        num_experts: Number of experts.
        bias: Whether to include a bias term.
    """

    def __init__(self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True):
        """Initialize the switch linear layer.

        Args:
            input_dims: Input feature dimension per token.
            output_dims: Output feature dimension per token.
            num_experts: Number of experts in the layer.
            bias: If ``True``, add a learnable bias per expert.
        """
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self) -> int:
        """The input feature dimension.

        Returns:
            The input dimension from the weight shape.
        """
        return self.weight.shape[2]

    @property
    def output_dims(self) -> int:
        """The output feature dimension.

        Returns:
            The output dimension from the weight shape.
        """
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        """The number of experts.

        Returns:
            The expert count from the weight shape.
        """
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False) -> mx.array:
        """Apply the expert linear transformation.

        Args:
            x: Input tensor of shape ``[..., input_dims]``.
            indices: Expert indices of shape ``[...]`` selecting which
                expert to apply to each input.
            sorted_indices: If ``True``, the indices are already sorted
                by expert, enabling kernel optimizations.

        Returns:
            Output tensor of shape ``[..., output_dims]``.
        """
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine") -> QuantizedSwitchLinear:
        """Convert this layer to a quantized version.

        Args:
            group_size: Quantization group size.
            bits: Number of quantization bits.
            mode: Quantization mode.

        Returns:
            A :class:`QuantizedSwitchLinear` with quantized weights
            copied from this layer.
        """
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(self.weight, group_size, bits, mode=mode)
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


@partial(mx.compile, shapeless=True)
def swiglu(x: mx.array, gate: mx.array) -> mx.array:
    """Compute SwiGLU activation: ``silu(gate) * x``.

    This function is compiled with ``mx.compile`` for performance.

    Args:
        x: The input tensor.
        gate: The gate tensor (same shape as *x*).

    Returns:
        Element-wise ``silu(gate) * x``.
    """
    return nn.silu(gate) * x


class SwiGLU(nn.Module):
    """SwiGLU activation module wrapping the compiled :func:`swiglu`.

    Computes ``silu(gate) * x`` element-wise.
    """

    def __init__(self):
        """Initialize the SwiGLU module."""
        super().__init__()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        """Apply SwiGLU activation.

        Args:
            x: The input tensor.
            gate: The gate tensor (same shape as *x*).

        Returns:
            Element-wise ``silu(gate) * x``.
        """
        return swiglu(x, gate)


class SwitchGLU(nn.Module):
    """GLU-based expert feed-forward block.

    Implements the gated linear unit pattern with expert-routed
    projections: ``down_proj(activation(up_proj(x), gate_proj(x)))``.

    Args:
        input_dims: Input/output feature dimension.
        hidden_dims: Intermediate (hidden) dimension.
        num_experts: Number of experts.
        activation: Gated activation module.
        bias: Whether to use bias in expert linear layers.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: nn.Module = SwiGLU(),
        bias: bool = False,
    ):
        """Initialize the SwitchGLU block.

        Args:
            input_dims: Input and output feature dimension.
            hidden_dims: Intermediate feature dimension.
            num_experts: Number of experts.
            activation: The gated activation module (default:
                :class:`SwiGLU`).
            bias: If ``True``, include bias in expert linear layers.
        """
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Apply the gated expert feed-forward transformation.

        Tokens are optionally sorted by expert assignment for improved
        memory access patterns when the number of tokens exceeds 64.

        Args:
            x: Input tensor of shape ``[..., input_dims]``.
            indices: Expert indices of shape ``[..., top_k]``.

        Returns:
            Output tensor of shape ``[..., input_dims]``.
        """
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    """Standard expert feed-forward block.

    Implements a two-layer MLP with expert-routed projections:
    ``fc2(activation(fc1(x)))``.

    Args:
        input_dims: Input/output feature dimension.
        hidden_dims: Intermediate (hidden) dimension.
        num_experts: Number of experts.
        activation: Activation module applied after the first linear.
        bias: Whether to use bias in expert linear layers.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: nn.Module = nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        """Initialize the SwitchMLP block.

        Args:
            input_dims: Input and output feature dimension.
            hidden_dims: Intermediate feature dimension.
            num_experts: Number of experts.
            activation: The activation module (default: GELU).
            bias: If ``True``, include bias in expert linear layers.
        """
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """Apply the expert MLP transformation.

        Tokens are optionally sorted by expert assignment for improved
        memory access patterns when the number of tokens exceeds 64.

        Args:
            x: Input tensor of shape ``[..., input_dims]``.
            indices: Expert indices of shape ``[..., top_k]``.

        Returns:
            Output tensor of shape ``[..., input_dims]``.
        """
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


__all__ = ("QuantizedSwitchLinear", "SwitchGLU", "SwitchLinear", "SwitchMLP")
