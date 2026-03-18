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

"""Vanilla attention operation with explicit attention weights.

Provides a manual matmul-based attention implementation that returns
attention weights alongside the output. Supports GQA, masking, and
attention sink tokens.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx

from .._attention_outputs import AttentionOutput
from .._base_operation import BaseOperation, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, RequirementsBuilder


def _normalize_sinks_shape(
    sinks: mx.ArrayLike,
    num_qheads: int,
    num_kheads: int,
) -> mx.ArrayLike | None:
    """Reshape sink token logits to match the GQA head layout.

    Args:
        sinks: Sink logits tensor (1-D or 2-D) or None.
        num_qheads: Number of query heads.
        num_kheads: Number of key/value heads.

    Returns:
        Reshaped sink logits of shape ``[num_kheads, num_reps, num_sinks]``,
        or None if ``sinks`` is None.

    Raises:
        ValueError: If the sinks tensor has unsupported dimensions or shape.
    """
    if sinks is None:
        return None
    num_reps = num_qheads // num_kheads
    if sinks.ndim == 1:
        if sinks.shape[0] == num_qheads:
            per_head = sinks[:, None]
        elif sinks.shape[0] == num_kheads:
            per_head = mx.repeat(sinks, repeats=num_reps, axis=0)[:, None]
        else:
            per_head = mx.broadcast_to(sinks[None, :], (num_qheads, sinks.shape[0]))
        return per_head.reshape(num_kheads, num_reps, per_head.shape[-1])
    if sinks.ndim == 2:
        if sinks.shape[0] == num_qheads:
            per_head = sinks
        elif sinks.shape[0] == num_kheads:
            per_head = mx.repeat(sinks, repeats=num_reps, axis=0)
        elif sinks.shape[0] == 1:
            per_head = mx.broadcast_to(sinks, (num_qheads, sinks.shape[1]))
        else:
            raise ValueError("Unsupported sinks first dimension.")
        return per_head.reshape(num_kheads, num_reps, per_head.shape[-1])
    raise ValueError("Only 1D or 2D sinks are supported.")


@OperationRegistry.register
class VanillaAttention(BaseOperation):
    """Vanilla attention with explicit matmul computation and weight output.

    Registered under ``"vanilla_attention"`` and ``"vanilla"``. Unlike SDPA,
    this implementation returns the attention weight matrix, which is useful
    for debugging and visualization. Supports GQA, masking, and sink tokens.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names for this operation.

        Returns:
            A tuple of name aliases: ``("vanilla_attention", "vanilla")``.
        """
        return ("vanilla_attention", "vanilla")

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):
        return (
            RequirementsBuilder("vanilla_attention").optional_metadata(MetadataField.MASK, MetadataField.SINKS).build()
        )

    def forward_native(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        scale: float | None = None,
        mask: mx.array | None = None,
        sinks: mx.array | None = None,
        **_: tp.Any,
    ) -> AttentionOutput:
        """Execute vanilla attention with explicit weight computation.

        Args:
            query: Query tensor of shape ``[batch, q_heads, q_len, head_dim]``.
            key: Key tensor of shape ``[batch, kv_heads, kv_len, head_dim]``.
            value: Value tensor of shape ``[batch, kv_heads, kv_len, head_dim]``.
            scale: Softmax scaling factor. Defaults to ``1/sqrt(head_dim)``.
            mask: Optional boolean attention mask.
            sinks: Optional sink token logits for attention sink support.
            **_: Additional keyword arguments (ignored).

        Returns:
            An ``AttentionOutput`` containing attention outputs and weights.
        """
        if scale is None:
            scale = query.shape[-1] ** -0.5

        batch, qhead, qlen, _qdim = query.shape
        _, khead, klen, _kdim = key.shape
        num_reps = qhead // khead
        q = query.reshape(batch, khead, num_reps, qlen, -1) * scale
        k = mx.expand_dims(key, -3).swapaxes(-1, -2)
        v = mx.expand_dims(value, -3)
        weights = mx.matmul(q, k)

        if mask is not None:
            if mask.ndim == 4:
                if mask.shape[1] == 1:
                    mask = mx.broadcast_to(mask, (batch, khead, qlen, klen)).reshape(batch, khead, 1, qlen, klen)
                elif mask.shape[1] == khead:
                    mask = mask.reshape(batch, khead, 1, qlen, klen)
                elif mask.shape[1] == qhead:
                    mask = mask.reshape(batch, khead, num_reps, qlen, klen)
                else:
                    raise ValueError("Unsupported attention mask head dimension.")
            elif mask.ndim == 3:
                mask = mask.reshape(batch, 1, 1, qlen, klen)
            elif mask.ndim == 2:
                mask = mask.reshape(1, 1, 1, qlen, klen)
            else:
                raise ValueError("Unsupported mask shape.")
            weights = mx.where(mask, weights, mx.finfo(weights.dtype).min)

        if sinks is not None:
            sinks = _normalize_sinks_shape(sinks, qhead, khead)
            sinks = mx.broadcast_to(sinks[None, :, :, None, :], (batch, khead, num_reps, qlen, sinks.shape[-1]))
            concat = mx.concatenate([weights, sinks], axis=-1)
            concat = concat - mx.max(concat, axis=-1, keepdims=True)
            concat = mx.softmax(concat, axis=-1)
            weights = concat[..., :klen]
        else:
            weights = mx.softmax(weights, axis=-1)

        outputs = mx.matmul(weights, v).reshape(batch, qhead, qlen, -1)
        return AttentionOutput(
            attention_outputs=outputs,
            attention_weights=weights.reshape(batch, qhead, qlen, klen),
        )


Vanilla = VanillaAttention

__all__ = ("Vanilla", "VanillaAttention")
