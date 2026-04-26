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

"""MLX scaled dot product attention operation.

Wraps ``mx.fast.scaled_dot_product_attention`` as a registered
``BaseOperation`` for use in the easymlx operation framework.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx

from .._attention_outputs import AttentionOutput
from .._base_operation import BaseOperation, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, RequirementsBuilder


@OperationRegistry.register
class ScaledDotProductAttention(BaseOperation):
    """Scaled dot-product attention using MLX's fast SDPA implementation.

    Registered under the names ``"scaled_dot_product_attention"`` and ``"sdpa"``.
    Optionally accepts a mask and sink tokens.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered names for this operation.

        Returns:
            A tuple of name aliases: ``("scaled_dot_product_attention", "sdpa")``.
        """
        return ("scaled_dot_product_attention", "sdpa")

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):
        return (
            RequirementsBuilder("scaled_dot_product_attention")
            .optional_metadata(MetadataField.MASK, MetadataField.SINKS)
            .build()
        )

    def forward_native(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        scale: float | None = None,
        mask: mx.array | str | None = None,
        sinks: mx.array | None = None,
        **_: tp.Any,
    ) -> AttentionOutput:
        """Execute scaled dot-product attention via MLX fast path.

        Args:
            query: Query tensor of shape ``[batch, heads, q_len, head_dim]``.
            key: Key tensor of shape ``[batch, heads, kv_len, head_dim]``.
            value: Value tensor of shape ``[batch, heads, kv_len, head_dim]``.
            scale: Softmax scaling factor. Defaults to ``1/sqrt(head_dim)``.
            mask: Optional attention mask or string sentinel.
            sinks: Optional sink token logits for attention sink support.
            **_: Additional keyword arguments (ignored).

        Returns:
            An ``AttentionOutput`` containing the attention output tensor.
        """
        if scale is None:
            scale = query.shape[-1] ** -0.5
        outputs = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
        return AttentionOutput(attention_outputs=outputs)


__all__ = "ScaledDotProductAttention"
