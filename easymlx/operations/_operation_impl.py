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

"""Operation implementation layer with attention utilities.

Mirrors EasyDeL's ``OperationImpl`` which sits between ``BaseOperation`` and
concrete kernels, providing GQA head repetition, mask manipulation, and
mode detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ._base_operation import BaseOperation
from ._operation_meta import OperationMetadata
from .requirements import ExecutionMode, OperationRequirements


@dataclass(slots=True)
class OperationOutput:
    """Base output container for all operations.

    Attributes:
        output: The primary output tensor, or None if no output was produced.
    """

    output: mx.array | None = None


class OperationImpl(BaseOperation):
    """Extended base that adds mask utilities, GQA helpers, and mode detection.

    Concrete attention kernels can inherit from this instead of raw
    ``BaseOperation`` to get these utilities for free.
    """

    def __init__(self, metadata: OperationMetadata | None = None) -> None:
        """Initialize the operation implementation.

        Args:
            metadata: Optional operation metadata for runtime configuration.
        """
        super().__init__(metadata=metadata)

    def get_impl_metadata(self) -> OperationMetadata | None:
        """Return the metadata associated with this operation instance.

        Returns:
            The ``OperationMetadata`` if set, otherwise None.
        """
        return self.metadata

    def get_instance_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Return requirements for this specific instance.

        Args:
            mode: The execution mode.

        Returns:
            An ``OperationRequirements`` instance.
        """
        return self.get_requirements(mode)

    @staticmethod
    def get_mode(query: mx.array, *, bthd: bool = True) -> ExecutionMode:
        """Determine execution mode from query shape.

        Parameters
        ----------
        query : mx.array
            Query tensor.
        bthd : bool
            If ``True`` (default), layout is ``[batch, seq, heads, dim]``;
            otherwise ``[batch, heads, seq, dim]``.
        """
        seq_len = query.shape["seq_dim"]
        return ExecutionMode.DECODE if seq_len == 1 else ExecutionMode.PREFILL

    @staticmethod
    def repeat_kv_heads(
        key: mx.array,
        value: mx.array,
        num_reps: int,
    ) -> tuple[mx.array, mx.array]:
        """Repeat KV heads to match the number of query heads (GQA → MHA).

        Parameters
        ----------
        key, value : mx.array
            Shape ``[batch, seq, num_kv_heads, head_dim]``.
        num_reps : int
            ``num_q_heads // num_kv_heads``.  If 1 this is a no-op.

        Returns
        -------
        key, value : mx.array
            Shape ``[batch, seq, num_q_heads, head_dim]``.
        """
        if num_reps <= 1:
            return key, value

        _batch, _seq, _kv_heads, _head_dim = key.shape
        key = mx.repeat(key, repeats=num_reps, axis=2)
        value = mx.repeat(value, repeats=num_reps, axis=2)
        return key, value

    @staticmethod
    def create_causal_mask(seq_len: int, *, dtype: mx.Dtype = mx.bool_) -> mx.array:
        """Lower-triangular causal mask ``[seq_len, seq_len]``."""
        idx = mx.arange(seq_len)
        mask = idx[:, None] >= idx[None, :]
        return mask.astype(dtype)

    @staticmethod
    def split_attention_mask(
        mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Split a 2-D attention mask into query and key-value masks.

        Parameters
        ----------
        mask : mx.array
            Shape ``[batch, seq_len]``.

        Returns
        -------
        q_mask, kv_mask : mx.array
            Both shape ``[batch, seq_len]``.
        """
        return mask, mask

    @staticmethod
    def combine_query_kv_masks(
        q_mask: mx.array,
        kv_mask: mx.array,
    ) -> mx.array:
        """Combine separate Q / KV masks into a full 4-D attention mask.

        Parameters
        ----------
        q_mask : mx.array
            Shape ``[batch, q_len]``.
        kv_mask : mx.array
            Shape ``[batch, kv_len]``.

        Returns
        -------
        mask : mx.array
            Shape ``[batch, 1, q_len, kv_len]``.
        """
        return q_mask[:, None, :, None] & kv_mask[:, None, None, :]

    @staticmethod
    def expand_mask_to_4d(
        mask: mx.array,
        *,
        num_heads: int = 1,
    ) -> mx.array:
        """Expand a 2-D or 3-D mask to ``[batch, num_heads, q_len, kv_len]``.

        Parameters
        ----------
        mask : mx.array
            - 2-D ``[batch, kv_len]``: token-level mask, broadcast over Q.
            - 3-D ``[batch, q_len, kv_len]``: per-query mask.
            - 4-D: returned as-is.
        num_heads : int
            Number of attention heads for broadcasting.
        """
        if mask.ndim == 4:
            return mask
        if mask.ndim == 2:
            return mask[:, None, None, :]
        if mask.ndim == 3:
            return mask[:, None, :, :]
        raise ValueError(f"Unsupported mask ndim={mask.ndim}")


__all__ = ("OperationImpl", "OperationOutput")
