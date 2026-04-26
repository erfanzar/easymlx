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

"""Mode-aware operation executor.

Provides ``OperationExecutor``, which holds separate operation implementations
for prefill, decode, and mixed modes and dispatches to the appropriate one
based on the current execution mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._base_operation import BaseOperation
from .requirements import ExecutionMode, OperationRequirements


@dataclass(slots=True)
class OperationExecutor:
    """Mode-aware executor that dispatches to the appropriate operation.

    Holds optional implementations for prefill, decode, and mixed (mixin)
    modes. Falls back through the chain: mode-specific -> prefill -> mixin.

    Attributes:
        prefill_impl: Operation used during prefill (prompt processing).
        decode_impl: Operation used during autoregressive decoding.
        mixin_impl: Fallback operation used when no mode-specific impl exists.
    """

    prefill_impl: BaseOperation | None = None
    decode_impl: BaseOperation | None = None
    mixin_impl: BaseOperation | None = None

    @property
    def prefill_operation(self) -> BaseOperation | None:
        """Return the operation for prefill mode, falling back to mixin."""
        return self.prefill_impl or self.mixin_impl

    @property
    def decode_operation(self) -> BaseOperation | None:
        """Return the operation for decode mode, falling back to prefill then mixin."""
        return self.decode_impl or self.prefill_impl or self.mixin_impl

    def get_operation(self, mode: ExecutionMode) -> BaseOperation | None:
        """Select the operation for the given execution mode.

        Args:
            mode: The current execution mode.

        Returns:
            The appropriate operation, or None if no implementation is available.
        """
        if mode == ExecutionMode.DECODE:
            return self.decode_operation
        return self.prefill_operation

    def get_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Return requirements for the operation in the given mode.

        Args:
            mode: The execution mode.

        Returns:
            An ``OperationRequirements`` for the selected operation, or
            a default if no operation is available.
        """
        op = self.get_operation(mode)
        if op is None:
            return OperationRequirements.default()
        return op.get_instance_requirements(mode)

    def get_combined_requirements(self) -> OperationRequirements:
        """Merge requirements from both prefill and decode operations.

        Returns:
            A combined ``OperationRequirements`` that satisfies both
            prefill and decode needs.
        """
        prefill = self.prefill_operation
        decode = self.decode_operation
        if prefill is None and decode is None:
            return OperationRequirements.default()
        if prefill is None:
            return decode.get_instance_requirements(ExecutionMode.DECODE)
        if decode is None:
            return prefill.get_instance_requirements(ExecutionMode.PREFILL)
        return prefill.get_instance_requirements(ExecutionMode.PREFILL) | decode.get_instance_requirements(
            ExecutionMode.DECODE
        )

    @classmethod
    def from_operations(
        cls,
        *,
        prefill: BaseOperation | None = None,
        decode: BaseOperation | None = None,
        mixin: BaseOperation | None = None,
    ) -> OperationExecutor:
        """Create an executor from named operation instances.

        Args:
            prefill: Operation for prefill mode.
            decode: Operation for decode mode.
            mixin: Fallback operation for any mode.

        Returns:
            A configured ``OperationExecutor`` instance.
        """
        return cls(prefill_impl=prefill, decode_impl=decode, mixin_impl=mixin)


__all__ = "OperationExecutor"
