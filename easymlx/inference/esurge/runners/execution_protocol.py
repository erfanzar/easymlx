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

"""Execution protocols for local model runners.

Defines the :class:`ModelRunnerProtocol` structural typing interface
that any runner implementation must satisfy in order to be used with
:class:`~easymlx.inference.esurge.runners.execution_manager.ExecutionManager`.
"""

from __future__ import annotations

from typing import Protocol

from .execution_types import ExecutionRequest, ExecutionResult


class ModelRunnerProtocol(Protocol):
    """Protocol for runner objects that can execute a request.

    Any class that implements a ``run`` method accepting an
    :class:`ExecutionRequest` and returning an :class:`ExecutionResult`
    is compatible with this protocol.
    """

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a single scheduling step.

        Args:
            request: The execution request describing sequences and
                sampling parameters for this step.

        Returns:
            An :class:`ExecutionResult` containing per-sequence updates
            and optional logits.
        """
        ...


__all__ = "ModelRunnerProtocol"
