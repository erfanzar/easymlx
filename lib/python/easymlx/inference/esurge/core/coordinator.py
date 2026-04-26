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

"""Execution coordination state shared by scheduler/runner orchestration.

Provides :class:`ExecutionCoordinatorState`, a mutable dataclass that tracks
counters and flags for active requests, preemptions, cancellations, and
failures during inference engine operation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ExecutionCoordinatorState:
    """Snapshot and counters for one coordinator lifecycle.

    Tracks the runtime state of the inference engine's execution coordinator
    including request counts, step counts, and error tallies.

    Attributes:
        active_requests: Number of currently active (in-flight) requests.
        queued_requests: Number of requests waiting in the queue.
        paused: Whether the coordinator is currently paused.
        distributed_enabled: Whether distributed execution is enabled.
        multimodal_enabled: Whether multimodal inputs are supported.
        total_steps: Total number of scheduler steps executed.
        last_step_at: Unix timestamp of the most recent step, or ``None``
            if no steps have been executed.
        total_preemptions: Total number of request preemptions.
        total_cancellations: Total number of request cancellations.
        total_failures: Total number of request failures.
        metadata: Arbitrary string metadata for extensibility.
    """

    active_requests: int = 0
    queued_requests: int = 0
    paused: bool = False
    distributed_enabled: bool = False
    multimodal_enabled: bool = False
    total_steps: int = 0
    last_step_at: float | None = None
    total_preemptions: int = 0
    total_cancellations: int = 0
    total_failures: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def mark_step(self) -> None:
        """Record that a scheduler step has been executed.

        Increments :attr:`total_steps` and updates :attr:`last_step_at`
        to the current time.
        """
        self.total_steps += 1
        self.last_step_at = time.time()

    def mark_preemption(self) -> None:
        """Record that a request preemption has occurred."""
        self.total_preemptions += 1

    def mark_cancellation(self) -> None:
        """Record that a request cancellation has occurred."""
        self.total_cancellations += 1

    def mark_failure(self) -> None:
        """Record that a request failure has occurred."""
        self.total_failures += 1

    def snapshot(self) -> dict[str, int | bool | float | None]:
        """Return a dictionary snapshot of the coordinator's current state.

        Returns:
            A dict containing all tracked counters and flags, suitable for
            serialization or logging.
        """
        return {
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "paused": self.paused,
            "distributed_enabled": self.distributed_enabled,
            "multimodal_enabled": self.multimodal_enabled,
            "total_steps": self.total_steps,
            "last_step_at": self.last_step_at,
            "total_preemptions": self.total_preemptions,
            "total_cancellations": self.total_cancellations,
            "total_failures": self.total_failures,
        }
