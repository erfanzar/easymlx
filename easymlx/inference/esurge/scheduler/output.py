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

"""Scheduler output contracts.

Defines the data structures emitted by the scheduler to describe the
work planned for a single engine step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ScheduledRequest:
    """One scheduled slice of work for the runner.

    Represents a single request's contribution to a forward-pass batch,
    carrying the token IDs to process, cache metadata, and whether the
    slice is a prefill or decode step.

    Attributes:
        request_id: Unique request identifier.
        row_index: Row index in the sequence buffer / KV cache.
        token_ids: Token IDs to process in this step.
        is_prefill: ``True`` if this slice is a prefill (prompt) step.
        num_tokens: Number of tokens consumed from the token budget.
        page_ids: KV-cache page IDs assigned to the request.
        cache_group: Name of the cache group.
        prefix_cache_hit: Whether a prefix cache hit was obtained.
    """

    request_id: str
    row_index: int
    token_ids: list["int"]
    is_prefill: bool
    num_tokens: int
    page_ids: list["int"] = field(default_factory=list)
    cache_group: str = "default"
    prefix_cache_hit: bool = False


@dataclass(slots=True)
class SchedulerStepOutput:
    """Full scheduling decision for one engine step.

    Aggregates all scheduled slices, preemption events, and budget
    statistics for a single iteration of the engine loop.

    Attributes:
        scheduled: List of :class:`ScheduledRequest` entries to process.
        preempted_request_ids: Requests that were preempted to free
            resources.
        newly_running_request_ids: Requests promoted from waiting to
            running in this step.
        finished_request_ids: Requests that finished during this step
            (populated by :meth:`update_from_model_output`).
        num_scheduled: Total number of scheduled entries.
        num_waiting: Number of requests still in the waiting queue.
        num_running: Number of requests in the running set.
        token_budget_remaining: Tokens left in the step's budget after
            scheduling.
        decode_only: ``True`` if all scheduled slices are decode steps.
        pending_structured_output_tokens: ``True`` if any scheduled
            request uses structured output.
        timestamp: Wall-clock time when the output was created.
    """

    scheduled: list[ScheduledRequest] = field(default_factory=list)
    preempted_request_ids: list["str"] = field(default_factory=list)
    newly_running_request_ids: list["str"] = field(default_factory=list)
    finished_request_ids: list["str"] = field(default_factory=list)
    num_scheduled: int = 0
    num_waiting: int = 0
    num_running: int = 0
    token_budget_remaining: int = 0
    decode_only: bool = False
    pending_structured_output_tokens: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def request_ids(self) -> list["str"]:
        """Return the request IDs of all scheduled entries.

        Returns:
            List of request ID strings.
        """
        return [entry.request_id for entry in self.scheduled]
