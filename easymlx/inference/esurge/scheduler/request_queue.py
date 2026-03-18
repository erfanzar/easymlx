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

"""Request queue primitives for schedulers.

Provides :class:`RequestQueue`, a queue wrapper that supports both
first-come-first-served (FCFS) and stable priority scheduling policies.
The FCFS mode is backed by a :class:`~collections.deque` and the
priority mode by a min-heap with lazy deletion.
"""

from __future__ import annotations

import enum
import heapq
from collections import deque
from dataclasses import dataclass, field

from ..request import EngineRequest


class SchedulingPolicy(enum.StrEnum):
    """Scheduling policy for the request queue.

    Members:
        FCFS: First-come, first-served ordering.
        PRIORITY: Priority-based ordering with arrival-time tiebreaker.
    """

    FCFS = "fcfs"
    PRIORITY = "priority"


@dataclass(order=True, slots=True)
class _PriorityEntry:
    """Internal heap entry wrapping a request with a sort key.

    Attributes:
        sort_key: Tuple of ``(-priority, arrival_time, sequence_number)``
            used for heap ordering.
        request_id: The request's unique identifier (excluded from
            ordering).
        request: The :class:`EngineRequest` object (excluded from
            ordering).
    """

    sort_key: tuple[int, float, int]
    request_id: str = field(compare=False)
    request: EngineRequest = field(compare=False)


class RequestQueue:
    """Queue wrapper supporting FCFS and stable priority scheduling.

    In FCFS mode, requests are served in arrival order via a deque.  In
    priority mode, a min-heap with lazy deletion provides
    ``O(log n)`` push and pop operations while maintaining insertion-
    order stability among equal-priority requests.

    Attributes:
        policy: The active scheduling policy.
    """

    def __init__(self, policy: SchedulingPolicy):
        """Initialize the request queue.

        Args:
            policy: The scheduling policy to use.
        """
        self.policy = SchedulingPolicy(policy)
        self._deque: deque[EngineRequest] = deque()
        self._heap: list["_PriorityEntry"] = []
        self._removed: set["str"] = set()
        self._sequence = 0

    def push(self, request: EngineRequest) -> None:
        """Add a request to the back of the queue.

        Args:
            request: The :class:`EngineRequest` to enqueue.
        """
        if self.policy == SchedulingPolicy.FCFS:
            self._deque.append(request)
            return
        entry = _PriorityEntry(
            sort_key=(-int(request.priority), float(request.arrival_time), self._sequence),
            request_id=request.request_id,
            request=request,
        )
        self._sequence += 1
        heapq.heappush(self._heap, entry)
        self._removed.discard(request.request_id)

    def push_front(self, request: EngineRequest) -> None:
        """Add a request to the front of the queue (high priority re-insertion).

        In FCFS mode the request is prepended.  In priority mode the
        arrival time is minimized to ensure the request is served first.

        Args:
            request: The :class:`EngineRequest` to enqueue at the front.
        """
        if self.policy == SchedulingPolicy.FCFS:
            self._deque.appendleft(request)
            return
        request.arrival_time = min(request.arrival_time, 0.0)
        self.push(request)

    def _pop_priority(self) -> EngineRequest:
        """Pop the highest-priority request from the heap.

        Skips lazily-deleted entries.

        Returns:
            The next :class:`EngineRequest`.

        Raises:
            IndexError: If the queue is empty.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.request_id in self._removed:
                self._removed.discard(entry.request_id)
                continue
            return entry.request
        raise IndexError("pop from empty queue")

    def pop(self) -> EngineRequest:
        """Remove and return the next request according to the active policy.

        Returns:
            The next :class:`EngineRequest`.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.policy == SchedulingPolicy.FCFS:
            return self._deque.popleft()
        return self._pop_priority()

    def peek(self) -> EngineRequest:
        """Return the next request without removing it.

        Returns:
            The next :class:`EngineRequest`.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.policy == SchedulingPolicy.FCFS:
            if not self._deque:
                raise IndexError("peek from empty queue")
            return self._deque[0]
        while self._heap:
            entry = self._heap[0]
            if entry.request_id in self._removed:
                heapq.heappop(self._heap)
                self._removed.discard(entry.request_id)
                continue
            return entry.request
        raise IndexError("peek from empty queue")

    def remove(self, request_id: str) -> bool:
        """Remove a request by ID (lazy deletion in priority mode).

        Args:
            request_id: The request to remove.

        Returns:
            ``True`` if the request was found and marked for removal.
        """
        if self.policy == SchedulingPolicy.FCFS:
            for index, request in enumerate(self._deque):
                if request.request_id == request_id:
                    del self._deque[index]
                    return True
            return False
        for entry in self._heap:
            if entry.request_id == request_id:
                self._removed.add(request_id)
                return True
        return False

    def __len__(self) -> int:
        """Return the number of live entries in the queue.

        Returns:
            Queue length, excluding lazily-deleted entries.
        """
        if self.policy == SchedulingPolicy.FCFS:
            return len(self._deque)
        return max(len(self._heap) - len(self._removed), 0)

    def __bool__(self) -> bool:
        """Return whether the queue contains any live entries.

        Returns:
            ``True`` if the queue is non-empty.
        """
        return len(self) > 0

    def snapshot_ids(self) -> list["str"]:
        """Return the request IDs in current queue order.

        Returns:
            Ordered list of request ID strings.
        """
        if self.policy == SchedulingPolicy.FCFS:
            return [request.request_id for request in self._deque]
        return [entry.request_id for entry in sorted(self._heap) if entry.request_id not in self._removed]


def create_request_queue(policy: SchedulingPolicy | str) -> RequestQueue:
    """Factory function to create a :class:`RequestQueue` from a policy name.

    Args:
        policy: A :class:`SchedulingPolicy` member or its string value
            (e.g. ``"fcfs"`` or ``"priority"``).

    Returns:
        A new :class:`RequestQueue` instance.
    """
    return RequestQueue(policy=SchedulingPolicy(policy))
