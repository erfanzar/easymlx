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

"""Utility helpers for schedulers.

Provides small pure functions used by scheduler implementations.
"""

from __future__ import annotations

from ..request import EngineRequest


def clamp_token_budget(total: int, reserved: int) -> int:
    """Compute the available token budget after reserving a safety margin.

    Args:
        total: Total token capacity.
        reserved: Tokens to reserve.

    Returns:
        ``total - reserved``, clamped to >= 0.
    """
    return max(int(total) - int(reserved), 0)


def request_sort_key(request: EngineRequest, *, priority_mode: bool) -> tuple[int, float]:
    """Produce a sort key for ordering requests.

    In priority mode, higher-priority requests sort first (negated
    priority), with ties broken by arrival time.  In FCFS mode, only
    arrival time matters.

    Args:
        request: The request to generate a key for.
        priority_mode: Whether to use priority-based ordering.

    Returns:
        A ``(priority_component, arrival_time)`` tuple suitable for
        ``sorted()`` or ``min()`` comparisons.
    """
    if priority_mode:
        return (-int(request.priority), float(request.arrival_time))
    return (0, float(request.arrival_time))
