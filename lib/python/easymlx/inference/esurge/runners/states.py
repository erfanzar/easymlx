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

"""Runner state snapshots.

Lightweight dataclasses used to report the current state of the runner
for diagnostic dashboards and health checks without exposing mutable
internal data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RowPlacement:
    """Current row placement for one request.

    Attributes:
        request_id: Unique request identifier.
        row_index: Row index in the sequence buffer / KV cache.
        num_tokens: Total token count (prompt + generated).
        num_computed_tokens: Tokens whose KV entries have been computed.
        page_ids: KV-cache page indices assigned to this row.
    """

    request_id: str
    row_index: int
    num_tokens: int
    num_computed_tokens: int
    page_ids: list["int"] = field(default_factory=list)


@dataclass(slots=True)
class RunnerState:
    """High-level runner state for lifecycle and diagnostics.

    Attributes:
        busy: Whether the runner is currently executing a step.
        active_requests: Number of requests bound to the runner.
        rows: Detailed per-request row placement information.
    """

    busy: bool = False
    active_requests: int = 0
    rows: list[RowPlacement] = field(default_factory=list)


__all__ = ("RowPlacement", "RunnerState")
