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

"""Async runner result types.

Provides the :class:`AsyncExecutionResult` dataclass that bundles an
:class:`ExecutionRequest` with a :class:`~concurrent.futures.Future`
so that callers can track in-flight asynchronous runner steps.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

from .execution_types import ExecutionRequest, ExecutionResult


@dataclass(slots=True)
class AsyncExecutionResult:
    """Asynchronous runner step payload.

    Wraps the original request, the future that will resolve to an
    :class:`ExecutionResult`, and the wall-clock time at which the
    step was dispatched.

    Attributes:
        request: The :class:`ExecutionRequest` that was submitted.
        future: A future that resolves to the :class:`ExecutionResult`
            once the runner finishes.
        started_at: ``time.perf_counter()`` timestamp when the step was
            submitted.
    """

    request: ExecutionRequest
    future: Future[ExecutionResult]
    started_at: float


__all__ = "AsyncExecutionResult"
