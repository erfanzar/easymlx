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

"""Execution manager for runner-based model steps.

Wraps a :class:`ModelRunnerProtocol` and provides both synchronous and
asynchronous (thread-pool-backed) execution of
:class:`ExecutionRequest` objects.  The manager is usable as a context
manager for deterministic shutdown.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from easymlx.workers.loggers import get_logger

from .execution_protocol import ModelRunnerProtocol
from .execution_types import ExecutionRequest, ExecutionResult

logger = get_logger("eSurge-ExecutionManger")


@dataclass(slots=True)
class PendingExecution:
    """Handle to a runner step executing in a worker thread.

    Attributes:
        request: The :class:`ExecutionRequest` that was submitted.
        future: A future that resolves to the :class:`ExecutionResult`.
        started_at: ``time.perf_counter()`` timestamp when submission
            occurred.
    """

    request: ExecutionRequest
    future: Future[ExecutionResult]
    started_at: float

    def done(self) -> bool:
        """Return whether the execution has completed.

        Returns:
            ``True`` if the future is resolved (successfully or with an
            exception).
        """
        return self.future.done()


class ExecutionManager:
    """Coordinate runner execution in sync and async modes.

    Provides :meth:`execute` for blocking calls and :meth:`execute_async`
    / :meth:`collect` for non-blocking dispatch with later collection.

    Example:
        >>> with ExecutionManager(runner) as mgr:
        ...     result = mgr.execute(request)
    """

    def __init__(self, runner: ModelRunnerProtocol, *, max_workers: int = 1):
        """Initialize the execution manager.

        Args:
            runner: An object implementing :class:`ModelRunnerProtocol`.
            max_workers: Maximum number of worker threads for async
                execution.
        """
        self.runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max(int(max_workers), 1), thread_name_prefix="esurge-runner")
        self._closed = False

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a request synchronously on the current thread.

        Args:
            request: The execution request to run.

        Returns:
            The :class:`ExecutionResult` produced by the runner.

        Raises:
            RuntimeError: If the manager has been closed.
        """
        if self._closed:
            raise RuntimeError("ExecutionManager is closed")
        logger.debug("Execution start: step_id=%d", request.step_id)
        started = time.perf_counter()
        result = self.runner.run(request)
        elapsed = time.perf_counter() - started
        logger.debug("Execution complete: step_id=%d, %.3fs", request.step_id, elapsed)
        return result

    def execute_many(self, requests: list[ExecutionRequest]) -> list[ExecutionResult]:
        """Execute multiple requests sequentially.

        Args:
            requests: Ordered list of execution requests.

        Returns:
            Corresponding list of :class:`ExecutionResult` objects.
        """
        return [self.execute(request) for request in requests]

    def execute_async(self, request: ExecutionRequest) -> PendingExecution:
        """Submit a request for asynchronous execution in a worker thread.

        Args:
            request: The execution request to submit.

        Returns:
            A :class:`PendingExecution` handle for later collection.

        Raises:
            RuntimeError: If the manager has been closed.
        """
        if self._closed:
            raise RuntimeError("ExecutionManager is closed")
        started_at = time.perf_counter()
        return PendingExecution(
            request=request,
            future=self._executor.submit(self.runner.run, request),
            started_at=started_at,
        )

    def collect(self, pending: PendingExecution, *, timeout: float | None = None) -> ExecutionResult:
        """Block until a pending execution completes and return its result.

        If the result's ``elapsed_seconds`` field is zero, it is
        back-filled from the pending handle's start time.

        Args:
            pending: The :class:`PendingExecution` to wait on.
            timeout: Optional timeout in seconds.

        Returns:
            The :class:`ExecutionResult` from the runner.

        Raises:
            concurrent.futures.TimeoutError: If the timeout expires.
            Exception: Any exception raised by the runner.
        """
        result = pending.future.result(timeout=timeout)
        if result.elapsed_seconds <= 0.0:
            result.elapsed_seconds = max(time.perf_counter() - pending.started_at, 0.0)
        return result

    def close(self) -> None:
        """Shut down the thread pool and mark the manager as closed.

        Blocks until all running tasks complete.  Subsequent calls to
        :meth:`execute` or :meth:`execute_async` will raise
        ``RuntimeError``.
        """
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._closed = True

    def __enter__(self) -> ExecutionManager:
        """Enter the context manager.

        Returns:
            This :class:`ExecutionManager` instance.
        """
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Exit the context manager, closing the thread pool."""
        self.close()


__all__ = ("ExecutionManager", "PendingExecution")
