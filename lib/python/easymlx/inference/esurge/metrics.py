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

"""Metrics helpers for easymlx eSurge.

Provides :class:`MetricsCollector`, a thread-safe in-memory counters store
for tracking request throughput, token generation rates, tool executions,
and administrative actions. A module-level singleton is maintained and
exposed through :func:`get_metrics_collector`.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any

LOGGER = logging.getLogger("__name__")


class MetricsCollector:
    """Thread-safe in-memory metrics collector for local eSurge serving.

    All mutating methods acquire an internal :class:`~threading.RLock` so
    they can be called safely from multiple threads (e.g. the scheduler
    thread and HTTP request handlers).

    Args:
        enabled: If ``False``, all recording operations become no-ops and
            :meth:`snapshot` returns zeroed counters.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        """Reset all counters and timestamps to their initial values."""
        now = time.time()
        with self._lock:
            self.start_time = now
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_tokens_generated = 0
            self.total_prompt_tokens = 0
            self.total_generation_seconds = 0.0
            self.active_requests = 0
            self._active_progress: dict[float, dict[str, float | int]] = {}
            self.total_tool_executions = 0
            self.failed_tool_executions = 0
            self.total_admin_actions = 0

    def start_request(self, *, endpoint: str | None = None, model: str | None = None) -> float:
        """Record the start of a new inference request.

        Args:
            endpoint: Optional endpoint name (currently unused but
                reserved for per-endpoint metrics).
            model: Optional model identifier (currently unused but
                reserved for per-model metrics).

        Returns:
            A high-resolution timestamp (via :func:`time.perf_counter`)
            that should be passed to :meth:`finish_request`.
        """
        del endpoint, model
        started_at = time.perf_counter()
        if not self.enabled:
            return started_at
        with self._lock:
            self.total_requests += 1
            self.active_requests += 1
            self._active_progress[started_at] = {
                "started_at": started_at,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        return started_at

    def record_request_progress(
        self,
        started_at: float,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record cumulative token progress for an in-flight request.

        Streaming endpoints call this with the request's cumulative generated
        token count. The collector adds only the newly observed token delta so
        the final ``finish_request`` call cannot double count.
        """
        if not self.enabled:
            return
        prompt_tokens = max(int(prompt_tokens), 0)
        completion_tokens = max(int(completion_tokens), 0)
        now = time.perf_counter()
        with self._lock:
            progress = self._active_progress.get(started_at)
            if progress is None:
                return
            if prompt_tokens:
                progress["prompt_tokens"] = max(int(progress.get("prompt_tokens", 0)), prompt_tokens)
            previous_completion_tokens = int(progress.get("completion_tokens", 0))
            if completion_tokens <= previous_completion_tokens:
                return
            self.total_tokens_generated += completion_tokens - previous_completion_tokens
            progress["completion_tokens"] = completion_tokens
            progress["last_progress_at"] = now

    def finish_request(
        self,
        started_at: float,
        *,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record the completion of an inference request.

        Args:
            started_at: Timestamp returned by :meth:`start_request`.
            success: Whether the request completed successfully.
            prompt_tokens: Number of prompt tokens processed.
            completion_tokens: Number of tokens generated.
        """
        duration = max(time.perf_counter() - started_at, 0.0)
        if not self.enabled:
            return
        with self._lock:
            progress = self._active_progress.pop(started_at, None)
            recorded_completion_tokens = int(progress.get("completion_tokens", 0)) if progress else 0
            self.active_requests = max(self.active_requests - 1, 0)
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            self.total_prompt_tokens += max(int(prompt_tokens), 0)
            self.total_tokens_generated += max(int(completion_tokens) - recorded_completion_tokens, 0)
            self.total_generation_seconds += duration

    def record_tool_execution(self, *, success: bool) -> None:
        """Record a tool-call execution event.

        Args:
            success: Whether the tool call succeeded.
        """
        if not self.enabled:
            return
        with self._lock:
            self.total_tool_executions += 1
            if not success:
                self.failed_tool_executions += 1

    def record_admin_action(self) -> None:
        """Increment the administrative action counter."""
        if not self.enabled:
            return
        with self._lock:
            self.total_admin_actions += 1

    def snapshot(
        self,
        *,
        models_loaded: int,
        status: str,
        auth_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Produce a point-in-time snapshot of all collected metrics.

        Args:
            models_loaded: Number of models currently loaded.
            status: Current engine status string.
            auth_stats: Optional authentication statistics to merge into
                the snapshot.

        Returns:
            A dictionary containing uptime, request counts, token
            throughput, and other operational metrics.
        """
        if not self.enabled:
            return {
                "uptime_seconds": 0.0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens_generated": 0,
                "average_tokens_per_second": 0.0,
                "active_requests": 0,
                "models_loaded": int(models_loaded),
                "status": status,
                "auth_stats": auth_stats or {},
                "tool_executions": 0,
                "failed_tool_executions": 0,
                "admin_actions": 0,
            }

        with self._lock:
            uptime_seconds = max(time.time() - self.start_time, 0.0)
            now = time.perf_counter()
            active_generation_seconds = sum(
                max(now - float(progress.get("started_at", now)), 0.0)
                for progress in self._active_progress.values()
                if int(progress.get("completion_tokens", 0)) > 0
            )
            generation_seconds = self.total_generation_seconds + active_generation_seconds
            average_tps = float(self.total_tokens_generated) / generation_seconds if generation_seconds > 0 else 0.0
            return {
                "uptime_seconds": round(uptime_seconds, 6),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "total_tokens_generated": self.total_tokens_generated,
                "average_tokens_per_second": round(average_tps, 6),
                "active_requests": self.active_requests,
                "models_loaded": int(models_loaded),
                "status": status,
                "auth_stats": auth_stats or {},
                "tool_executions": self.total_tool_executions,
                "failed_tool_executions": self.failed_tool_executions,
                "admin_actions": self.total_admin_actions,
            }


_METRICS = MetricsCollector()


def initialize_metrics(*, enabled: bool = True, reset: bool = False) -> MetricsCollector:
    """Initialize or reinitialize the module-level metrics collector.

    Args:
        enabled: Whether the new collector should be active.
        reset: If ``True``, force a reset even when the enabled state has
            not changed.

    Returns:
        The (possibly new) :class:`MetricsCollector` singleton.
    """
    global _METRICS
    if reset or _METRICS.enabled != enabled:
        _METRICS = MetricsCollector(enabled=enabled)
    elif reset:
        _METRICS.reset()
    return _METRICS


def get_metrics_collector() -> MetricsCollector:
    """Return the module-level metrics collector singleton.

    Returns:
        The current :class:`MetricsCollector` instance.
    """
    return _METRICS


def log_metrics_summary(*, metrics: MetricsCollector | None = None, prefix: str = "easymlx esurge") -> None:
    """Log a one-line summary of the current metrics to the module logger.

    Args:
        metrics: Collector to summarize. Falls back to the module-level
            singleton when ``None``.
        prefix: Prefix label for the log message.
    """
    collector = metrics or _METRICS
    snapshot = collector.snapshot(models_loaded=0, status="unknown", auth_stats={})
    LOGGER.info(
        "%s metrics requests=%s success=%s failed=%s generated_tokens=%s tool_executions=%s",
        prefix,
        snapshot["total_requests"],
        snapshot["successful_requests"],
        snapshot["failed_requests"],
        snapshot["total_tokens_generated"],
        snapshot["tool_executions"],
    )
