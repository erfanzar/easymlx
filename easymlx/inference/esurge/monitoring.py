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

"""Monitoring helpers mirroring the EasyDeL eSurge layout.

This module provides convenience functions for collecting and returning
point-in-time monitoring snapshots suitable for dashboards, health-check
endpoints, or logging pipelines. It re-exports the core metrics primitives
from :mod:`easymlx.inference.esurge.metrics`.
"""

from __future__ import annotations

from typing import Any

from .metrics import MetricsCollector, get_metrics_collector, initialize_metrics, log_metrics_summary


def get_monitoring_snapshot(
    *,
    metrics: MetricsCollector | None = None,
    models_loaded: int = 0,
    status: str = "unknown",
    auth_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a point-in-time monitoring snapshot for dashboards or APIs.

    Aggregates current metrics, model loading state, server status, and
    optional authentication statistics into a single dictionary.

    Args:
        metrics: An existing ``MetricsCollector`` instance to query. When
            ``None`` the global collector returned by
            :func:`get_metrics_collector` is used.
        models_loaded: Number of models currently loaded in the runtime.
        status: Human-readable server status string (e.g. ``"running"``).
        auth_stats: Optional dictionary of authentication-related counters
            (e.g. total logins, failed attempts).

    Returns:
        A dictionary containing the merged monitoring snapshot as returned
        by :meth:`MetricsCollector.snapshot`.

    Example:
        >>> snapshot = get_monitoring_snapshot(models_loaded=1, status="running")
        >>> snapshot["status"]
        'running'
    """

    collector = metrics or get_metrics_collector()
    return collector.snapshot(models_loaded=models_loaded, status=status, auth_stats=auth_stats or {})


__all__ = (
    "MetricsCollector",
    "get_metrics_collector",
    "get_monitoring_snapshot",
    "initialize_metrics",
    "log_metrics_summary",
)
