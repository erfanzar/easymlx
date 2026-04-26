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

"""Dashboard-oriented helpers for easymlx eSurge.

Aggregates monitoring snapshots into a compact payload suitable for serving
through a dashboard or health-check endpoint.
"""

from __future__ import annotations

from typing import Any

from .metrics import MetricsCollector
from .monitoring import get_monitoring_snapshot


def build_dashboard_state(
    *,
    metrics: MetricsCollector | None = None,
    models_loaded: int = 0,
    status: str = "unknown",
    auth_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact dashboard payload from the current monitoring state.

    Args:
        metrics: Optional :class:`MetricsCollector` instance to pull live
            counters from. If ``None``, a zero-valued snapshot is produced.
        models_loaded: Number of models currently loaded in the engine.
        status: Current engine status string (e.g. ``"ready"``,
            ``"loading"``, ``"unknown"``).
        auth_stats: Optional dictionary of authentication-related
            statistics to include in the snapshot.

    Returns:
        A dictionary with keys ``"summary"`` (the full monitoring
        snapshot) and ``"ready"`` (boolean indicating whether the engine
        status is ``"ready"``).
    """

    snapshot = get_monitoring_snapshot(
        metrics=metrics,
        models_loaded=models_loaded,
        status=status,
        auth_stats=auth_stats,
    )
    return {
        "summary": snapshot,
        "ready": status == "ready",
    }


__all__ = "build_dashboard_state"
