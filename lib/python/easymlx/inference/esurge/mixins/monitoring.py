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

"""Monitoring wrappers for the eSurge mixins layout.

Provides a convenience function that delegates to
:meth:`~easymlx.inference.esurge.metrics.MetricsCollector.snapshot`.
"""

from __future__ import annotations

from typing import Any

from ..metrics import MetricsCollector


def get_engine_metrics_snapshot(
    metrics: MetricsCollector,
    *,
    models_loaded: int = 0,
    status: str = "unknown",
    auth_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a point-in-time metrics snapshot from the collector.

    Args:
        metrics: The :class:`MetricsCollector` to query.
        models_loaded: Number of models currently loaded.
        status: Current engine status string.
        auth_stats: Optional authentication statistics dictionary.

    Returns:
        A dictionary of metric values (see
        :meth:`MetricsCollector.snapshot` for the full key set).
    """
    return metrics.snapshot(models_loaded=models_loaded, status=status, auth_stats=auth_stats or {})


__all__ = "get_engine_metrics_snapshot"
