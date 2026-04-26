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

"""Compatibility re-export for worker-owned security state.

This module re-exports the core admin-state classes from
``easymlx.workers.esurge.admin_state`` so that consumers within the
``inference.esurge.server`` package can import them via a shorter path.

Re-exported symbols:
    AdminState: Central security and API-key management object.
    ApiKeyRecord: Dataclass representing a single API key and its metadata.
    AuditEntry: Dataclass representing a single audit-log entry.
    AuthorizationError: Raised when a request fails authorization.
    QuotaExceededError: Raised when a key exceeds its quota limits.
    RateLimitError: Raised when a key exceeds its rate limits.
"""

from __future__ import annotations

from easymlx.workers.esurge.admin_state import (
    AdminState,
    ApiKeyRecord,
    AuditEntry,
    AuthorizationError,
    QuotaExceededError,
    RateLimitError,
)

__all__ = ("AdminState", "ApiKeyRecord", "AuditEntry", "AuthorizationError", "QuotaExceededError", "RateLimitError")
