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

"""Compatibility re-export for worker-owned API-key helpers.

This module re-exports API-key-related classes from
``easymlx.workers.esurge.admin_state`` so that server-side code can
import them without reaching into the worker package directly.

Re-exported symbols:
    AdminState: Central security and API-key management object.
    ApiKeyRecord: Dataclass representing a single API key and its metadata.
    AuditEntry: Dataclass representing a single audit-log entry.
"""

from __future__ import annotations

from easymlx.workers.esurge.admin_state import AdminState, ApiKeyRecord, AuditEntry

__all__ = ("AdminState", "ApiKeyRecord", "AuditEntry")
