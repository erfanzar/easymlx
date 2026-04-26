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

"""Compatibility re-export for worker-owned auth helpers.

This module provides a stable import path for ``AuthEndpointsMixin``.
The actual implementation lives in ``easymlx.workers.esurge.auth_endpoints``
and is loaded lazily via the package-level ``__getattr__`` in the server
``__init__`` module to avoid circular imports.
"""

from __future__ import annotations

__all__ = "AuthEndpointsMixin"
