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

"""Cache metadata builders.

Provides lightweight dataclass containers for cache-level metadata used
during cache allocation and memory budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheMetadata:
    """Metadata describing cache allocation parameters.

    Used by cache builders and memory budgeting utilities to communicate
    high-level constraints such as maximum sequence length and page size.

    Attributes:
        max_length: Maximum sequence length the cache should support. ``None``
            indicates no explicit limit.
        page_size: Number of tokens per page for paged caches. ``None`` when
            paged caching is not in use.
    """

    max_length: int | None = None
    page_size: int | None = None


__all__ = "CacheMetadata"
