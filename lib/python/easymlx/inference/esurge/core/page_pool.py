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

"""Page pool with deterministic allocation and LRU-ready bookkeeping.

This module provides a bounded pool of integer page identifiers that can be
allocated and released. Each page tracks an ``in_use`` flag and a
``last_used_at`` timestamp suitable for LRU eviction decisions in the
higher-level :class:`~easymlx.inference.esurge.core.manager.CacheManager`.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class PageState:
    """Tracks the runtime state of a single page in the pool.

    Attributes:
        page_id: Integer identifier for this page (0-indexed).
        in_use: Whether the page is currently allocated to a request.
        last_used_at: Epoch timestamp of the last allocation or touch event.
    """

    page_id: int
    in_use: bool = False
    last_used_at: float = 0.0


class PagePool:
    """Allocate and release integer page ids from a bounded pool.

    Pages are allocated in FIFO order from a free-list and returned to the
    back of the free-list on release. This guarantees deterministic
    allocation ordering and low overhead.

    Args:
        capacity: Maximum number of pages in the pool. Must be non-negative.

    Raises:
        RuntimeError: If an :meth:`allocate` call requests more pages than
            are currently free.

    Example:
        >>> pool = PagePool(capacity=8)
        >>> pages = pool.allocate(3)
        >>> len(pages)
        3
        >>> pool.free_count()
        5
    """

    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 0)
        self._states: list[PageState] = [PageState(page_id=idx) for idx in range(self.capacity)]
        self._free = deque(range(self.capacity))
        self._in_use: set["int"] = set()

    def allocate(self, count: int = 1) -> list["int"]:
        """Allocate one or more pages from the free pool.

        Args:
            count: Number of pages to allocate. Defaults to 1.

        Returns:
            A list of allocated page ids.

        Raises:
            RuntimeError: If *count* exceeds the number of free pages.
        """
        wanted = max(int(count), 0)
        if wanted == 0:
            return []
        if len(self._free) < wanted:
            raise RuntimeError(f"No free pages available (requested={wanted}, free={len(self._free)})")
        now = time.time()
        page_ids: list["int"] = []
        for _ in range(wanted):
            page_id = self._free.popleft()
            state = self._states[page_id]
            state.in_use = True
            state.last_used_at = now
            self._in_use.add(page_id)
            page_ids.append(page_id)
        return page_ids

    def release(self, page_ids: int | list["int"]) -> list["int"]:
        """Return pages to the free pool.

        Args:
            page_ids: A single page id or a list of page ids to release.

        Returns:
            A list of page ids that were actually released (skipping ids
            that were already free or out of range).
        """
        ids = [int(page_ids)] if isinstance(page_ids, int) else [int(page_id) for page_id in page_ids]
        released: list["int"] = []
        now = time.time()
        for page_id in ids:
            if page_id < 0 or page_id >= self.capacity:
                continue
            state = self._states[page_id]
            if not state.in_use:
                continue
            state.in_use = False
            state.last_used_at = now
            self._in_use.discard(page_id)
            self._free.append(page_id)
            released.append(page_id)
        return released

    def mark_used(self, page_id: int) -> None:
        """Update the ``last_used_at`` timestamp for a page without changing ownership.

        Args:
            page_id: The page whose timestamp should be refreshed.
        """
        if page_id < 0 or page_id >= self.capacity:
            return
        state = self._states[page_id]
        state.last_used_at = time.time()

    def is_free(self, page_id: int) -> bool:
        """Check whether a page is currently free (not in use).

        Args:
            page_id: The page id to check.

        Returns:
            ``True`` if the page is not currently allocated.
        """
        return int(page_id) not in self._in_use

    def free_count(self) -> int:
        """Return the number of pages currently available for allocation.

        Returns:
            Count of free pages.
        """
        return len(self._free)

    def in_use_count(self) -> int:
        """Return the number of pages currently allocated.

        Returns:
            Count of in-use pages.
        """
        return len(self._in_use)

    def snapshot(self) -> list[PageState]:
        """Create a deep copy of the current page states.

        Returns:
            A list of :class:`PageState` copies, one per page in the pool.
        """
        return [PageState(page_id=s.page_id, in_use=s.in_use, last_used_at=s.last_used_at) for s in self._states]
