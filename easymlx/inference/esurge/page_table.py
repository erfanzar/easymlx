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

"""Minimal page-table helpers for the easymlx paged eSurge runtime.

Provides a lightweight, thread-safe in-memory page table that maps
request IDs to cache slot and page ownership metadata.  This abstraction
is used by the engine to track which KV-cache pages are assigned to
each active request.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(slots=True)
class PageTableEntry:
    """Mapping between a request and the slot/page ids it owns.

    Attributes:
        request_id: Unique identifier of the owning request.
        slot_id: Cache slot index assigned to the request.
        page_ids: Tuple of page indices allocated to the request.
    """

    request_id: str
    slot_id: int
    page_ids: tuple[int, ...] = ()


class PageTable:
    """Small in-memory page-table abstraction for slot/page bookkeeping.

    All public methods are guarded by a reentrant lock to allow safe
    concurrent access from scheduler and runner threads.
    """

    def __init__(self):
        """Initialize an empty page table."""
        self._lock = RLock()
        self._entries: dict[str, PageTableEntry] = {}

    def add(self, entry: PageTableEntry) -> None:
        """Insert or replace a page-table entry.

        Args:
            entry: The :class:`PageTableEntry` to store.  If an entry
                with the same ``request_id`` already exists it is
                overwritten.
        """
        with self._lock:
            self._entries[entry.request_id] = entry

    def get(self, request_id: str) -> PageTableEntry | None:
        """Retrieve an entry by request ID.

        Args:
            request_id: The request whose entry to look up.

        Returns:
            The corresponding :class:`PageTableEntry`, or ``None`` if
            not found.
        """
        with self._lock:
            return self._entries.get(request_id)

    def remove(self, request_id: str) -> PageTableEntry | None:
        """Remove and return an entry by request ID.

        Args:
            request_id: The request whose entry to remove.

        Returns:
            The removed :class:`PageTableEntry`, or ``None`` if the
            request was not tracked.
        """
        with self._lock:
            return self._entries.pop(request_id, None)

    def snapshot(self) -> dict[str, PageTableEntry]:
        """Return a shallow copy of all current entries.

        Returns:
            A new dictionary mapping request IDs to
            :class:`PageTableEntry` objects.
        """
        with self._lock:
            return dict(self._entries)


__all__ = ("PageTable", "PageTableEntry")
