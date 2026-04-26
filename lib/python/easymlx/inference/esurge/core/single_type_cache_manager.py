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

"""Single-type LRU cache utility used by runtime managers.

Provides a generic bounded LRU dictionary with optional time-to-live (TTL)
expiration. Expired entries are lazily evicted on access, and the cache is
pruned to :attr:`max_entries` on every :meth:`set`.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class CacheItem[T]:
    """Internal wrapper that pairs a cached value with timestamps.

    Attributes:
        value: The stored value.
        created_at: Epoch timestamp when the entry was first inserted.
        touched_at: Epoch timestamp of the most recent access or update.
    """

    value: T
    created_at: float
    touched_at: float


class SingleTypeCacheManager[K, V]:
    """Simple bounded LRU cache with optional TTL enforcement.

    Entries are stored in an :class:`~collections.OrderedDict` and promoted
    to the tail on every access, giving O(1) amortized eviction of the
    least-recently-used item.

    Args:
        max_entries: Maximum number of entries the cache may hold. When
            exceeded, the oldest entries are evicted on the next
            :meth:`set` call.
        ttl_seconds: Optional time-to-live in seconds. Entries not touched
            within this window are considered expired and transparently
            removed on access. ``None`` disables TTL expiration.

    Example:
        >>> cache: SingleTypeCacheManager[str, int] = SingleTypeCacheManager(max_entries=2)
        >>> cache.set("a", 1)
        >>> cache.set("b", 2)
        >>> cache.get("a")
        1
        >>> cache.set("c", 3)  # evicts "b" (LRU)
        >>> cache.get("b") is None
        True
    """

    def __init__(self, *, max_entries: int = 4096, ttl_seconds: float | None = None):
        self.max_entries = max(int(max_entries), 0)
        self.ttl_seconds = None if ttl_seconds is None else max(float(ttl_seconds), 0.0)
        self._cache: OrderedDict[K, CacheItem[V]] = OrderedDict()

    def _is_expired(self, item: CacheItem[V], now: float) -> bool:
        """Check whether an item has exceeded its TTL.

        Args:
            item: The cache item to check.
            now: Current epoch timestamp.

        Returns:
            ``True`` if the item is expired.
        """
        if self.ttl_seconds is None:
            return False
        return (now - item.touched_at) > self.ttl_seconds

    def _prune(self) -> None:
        """Remove oldest entries until the cache size is within bounds."""
        if self.max_entries <= 0:
            self._cache.clear()
            return
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get(self, key: K, default: V | None = None) -> V | None:
        """Retrieve a value by key, refreshing its LRU position.

        Expired entries are silently removed and treated as absent.

        Args:
            key: The cache key to look up.
            default: Value to return if *key* is not found or expired.

        Returns:
            The cached value, or *default*.
        """
        item = self._cache.get(key)
        if item is None:
            return default
        now = time.time()
        if self._is_expired(item, now):
            self._cache.pop(key, None)
            return default
        item.touched_at = now
        self._cache.move_to_end(key)
        return item.value

    def set(self, key: K, value: V) -> None:
        """Insert or update a cache entry.

        If the key already exists its ``created_at`` timestamp is
        preserved. After insertion the cache is pruned to
        :attr:`max_entries`.

        Args:
            key: The cache key.
            value: The value to store.
        """
        now = time.time()
        current = self._cache.get(key)
        created_at = current.created_at if current is not None else now
        self._cache[key] = CacheItem(value=value, created_at=created_at, touched_at=now)
        self._cache.move_to_end(key)
        self._prune()

    def pop(self, key: K, default: V | None = None) -> V | None:
        """Remove and return a value by key.

        Args:
            key: The cache key to remove.
            default: Value to return if *key* is not found.

        Returns:
            The removed value, or *default*.
        """
        item = self._cache.pop(key, None)
        if item is None:
            return default
        return item.value

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    def __contains__(self, key: object) -> bool:
        """Check membership, respecting TTL expiration.

        Args:
            key: The key to test.

        Returns:
            ``True`` if the key is present and not expired.
        """
        if key not in self._cache:
            return False
        value = self.get(key)
        return value is not None

    def __len__(self) -> int:
        """Return the number of live (non-expired) entries.

        Returns:
            Count of entries currently in the cache after GC.
        """
        self._gc_expired()
        return len(self._cache)

    def _gc_expired(self) -> None:
        """Garbage-collect all expired entries in a single pass."""
        if self.ttl_seconds is None or not self._cache:
            return
        now = time.time()
        expired_keys = [key for key, item in self._cache.items() if self._is_expired(item, now)]
        for _key in expired_keys:
            self._cache.pop(_key, None)

    def keys(self) -> list[K]:
        """Return a list of all live (non-expired) keys.

        Returns:
            List of keys currently in the cache.
        """
        self._gc_expired()
        return list(self._cache.keys())
