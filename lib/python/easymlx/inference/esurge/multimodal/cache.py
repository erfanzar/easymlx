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

"""Vision encoder output cache for multimodal preprocessing.

Provides a thread-safe, memory-bounded LRU cache that stores vision
encoder embeddings so that identical images do not need to be re-encoded
across requests. Entries are keyed by an MD5 hash of pixel data and are
evicted in least-recently-used order when the memory budget is exceeded.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np


class VisionEncoderCache:
    """Thread-safe memory-bounded LRU cache for vision encoder embeddings.

    The cache stores ``(embeddings, size_bytes)`` tuples keyed by an MD5
    hash of the input pixel values.  Concurrent access is guarded by a
    reentrant lock, and evictions follow LRU order.

    Attributes:
        capacity_bytes: Maximum cache capacity in bytes.
        current_size: Current total size of cached entries in bytes.
        hits: Running count of cache hits.
        misses: Running count of cache misses.
    """

    def __init__(self, capacity_mb: int = 1024):
        """Initialize the vision encoder cache.

        Args:
            capacity_mb: Maximum cache capacity in megabytes.  A value of
                ``0`` effectively disables caching because no entry can
                fit.
        """
        self.capacity_bytes = max(int(capacity_mb), 0) * 1024 * 1024
        self.current_size = 0
        self._cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_hash(pixel_values: Any) -> str:
        """Compute an MD5 hash for a pixel-value array.

        For large arrays (> 1 million elements) a strided subsample is
        hashed instead of the full buffer to keep latency low.

        Args:
            pixel_values: Pixel data to hash.

        Returns:
            Hex-encoded MD5 digest string.

        Example:
            >>> import numpy as np
            >>> h = VisionEncoderCache.compute_hash(np.zeros((3, 224, 224), dtype=np.float32))
            >>> isinstance(h, str) and len(h) == 32
            True
        """
        array = np.asarray(pixel_values)
        shape_bytes = np.asarray(array.shape, dtype=np.int32).tobytes()
        if array.size > 1_000_000:
            sampled = array.flat[::100]
            content_bytes = sampled.tobytes()
        else:
            content_bytes = array.tobytes()
        return hashlib.md5(shape_bytes + content_bytes).hexdigest()

    def get(self, hash_key: str) -> Any | None:
        """Retrieve cached embeddings and promote the entry to most-recent.

        Args:
            hash_key: MD5 hash key previously returned by
                :meth:`compute_hash`.

        Returns:
            The cached embeddings object, or ``None`` on a cache miss.
        """
        with self._lock:
            if hash_key not in self._cache:
                self.misses += 1
                return None
            self._cache.move_to_end(hash_key)
            self.hits += 1
            return self._cache[hash_key][0]

    def put(self, hash_key: str, embeddings: Any) -> None:
        """Insert embeddings into the cache, evicting LRU entries as needed.

        If the entry already exists it is promoted without re-insertion.
        Entries whose byte size exceeds the total cache capacity are
        silently dropped.

        Args:
            hash_key: MD5 hash key for the pixel values.
            embeddings: The embeddings object to cache.  Must expose an
                ``nbytes`` attribute for size accounting.
        """
        size_bytes = int(getattr(embeddings, "nbytes", 0))
        if size_bytes > self.capacity_bytes:
            return
        with self._lock:
            if hash_key in self._cache:
                self._cache.move_to_end(hash_key)
                return
            while self.current_size + size_bytes > self.capacity_bytes and self._cache:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self.current_size -= evicted_size
            self._cache[hash_key] = (embeddings, size_bytes)
            self.current_size += size_bytes

    def contains(self, hash_key: str) -> bool:
        """Check whether a hash key is present in the cache.

        Args:
            hash_key: MD5 hash key to look up.

        Returns:
            ``True`` if the key exists, ``False`` otherwise.
        """
        with self._lock:
            return hash_key in self._cache

    def clear(self) -> None:
        """Remove all entries and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        """Return the number of entries currently in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def size_mb(self) -> float:
        """Return the current cache size in megabytes."""
        return self.current_size / (1024 * 1024)

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float in ``[0.0, 1.0]``."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Return a dictionary of cache statistics.

        Returns:
            A dict with keys ``hits``, ``misses``, ``hit_rate``,
            ``size_mb``, ``num_entries``, and ``capacity_mb``.
        """
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hit_rate,
                "size_mb": self.size_mb,
                "num_entries": len(self._cache),
                "capacity_mb": self.capacity_bytes / (1024 * 1024),
            }


class MultimodalCache(VisionEncoderCache):
    """Backward-compatible alias for multimodal cache users.

    Inherits all behaviour from :class:`VisionEncoderCache` without
    modification. Existing code that imports ``MultimodalCache`` will
    continue to work as expected.
    """


__all__ = ("MultimodalCache", "VisionEncoderCache")
