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

"""Cache/page manager for the eSurge runtime.

Implements the :class:`CacheManagerProtocol` with page-level allocation,
LRU eviction, per-group capacity enforcement, and optional prefix caching.
The manager sits between the scheduler (which decides what to run) and the
page pool (which owns the raw page ids).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

from ..request import EngineRequest
from .interface import CacheAcquireResult, CacheGroupPolicy, CacheGroupsConfig, CacheManagerProtocol
from .page_pool import PagePool


@dataclass(slots=True)
class PageRecord:
    """Internal bookkeeping record for a single cache page.

    Attributes:
        page_id: The integer page identifier.
        cache_group: Name of the cache group this page belongs to.
        ref_count: Number of active requests currently pinning this page.
        owners: Set of request ids that hold a reference to this page.
        last_used_at: Epoch timestamp of the most recent access.
    """

    page_id: int
    cache_group: str
    ref_count: int = 0
    owners: set["str"] = field(default_factory=set)
    last_used_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class EngineManagerState:
    """Runtime state and statistics for cache/page management.

    Attributes:
        max_model_len: Maximum sequence length the engine supports.
        max_num_seqs: Maximum number of concurrent sequences.
        reserve_tokens: Number of tokens reserved for scheduling headroom.
        total_evictions: Cumulative count of page evictions since creation.
        prefix_cache_hits: Cumulative count of successful prefix lookups.
        prefix_cache_misses: Cumulative count of failed prefix lookups.
    """

    max_model_len: int
    max_num_seqs: int
    reserve_tokens: int = 0
    total_evictions: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0


class CacheManager(CacheManagerProtocol):
    """Page-based cache manager with prefix-index and LRU eviction.

    Manages a pool of integer page ids partitioned into named cache groups.
    Supports prefix caching so that requests sharing identical prompt
    prefixes can reuse already-filled KV cache pages.

    Args:
        num_pages: Total number of pages in the underlying pool.
        page_size: Number of tokens per page (must be >= 1).
        cache_groups: Optional configuration for named cache groups with
            per-group capacity limits.
        enable_prefix_caching: If ``True``, maintain a prefix index so
            that requests with matching prompt hashes can share pages.
    """

    def __init__(
        self,
        *,
        num_pages: int,
        page_size: int,
        cache_groups: CacheGroupsConfig | None = None,
        enable_prefix_caching: bool = False,
    ):
        self.page_size = max(int(page_size), 1)
        self.pool = PagePool(capacity=max(int(num_pages), 0))
        self.cache_groups = cache_groups or CacheGroupsConfig()
        self.group_policies = self.cache_groups.by_name()
        self.enable_prefix_caching = bool(enable_prefix_caching)

        self._request_pages: dict[str, list["int"]] = {}
        self._records: dict[int, PageRecord] = {}
        self._evictable = OrderedDict()
        self._prefix_index: dict[str, list["int"]] = {}
        self._page_to_prefix_keys: dict[int, set["str"]] = defaultdict(set)
        self._group_usage: dict[str, int] = defaultdict(int)
        self._total_evictions = 0
        self._prefix_hits = 0
        self._prefix_misses = 0

    @staticmethod
    def prefix_hash_from_tokens(token_ids: list["int"], *, max_tokens: int | None = None) -> str:
        """Compute a SHA-256 hex digest from a list of token ids.

        This is used to build prefix-cache keys. Each token id is encoded
        as an 8-byte little-endian signed integer.

        Args:
            token_ids: Sequence of integer token ids.
            max_tokens: If provided, only the first *max_tokens* ids are
                included in the hash.

        Returns:
            Lowercase hexadecimal SHA-256 digest string.
        """
        if max_tokens is not None and max_tokens >= 0:
            token_ids = token_ids[:max_tokens]
        digest = hashlib.sha256()
        for token in token_ids:
            digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
        return digest.hexdigest()

    def pages_required(self, token_count: int) -> int:
        """Compute the number of pages needed for a given token count.

        Args:
            token_count: Total number of tokens that need cache space.

        Returns:
            The ceiling division of *token_count* by :attr:`page_size`.
        """
        tokens = max(int(token_count), 0)
        if tokens == 0:
            return 0
        return (tokens + self.page_size - 1) // self.page_size

    def _policy_for_group(self, cache_group: str) -> CacheGroupPolicy:
        """Retrieve the policy for a cache group, falling back to defaults.

        Args:
            cache_group: Name of the cache group.

        Returns:
            The :class:`CacheGroupPolicy` for the group.
        """
        return self.group_policies.get(cache_group, CacheGroupPolicy(name=cache_group))

    def _touch_page(self, page_id: int) -> None:
        """Refresh the last-used timestamp for a page and mark it as active.

        Args:
            page_id: The page to touch.
        """
        record = self._records.get(page_id)
        if record is None:
            return
        record.last_used_at = time.time()
        self.pool.mark_used(page_id)
        self._evictable.pop(page_id, None)

    def _mark_evictable(self, page_id: int) -> None:
        """Add a page to the evictable set if its ref_count has reached zero.

        Args:
            page_id: The page that may become eligible for eviction.
        """
        record = self._records.get(page_id)
        if record is None:
            return
        if record.ref_count > 0:
            return
        self._evictable.pop(page_id, None)
        self._evictable[page_id] = None

    def _pin_pages(self, request_id: str, page_ids: list["int"]) -> None:
        """Increment the reference count on pages for a given request.

        Each request may only pin a page once; duplicate pins are ignored.

        Args:
            request_id: The request that is claiming the pages.
            page_ids: Pages to pin.
        """
        for page_id in page_ids:
            record = self._records.get(page_id)
            if record is None:
                continue
            if request_id not in record.owners:
                record.owners.add(request_id)
                record.ref_count += 1
            self._touch_page(page_id)

    def _unpin_pages(self, request_id: str, page_ids: list["int"]) -> None:
        """Decrement the reference count on pages for a given request.

        Pages whose ref_count drops to zero are marked evictable.

        Args:
            request_id: The request releasing its hold.
            page_ids: Pages to unpin.
        """
        for page_id in page_ids:
            record = self._records.get(page_id)
            if record is None:
                continue
            if request_id in record.owners:
                record.owners.remove(request_id)
                record.ref_count = max(record.ref_count - 1, 0)
            if record.ref_count == 0:
                self._mark_evictable(page_id)

    def _detach_prefix_keys_for_page(self, page_id: int) -> None:
        """Remove a page from all prefix-index entries that reference it.

        Args:
            page_id: The page being evicted or otherwise removed.
        """
        keys = self._page_to_prefix_keys.pop(page_id, set())
        for key in keys:
            pages = self._prefix_index.get(key)
            if pages is None:
                continue
            filtered = [pid for pid in pages if pid != page_id]
            if filtered:
                self._prefix_index[key] = filtered
            else:
                self._prefix_index.pop(key, None)

    def _evict_pages(self, wanted: int, *, cache_group: str | None = None) -> list["int"]:
        """Evict up to *wanted* pages using LRU order.

        Args:
            wanted: Number of pages to evict.
            cache_group: If given, only pages belonging to this group are
                candidates for eviction.

        Returns:
            List of page ids that were actually evicted.
        """
        evicted: list["int"] = []
        if wanted <= 0:
            return evicted
        for page_id in list(self._evictable.keys()):
            record = self._records.get(page_id)
            if record is None:
                self._evictable.pop(page_id, None)
                continue
            if cache_group is not None and record.cache_group != cache_group:
                continue
            if record.ref_count > 0:
                self._evictable.pop(page_id, None)
                continue
            self._evictable.pop(page_id, None)
            self._detach_prefix_keys_for_page(page_id)
            self._records.pop(page_id, None)
            self._group_usage[record.cache_group] = max(self._group_usage[record.cache_group] - 1, 0)
            self.pool.release(page_id)
            evicted.append(page_id)
            self._total_evictions += 1
            if len(evicted) >= wanted:
                break
        return evicted

    def _ensure_group_capacity(self, *, cache_group: str, additional_pages: int) -> list["int"]:
        """Evict pages within a group if the allocation would exceed its cap.

        Args:
            cache_group: The cache group being checked.
            additional_pages: How many new pages are about to be allocated.

        Returns:
            List of page ids that were evicted to make room.

        Raises:
            RuntimeError: If not enough evictable pages exist to honour the
                group cap.
        """
        policy = self._policy_for_group(cache_group)
        if policy.max_pages is None:
            return []
        used = self._group_usage.get(cache_group, 0)
        overflow = (used + additional_pages) - int(policy.max_pages)
        if overflow <= 0:
            return []
        evicted = self._evict_pages(overflow, cache_group=cache_group)
        if len(evicted) < overflow:
            raise RuntimeError(
                f"Insufficient evictable pages in cache group {cache_group!r} (need={overflow}, evicted={len(evicted)})"
            )
        return evicted

    def _allocate_pages(self, *, count: int, cache_group: str) -> tuple[list["int"], list["int"]]:
        """Allocate *count* new pages, evicting as necessary.

        Args:
            count: Number of fresh pages needed.
            cache_group: Cache group the new pages will belong to.

        Returns:
            A 2-tuple of ``(new_page_ids, evicted_page_ids)``.
        """
        if count <= 0:
            return [], []
        evicted: list["int"] = []
        evicted.extend(self._ensure_group_capacity(cache_group=cache_group, additional_pages=count))
        needed = max(count - self.pool.free_count(), 0)
        if needed > 0:
            evicted.extend(self._evict_pages(needed))
        page_ids = self.pool.allocate(count)
        for page_id in page_ids:
            self._records[page_id] = PageRecord(page_id=page_id, cache_group=cache_group)
            self._group_usage[cache_group] += 1
        return page_ids, evicted

    def lookup_prefix(self, prefix_hash: str, *, required_pages: int) -> list["int"] | None:
        """Look up a cached prefix by its hash.

        Args:
            prefix_hash: SHA-256 digest of the prefix token ids.
            required_pages: Minimum number of pages needed for the prefix.

        Returns:
            A list of page ids if a valid cached prefix is found, or
            ``None`` if there is no match or prefix caching is disabled.
        """
        if not self.enable_prefix_caching:
            return None
        pages = self._prefix_index.get(prefix_hash)
        if not pages:
            self._prefix_misses += 1
            return None
        valid = [page_id for page_id in pages if page_id in self._records]
        if len(valid) < required_pages:
            self._prefix_misses += 1
            return None
        self._prefix_hits += 1
        return valid[:required_pages]

    def acquire_for_request(
        self,
        request: EngineRequest,
        *,
        required_tokens: int,
        cache_group: str = "default",
        prefix_hash: str | None = None,
    ) -> CacheAcquireResult:
        """Acquire cache pages for a request, allocating or reusing as needed.

        If the request already owns enough pages, no new allocation occurs.
        Otherwise the manager first attempts a prefix-cache lookup, then
        allocates any remaining deficit from the free pool (evicting if
        necessary).

        Args:
            request: The engine request that needs cache pages.
            required_tokens: Total number of tokens requiring cache space.
            cache_group: Cache group from which pages are allocated.
            prefix_hash: Optional prefix hash for prefix-cache lookup.

        Returns:
            A :class:`CacheAcquireResult` describing the allocated pages
            and any evictions that occurred.
        """
        request_id = request.request_id
        target_pages = self.pages_required(required_tokens)
        owned_pages = self._request_pages.get(request_id, [])
        prefix_hit = False
        prefix_pages: list["int"] = []
        evicted: list["int"] = []

        if target_pages == 0:
            self._request_pages[request_id] = []
            request.assign_cache_pages([], cache_group=cache_group)
            return CacheAcquireResult(page_ids=[], cache_group=cache_group)

        if not owned_pages and prefix_hash is not None:
            prompt_pages = self.pages_required(request.total_prompt_tokens)
            if prompt_pages > 0:
                resolved_prefix_pages = self.lookup_prefix(prefix_hash, required_pages=prompt_pages)
                if resolved_prefix_pages is not None:
                    prefix_hit = True
                    prefix_pages = list(resolved_prefix_pages)
                    owned_pages = list(prefix_pages)

        if len(owned_pages) >= target_pages:
            needed_pages = owned_pages[:target_pages]
            self._request_pages[request_id] = needed_pages
            self._pin_pages(request_id, needed_pages)
            request.assign_cache_pages(needed_pages, cache_group=cache_group)
            request.cache_state.prefix_hash = prefix_hash
            request.cache_state.prefix_cache_hit = prefix_hit
            if prefix_hit:
                request.cache_state.prefix_cached_tokens = request.total_prompt_tokens
            return CacheAcquireResult(page_ids=needed_pages, cache_group=cache_group)

        missing = target_pages - len(owned_pages)
        new_pages, evicted = self._allocate_pages(count=missing, cache_group=cache_group)
        full_pages = owned_pages + new_pages
        self._pin_pages(request_id, full_pages)
        self._request_pages[request_id] = full_pages
        request.assign_cache_pages(full_pages, cache_group=cache_group)
        request.cache_state.prefix_hash = prefix_hash
        request.cache_state.prefix_cache_hit = prefix_hit
        request.cache_state.prefix_cached_tokens = request.total_prompt_tokens if prefix_hit else 0
        request.cache_state.evicted_pages += len(evicted)
        return CacheAcquireResult(
            page_ids=full_pages,
            prefix_cache_hit=prefix_hit,
            evicted_page_ids=evicted,
            cache_group=cache_group,
        )

    def cache_prefix(self, prefix_hash: str, request_id: str, *, max_pages: int | None = None) -> None:
        """Register pages of a completed request as a cached prefix.

        Args:
            prefix_hash: The hash identifying this prefix.
            request_id: Request whose pages should back the cache entry.
            max_pages: If given, only the first *max_pages* pages are
                stored in the index.
        """
        if not self.enable_prefix_caching:
            return
        pages = self._request_pages.get(request_id, [])
        if not pages:
            return
        if max_pages is not None and max_pages >= 0:
            pages = pages[:max_pages]
        self._prefix_index[prefix_hash] = list(pages)
        for _page_id in pages:
            self._page_to_prefix_keys[_page_id].add(prefix_hash)

    def release_request(self, request_id: str) -> list["int"]:
        """Release all pages held by a request.

        Args:
            request_id: The unique identifier of the request.

        Returns:
            List of page ids that were released.
        """
        pages = self._request_pages.pop(request_id, [])
        if not pages:
            return []
        self._unpin_pages(request_id, pages)
        return list(pages)

    def get_request_pages(self, request_id: str) -> list["int"]:
        """Retrieve the page ids currently assigned to a request.

        Args:
            request_id: Unique identifier of the request.

        Returns:
            A copy of the list of page ids owned by the request.
        """
        return list(self._request_pages.get(request_id, []))

    def touch_request(self, request_id: str) -> None:
        """Refresh the last-used timestamp on every page owned by a request.

        Args:
            request_id: Unique identifier of the request.
        """
        for page_id in self._request_pages.get(request_id, []):
            self._touch_page(page_id)

    def reset(self) -> None:
        """Reset all internal state and recreate the page pool.

        After calling this method the manager behaves as if it were newly
        constructed. All request associations, prefix entries, and
        statistics are cleared.
        """
        self._request_pages.clear()
        self._records.clear()
        self._evictable.clear()
        self._prefix_index.clear()
        self._page_to_prefix_keys.clear()
        self._group_usage.clear()
        self._total_evictions = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self.pool = PagePool(capacity=self.pool.capacity)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of cache usage statistics.

        Returns:
            A dictionary with keys: ``"free_pages"``, ``"in_use_pages"``,
            ``"tracked_pages"``, ``"active_requests"``,
            ``"prefix_entries"``, ``"prefix_hits"``, ``"prefix_misses"``,
            ``"evictions"``.
        """
        return {
            "free_pages": self.pool.free_count(),
            "in_use_pages": self.pool.in_use_count(),
            "tracked_pages": len(self._records),
            "active_requests": len(self._request_pages),
            "prefix_entries": len(self._prefix_index),
            "prefix_hits": self._prefix_hits,
            "prefix_misses": self._prefix_misses,
            "evictions": self._total_evictions,
        }
