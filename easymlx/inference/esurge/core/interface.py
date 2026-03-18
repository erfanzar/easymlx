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

"""Core runtime contracts for eSurge components.

This module defines the protocol interfaces and configuration data classes that
form the contract between the cache management layer, inference engine, and
scheduler subsystems. All cache managers must conform to
:class:`CacheManagerProtocol`, and local inference engines must conform to
:class:`InferenceEngineProtocol`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..outputs import RequestOutput
from ..request import EngineRequest
from ..sampling_params import SamplingParams


@dataclass(slots=True)
class CacheGroupPolicy:
    """Policy descriptor for a single cache group.

    Cache groups allow partitioning the KV page pool so that different
    workloads (e.g. long-context vs. short-context) can be managed with
    independent capacity limits.

    Attributes:
        name: Human-readable identifier for this cache group.
        max_pages: Optional upper bound on the number of pages this group
            may consume. ``None`` means unlimited (bounded only by pool size).
        attention_type: The attention algorithm used by sequences in this
            group (e.g. ``"paged"``).
    """

    name: str = "default"
    max_pages: int | None = None
    attention_type: str = "paged"


@dataclass(slots=True)
class CacheGroupsConfig:
    """Collection of cache group policies.

    Attributes:
        groups: Tuple of :class:`CacheGroupPolicy` instances describing
            every configured cache group.
    """

    groups: tuple[CacheGroupPolicy, ...] = (CacheGroupPolicy(),)

    def by_name(self) -> dict[str, CacheGroupPolicy]:
        """Return a mapping from group name to its policy.

        Returns:
            A dictionary keyed by :pyattr:`CacheGroupPolicy.name`.
        """
        return {group.name: group for group in self.groups}


@dataclass(slots=True)
class CacheAcquireResult:
    """Result returned by cache manager page acquisition.

    Attributes:
        page_ids: List of integer page identifiers that were assigned to
            the request.
        prefix_cache_hit: Whether the acquisition was satisfied (fully or
            partially) from a cached prefix.
        evicted_page_ids: Page identifiers that had to be evicted to make
            room for this allocation.
        cache_group: Name of the cache group the pages belong to.
    """

    page_ids: list["int"]
    prefix_cache_hit: bool = False
    evicted_page_ids: list["int"] = field(default_factory=list)
    cache_group: str = "default"


class InferenceEngineProtocol(Protocol):
    """Protocol for local inference engines that look like eSurge.

    Any object exposing a :meth:`generate` method with the signature below
    can be used interchangeably with the concrete eSurge engine.
    """

    def generate(
        self,
        prompts: str | Iterable["str"],
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> list[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string or an iterable of prompt strings.
            sampling_params: Optional sampling parameters controlling
                temperature, top-k/top-p, stop sequences, etc.
            **generate_kwargs: Additional keyword arguments forwarded to the
                underlying model runner.

        Returns:
            A list of :class:`RequestOutput` objects, one per input prompt.
        """
        ...


class CacheManagerProtocol(Protocol):
    """Protocol for page-based KV cache managers.

    Implementations must support page allocation, release, prefix caching,
    and usage statistics. The :attr:`page_size` attribute determines how
    many tokens fit in a single cache page.

    Attributes:
        page_size: Number of tokens stored per cache page.
    """

    page_size: int

    def pages_required(self, token_count: int) -> int:
        """Compute the number of pages needed for a given token count.

        Args:
            token_count: Total number of tokens to accommodate.

        Returns:
            The minimum number of pages required.
        """
        ...

    def acquire_for_request(
        self,
        request: EngineRequest,
        *,
        required_tokens: int,
        cache_group: str = "default",
        prefix_hash: str | None = None,
    ) -> CacheAcquireResult:
        """Acquire cache pages for a request.

        Args:
            request: The engine request that needs cache pages.
            required_tokens: Number of tokens requiring cache space.
            cache_group: Cache group from which pages should be allocated.
            prefix_hash: Optional SHA-256 hash of the prompt prefix for
                prefix-cache lookup.

        Returns:
            A :class:`CacheAcquireResult` with allocated page ids and
            eviction metadata.
        """
        ...

    def release_request(self, request_id: str) -> list["int"]:
        """Release all pages held by a request.

        Args:
            request_id: Unique identifier of the request whose pages
                should be freed.

        Returns:
            List of page ids that were released.
        """
        ...

    def get_request_pages(self, request_id: str) -> list["int"]:
        """Retrieve the page ids currently assigned to a request.

        Args:
            request_id: Unique identifier of the request.

        Returns:
            List of page ids owned by the request.
        """
        ...

    def touch_request(self, request_id: str) -> None:
        """Update the last-used timestamp of all pages for a request.

        Args:
            request_id: Unique identifier of the request.
        """
        ...

    def cache_prefix(self, prefix_hash: str, request_id: str, *, max_pages: int | None = None) -> None:
        """Register pages of a request as a cached prefix.

        Args:
            prefix_hash: SHA-256 hash uniquely identifying the prefix.
            request_id: Request whose pages should back this prefix entry.
            max_pages: If given, only the first *max_pages* pages are cached.
        """
        ...

    def lookup_prefix(self, prefix_hash: str, *, required_pages: int) -> list["int"] | None:
        """Look up a cached prefix by hash.

        Args:
            prefix_hash: SHA-256 hash identifying the prefix.
            required_pages: Number of pages needed for the prefix.

        Returns:
            A list of page ids if the prefix is cached and has enough pages,
            or ``None`` if there is no match.
        """
        ...

    def stats(self) -> dict[str, int]:
        """Return a dictionary of cache usage statistics.

        Returns:
            A dictionary containing keys such as ``"free_pages"``,
            ``"in_use_pages"``, ``"tracked_pages"``, etc.
        """
        ...
