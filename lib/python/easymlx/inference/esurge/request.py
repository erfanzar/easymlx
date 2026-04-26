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

"""Request types for the eSurge runtime.

Provides the core request lifecycle types used by schedulers, runners,
and the engine loop:

* :class:`EngineRequestStatus` -- finite-state enum for request status.
* :class:`RequestCacheState` -- cache page ownership metadata.
* :class:`EngineRequest` -- authoritative mutable request state tracked
  throughout its lifetime from admission to completion.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

from .sampling_params import SamplingParams


class EngineRequestStatus(enum.StrEnum):
    """Finite-state lifecycle status of an engine request.

    States:
        WAITING: Request is queued and awaiting scheduling.
        RUNNING: Request is actively being processed.
        PAUSED: Request has been temporarily paused.
        PREEMPTED: Request was preempted to free resources.
        FINISHED: Request completed successfully.
        FAILED: Request encountered an error.
        CANCELED: Request was canceled by the client.
    """

    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(slots=True)
class RequestCacheState:
    """Cache ownership metadata for one request.

    Tracks the KV-cache pages assigned to a request, whether a prefix
    cache hit was obtained, and eviction statistics.

    Attributes:
        page_ids: List of page indices currently owned.
        cache_group: Name of the cache group this request belongs to.
        prefix_hash: Hash of the prompt token sequence for prefix
            caching lookups.
        prefix_cache_hit: Whether a prefix cache hit was found.
        prefix_cached_tokens: Number of tokens served from the prefix
            cache.
        evicted_pages: Cumulative count of pages evicted due to
            preemption.
    """

    page_ids: list["int"] = field(default_factory=list)
    cache_group: str = "default"
    prefix_hash: str | None = None
    prefix_cache_hit: bool = False
    prefix_cached_tokens: int = 0
    evicted_pages: int = 0


@dataclass(slots=True)
class EngineRequest:
    """Authoritative request state tracked by schedulers/runners.

    This mutable dataclass is the single source of truth for a request
    from the moment it enters the engine until it finishes (or fails /
    is canceled).  Schedulers, runners, and the output builder all
    mutate fields on this object.

    Attributes:
        request_id: Globally unique request identifier.
        prompt: The original prompt text.
        prompt_token_ids: Token IDs of the prompt.
        sampling_params: Sampling configuration for generation.
        eos_token_id: End-of-sequence token ID override.
        client_index: Index for client-side correlation.
        arrival_time: Wall-clock arrival timestamp (seconds since epoch).
        priority: Scheduling priority (higher = more important).
        status: Current lifecycle status.
        num_computed_tokens: Tokens whose KV entries have been computed.
        num_cached_tokens: High-water mark of computed tokens (survives
            preemption tracking).
        num_output_placeholders: Async decode placeholder count.
        generated_token_ids: Tokens generated so far.
        spec_token_ids: Speculative decoding candidate tokens.
        finished_reason: Reason the request finished, if applicable.
        failure_reason: Error description when the request fails.
        finished_at: Timestamp when the request finished.
        preemptions: Number of times the request was preempted.
        canceled: Whether the request was canceled.
        metadata: Arbitrary metadata dictionary.
        cache_state: Cache page ownership metadata.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list["int"]
    sampling_params: SamplingParams | None = None
    eos_token_id: int | None = None
    client_index: int = 0
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0
    status: EngineRequestStatus = EngineRequestStatus.WAITING
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0
    num_output_placeholders: int = 0
    generated_token_ids: list["int"] = field(default_factory=list)
    spec_token_ids: list["int"] = field(default_factory=list)
    finished_reason: str | None = None
    failure_reason: str | None = None
    finished_at: float | None = None
    preemptions: int = 0
    canceled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_state: RequestCacheState = field(default_factory=RequestCacheState)

    @property
    def max_new_tokens(self) -> int:
        """Return the maximum number of new tokens to generate.

        Returns:
            ``sampling_params.max_tokens`` clamped to >= 0, or ``0`` if
            sampling params are not set.
        """
        if self.sampling_params is None:
            return 0
        return max(int(self.sampling_params.max_tokens), 0)

    @property
    def num_generated_tokens(self) -> int:
        """Return the count of tokens generated so far.

        Returns:
            Length of ``generated_token_ids``.
        """
        return len(self.generated_token_ids)

    @property
    def all_token_ids(self) -> list["int"]:
        """Return the full token sequence (prompt + generated).

        Returns:
            Concatenation of ``prompt_token_ids`` and
            ``generated_token_ids``.
        """
        return list(self.prompt_token_ids) + list(self.generated_token_ids)

    @property
    def total_prompt_tokens(self) -> int:
        """Return the number of prompt tokens.

        Returns:
            Length of ``prompt_token_ids``.
        """
        return len(self.prompt_token_ids)

    @property
    def total_tokens(self) -> int:
        """Return the total token count (prompt + generated).

        Returns:
            Length of :attr:`all_token_ids`.
        """
        return len(self.all_token_ids)

    @property
    def remaining_prefill_tokens(self) -> int:
        """Return how many tokens still need prefill computation.

        Returns:
            Non-negative count of tokens not yet computed.
        """
        return max(self.total_tokens - self.num_computed_tokens, 0)

    @property
    def remaining_generation_budget(self) -> int:
        """Return how many more tokens may be generated.

        Returns:
            ``max_new_tokens - num_generated_tokens``, clamped to >= 0.
        """
        return max(self.max_new_tokens - self.num_generated_tokens, 0)

    @property
    def is_finished(self) -> bool:
        """Return whether the request has reached a terminal state.

        Returns:
            ``True`` if status is ``FINISHED``, ``FAILED``, or
            ``CANCELED``.
        """
        return self.status in {
            EngineRequestStatus.FINISHED,
            EngineRequestStatus.FAILED,
            EngineRequestStatus.CANCELED,
        }

    @property
    def should_decode(self) -> bool:
        """Return whether the request is ready for decode-phase generation.

        Returns:
            ``True`` when the request is running, prefill is complete,
            generation budget remains, and the request is not finished.
        """
        return (
            self.status == EngineRequestStatus.RUNNING
            and self.remaining_prefill_tokens == 0
            and self.remaining_generation_budget > 0
            and not self.is_finished
        )

    @property
    def use_structured_output(self) -> bool:
        """Return whether this request uses structured output (e.g. JSON schema).

        Returns:
            ``True`` if ``sampling_params.response_format`` is set.
        """
        if self.sampling_params is None:
            return False
        return self.sampling_params.response_format is not None

    def mark_waiting(self) -> None:
        """Transition the request to ``WAITING`` status."""
        self.status = EngineRequestStatus.WAITING

    def mark_running(self) -> None:
        """Transition the request to ``RUNNING`` status."""
        self.status = EngineRequestStatus.RUNNING

    def mark_paused(self) -> None:
        """Transition the request to ``PAUSED`` status."""
        self.status = EngineRequestStatus.PAUSED

    def mark_preempted(self) -> None:
        """Transition the request to ``PREEMPTED`` status and increment the preemption counter."""
        self.status = EngineRequestStatus.PREEMPTED
        self.preemptions += 1

    def mark_finished(self, reason: str | None = None) -> None:
        """Transition the request to ``FINISHED`` status.

        Args:
            reason: Human-readable finish reason (e.g. ``"eos"``,
                ``"length"``).
        """
        self.status = EngineRequestStatus.FINISHED
        self.finished_reason = reason
        self.finished_at = time.time()

    def mark_failed(self, reason: str) -> None:
        """Transition the request to ``FAILED`` status.

        Args:
            reason: Description of the failure.
        """
        self.status = EngineRequestStatus.FAILED
        self.failure_reason = reason
        self.finished_reason = "error"
        self.finished_at = time.time()

    def mark_canceled(self) -> None:
        """Transition the request to ``CANCELED`` status."""
        self.status = EngineRequestStatus.CANCELED
        self.finished_reason = "canceled"
        self.canceled = True
        self.finished_at = time.time()

    def append_generated_token(self, token_id: int) -> None:
        """Append a newly generated token to the output sequence.

        Args:
            token_id: The token ID to append.
        """
        self.generated_token_ids.append(int(token_id))

    def consume_computed_tokens(self, token_count: int) -> None:
        """Record that additional tokens have been computed by the runner.

        Updates both ``num_computed_tokens`` and the high-water mark
        ``num_cached_tokens``.

        Args:
            token_count: Number of newly computed tokens.
        """
        consumed = max(int(token_count), 0)
        self.num_computed_tokens += consumed
        self.num_cached_tokens = max(self.num_cached_tokens, self.num_computed_tokens)

    def mark_prefix_cache_hit(self, prefix_hash: str, page_ids: list["int"]) -> None:
        """Record a successful prefix cache lookup.

        Args:
            prefix_hash: Hash that matched in the prefix cache.
            page_ids: Page IDs restored from the cache.
        """
        self.cache_state.prefix_hash = prefix_hash
        self.cache_state.prefix_cache_hit = True
        self.cache_state.page_ids = [int(page_id) for page_id in page_ids]
        self.cache_state.prefix_cached_tokens = self.total_prompt_tokens

    def assign_cache_pages(self, page_ids: list["int"], *, cache_group: str = "default") -> None:
        """Assign KV-cache pages to this request.

        Args:
            page_ids: List of page indices to assign.
            cache_group: Name of the cache group.
        """
        self.cache_state.page_ids = [int(page_id) for page_id in page_ids]
        self.cache_state.cache_group = str(cache_group)

    def reset_cached_progress(self, *, cached_tokens: int = 0) -> None:
        """Reset computed/cached token counters, typically after preemption.

        Args:
            cached_tokens: Number of tokens to retain as cached.
                Defaults to ``0`` (full reset).
        """
        cached = max(int(cached_tokens), 0)
        self.num_computed_tokens = cached
        self.num_cached_tokens = cached
        self.num_output_placeholders = 0
