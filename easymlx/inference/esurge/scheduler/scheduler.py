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

"""Main scheduler implementation for the eSurge runtime.

Provides :class:`Scheduler`, a continuous-batching scheduler with
page-aware request admission, prefix caching, chunked prefill, and
preemption support.  :class:`LocalScheduler` is a backward-compatible
alias.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque

from easymlx.workers.loggers import get_logger

from ..config import CacheConfig, Config, SchedulerConfig
from ..core.interface import CacheGroupsConfig
from ..core.manager import CacheManager
from ..core.utils import prompt_token_hash
from ..request import EngineRequest, EngineRequestStatus
from .output import ScheduledRequest, SchedulerStepOutput
from .request_queue import SchedulingPolicy, create_request_queue
from .token_budget import TokenBudgetManager

logger = get_logger("eSurgeScheduler")


class Scheduler:
    """Continuous-batching scheduler with page-aware request admission.

    Manages a waiting queue, a running set, per-request row allocation,
    and a :class:`CacheManager` to acquire / release KV-cache pages.
    Each call to :meth:`schedule` produces a :class:`SchedulerStepOutput`
    describing the work for one engine iteration.

    Attributes:
        scheduler_config: Scheduler-specific configuration.
        max_num_running_reqs: Maximum concurrent running requests.
        max_num_scheduled_tokens: Maximum tokens per scheduling step.
        max_model_len: Maximum supported context length.
        page_size: KV-cache page size in tokens.
        cache_manager: The :class:`CacheManager` instance.
        policy: Active scheduling policy.
        waiting: Queue of requests awaiting admission.
        requests: All known requests by ID.
        running: Currently running requests (insertion-ordered).
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig | Config,
        *,
        cache_config: CacheConfig | None = None,
        cache_groups: CacheGroupsConfig | None = None,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize the scheduler.

        Args:
            scheduler_config: Scheduler configuration, or a top-level
                :class:`Config` from which scheduler and cache configs
                are extracted.
            cache_config: Cache configuration.  Ignored when
                *scheduler_config* is a :class:`Config`.
            cache_groups: Optional cache group configuration.
            cache_manager: Pre-existing :class:`CacheManager`.  One is
                created automatically when ``None``.
        """
        if isinstance(scheduler_config, Config):
            self.scheduler_config = scheduler_config.scheduler_config
            effective_cache_config = scheduler_config.cache_config
        else:
            self.scheduler_config = scheduler_config
            effective_cache_config = cache_config or CacheConfig()

        self.max_num_running_reqs = int(self.scheduler_config.max_num_seqs)
        self.max_num_scheduled_tokens = int(
            self.scheduler_config.max_num_batched_tokens or self.scheduler_config.max_model_len
        )
        self.max_model_len = int(self.scheduler_config.max_model_len)
        self.page_size = max(int(effective_cache_config.page_size), 1)

        num_pages = int(
            effective_cache_config.num_pages or (self.max_num_running_reqs * self.max_model_len // self.page_size)
        )
        self.cache_manager = cache_manager or CacheManager(
            num_pages=max(num_pages, self.max_num_running_reqs),
            page_size=self.page_size,
            cache_groups=cache_groups,
            enable_prefix_caching=bool(effective_cache_config.enable_prefix_caching),
        )
        self._prefix_caching_enabled = bool(effective_cache_config.enable_prefix_caching)
        self._token_budget_manager = TokenBudgetManager(
            max_batch_tokens=self.max_num_scheduled_tokens,
            safety_margin_tokens=max(int(self.scheduler_config.token_safety_margin or 0), 0),
        )

        self.policy = SchedulingPolicy(self.scheduler_config.policy)
        self.waiting = create_request_queue(self.policy)
        self.requests: dict[str, EngineRequest] = {}
        self.running: OrderedDict[str, EngineRequest] = OrderedDict()
        self._row_for_request: dict[str, int] = {}
        self._free_rows: deque["int"] = deque(range(self.max_num_running_reqs))

        logger.info(
            "Scheduler initialized: max_running=%d, max_tokens=%d, max_model_len=%d, "
            "page_size=%d, num_pages=%d, prefix_caching=%s, policy=%s",
            self.max_num_running_reqs,
            self.max_num_scheduled_tokens,
            self.max_model_len,
            self.page_size,
            (getattr(self.cache_manager, "pool", None) and getattr(self.cache_manager.pool, "capacity", "?"))
            or num_pages,
            self._prefix_caching_enabled,
            self.policy.value,
        )

    def add_request(self, request: EngineRequest) -> None:
        """Add a new request to the waiting queue.

        The request's status is set to ``WAITING`` before enqueuing.

        Args:
            request: The :class:`EngineRequest` to enqueue.
        """
        request.mark_waiting()
        self.requests[request.request_id] = request
        self.waiting.push(request)
        logger.debug(
            "Request %s added: %d prompt tokens",
            request.request_id,
            request.total_prompt_tokens,
        )

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request if it is still pending or running.

        Args:
            request_id: The unique request identifier.

        Returns:
            ``True`` if the request was found and canceled.
        """
        request = self.requests.get(request_id)
        if request is None:
            return False
        if request_id in self.running:
            request.mark_canceled()
            self._finalize_request(request, reason="canceled")
            return True
        if self.waiting.remove(request_id):
            request.mark_canceled()
            self._finalize_request(request, reason="canceled")
            return True
        return False

    def get_request(self, request_id: str) -> EngineRequest | None:
        """Look up a request by ID.

        Args:
            request_id: The unique request identifier.

        Returns:
            The :class:`EngineRequest`, or ``None`` if not found.
        """
        return self.requests.get(request_id)

    def _alloc_row(self, request_id: str) -> int:
        """Allocate a row index for *request_id*.

        If the request already owns a row, the existing index is
        returned.

        Args:
            request_id: The request to allocate for.

        Returns:
            The assigned row index.

        Raises:
            RuntimeError: If no free rows are available.
        """
        existing = self._row_for_request.get(request_id)
        if existing is not None:
            return existing
        if not self._free_rows:
            raise RuntimeError("No free scheduler rows")
        row = self._free_rows.popleft()
        self._row_for_request[request_id] = row
        return row

    def _release_row(self, request_id: str) -> None:
        """Release the row owned by *request_id*.

        Args:
            request_id: The request whose row to release.
        """
        row = self._row_for_request.pop(request_id, None)
        if row is not None:
            self._free_rows.append(row)

    def _cache_group_for_request(self, request: EngineRequest) -> str:
        """Determine the cache group name for a request.

        Args:
            request: The request to query.

        Returns:
            The cache group name, defaulting to ``"default"``.
        """
        configured = request.metadata.get("cache_group")
        if isinstance(configured, str) and configured.strip():
            return configured
        return "default"

    def _prepare_cache(self, request: EngineRequest) -> None:
        """Acquire cache pages for a request, optionally using prefix caching.

        Args:
            request: The request to prepare.

        Raises:
            RuntimeError: If insufficient pages are available.
        """
        total_target_tokens = min(request.total_prompt_tokens + request.max_new_tokens, self.max_model_len)
        cache_group = self._cache_group_for_request(request)
        prefix_hash = None
        if self._prefix_caching_enabled:
            prefix_hash = prompt_token_hash(request.prompt_token_ids)
        result = self.cache_manager.acquire_for_request(
            request,
            required_tokens=total_target_tokens,
            cache_group=cache_group,
            prefix_hash=prefix_hash,
        )
        request.assign_cache_pages(result.page_ids, cache_group=result.cache_group)
        request.cache_state.prefix_cache_hit = result.prefix_cache_hit
        request.cache_state.prefix_hash = prefix_hash
        if result.prefix_cache_hit:
            request.reset_cached_progress(cached_tokens=request.total_prompt_tokens)
        elif request.status == EngineRequestStatus.PREEMPTED:
            request.reset_cached_progress(cached_tokens=0)

    def _admit_waiting(self) -> tuple[list["str"], list["str"]]:
        """Admit waiting requests into the running set.

        Attempts to promote requests from the waiting queue until the
        running capacity or cache budget is exhausted.  May preempt a
        running request once to free resources.

        Returns:
            A ``(preempted_ids, admitted_ids)`` tuple listing the
            request IDs that were preempted and newly admitted.
        """
        preempted: list["str"] = []
        admitted: list["str"] = []
        preemption_happened = False
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.pop()
            if request.is_finished:
                continue
            try:
                self._alloc_row(request.request_id)
            except RuntimeError:
                break
            try:
                self._prepare_cache(request)
            except RuntimeError:
                if preemption_happened:
                    self._release_row(request.request_id)
                    self.waiting.push_front(request)
                    break
                preempted_request = self._preempt_one()
                if preempted_request is None:
                    self._release_row(request.request_id)
                    self.waiting.push_front(request)
                    break
                preempted.append(preempted_request.request_id)
                preemption_happened = True
                self._prepare_cache(request)
            request.mark_running()
            self.running[request.request_id] = request
            admitted.append(request.request_id)
            if preemption_happened:
                break
        return preempted, admitted

    def _preempt_one(self) -> EngineRequest | None:
        """Preempt the lowest-priority running request.

        The victim is marked as preempted, its cache pages are released,
        and it is pushed back to the front of the waiting queue.

        Returns:
            The preempted :class:`EngineRequest`, or ``None`` if no
            running requests exist.
        """
        if not self.running:
            return None
        candidate = sorted(
            self.running.values(),
            key=lambda request: (request.priority, request.arrival_time),
        )[0]
        self.running.pop(candidate.request_id, None)
        candidate.mark_preempted()
        candidate.cache_state.evicted_pages += len(self.cache_manager.release_request(candidate.request_id))
        candidate.reset_cached_progress(cached_tokens=0)
        self._release_row(candidate.request_id)
        self.waiting.push_front(candidate)
        return candidate

    def _make_prefill_slice(self, request: EngineRequest, budget_available: int) -> tuple[list["int"], int]:
        """Create a prefill token slice for a request.

        Respects chunked-prefill settings when enabled.

        Args:
            request: The request being prefilled.
            budget_available: Remaining token budget.

        Returns:
            A ``(token_ids, token_count)`` tuple.
        """
        remaining = request.remaining_prefill_tokens
        if remaining <= 0 or budget_available <= 0:
            return [], 0
        planned = min(remaining, budget_available)
        threshold = self.scheduler_config.long_prefill_token_threshold
        if bool(self.scheduler_config.chunked_prefill_enabled) and threshold is not None and threshold > 0:
            planned = min(planned, int(threshold))
        start = request.num_computed_tokens
        stop = start + planned
        return request.all_token_ids[start:stop], planned

    def _make_decode_slice(self, request: EngineRequest, budget_available: int) -> tuple[list["int"], int]:
        """Create a single-token decode slice for a request.

        Args:
            request: The request being decoded.
            budget_available: Remaining token budget.

        Returns:
            A ``(token_ids, token_count)`` tuple.
        """
        if budget_available <= 0 or request.remaining_generation_budget <= 0:
            return [], 0
        last_token = (
            request.generated_token_ids[-1]
            if request.generated_token_ids
            else (request.prompt_token_ids[-1] if request.prompt_token_ids else 0)
        )
        return [int(last_token)], 1

    def _iter_running(self) -> list[EngineRequest]:
        """Return running requests in policy-determined order.

        Returns:
            Ordered list of running :class:`EngineRequest` objects.
        """
        requests = list(self.running.values())
        if self.policy == SchedulingPolicy.PRIORITY:
            return sorted(requests, key=lambda request: (-request.priority, request.arrival_time))
        return requests

    def schedule(self) -> SchedulerStepOutput:
        """Produce a scheduling decision for the next engine step.

        Admits waiting requests, allocates token budget across running
        requests, and returns a :class:`SchedulerStepOutput` describing
        the planned work.

        Returns:
            The :class:`SchedulerStepOutput` for this step.
        """
        budget = self._token_budget_manager.make_step_budget()
        logger.debug(
            "Scheduling: budget=%d, running=%d, waiting=%d",
            budget.available,
            len(self.running),
            len(self.waiting),
        )
        preempted_ids, newly_running_ids = self._admit_waiting()
        scheduled: list[ScheduledRequest] = []
        pending_structured = False

        for request in self._iter_running():
            if request.canceled:
                self._finalize_request(request, reason="canceled")
                continue
            if budget.available <= 0:
                break
            token_ids: list["int"]
            token_count: int
            is_prefill = request.remaining_prefill_tokens > 0
            if is_prefill:
                token_ids, token_count = self._make_prefill_slice(request, budget.available)
            else:
                token_ids, token_count = self._make_decode_slice(request, budget.available)
                pending_structured = pending_structured or request.use_structured_output

            granted = budget.consume(token_count)
            if granted <= 0:
                continue
            if granted < len(token_ids):
                token_ids = token_ids[:granted]
            row_index = self._alloc_row(request.request_id)
            scheduled.append(
                ScheduledRequest(
                    request_id=request.request_id,
                    row_index=row_index,
                    token_ids=[int(token_id) for token_id in token_ids],
                    is_prefill=is_prefill,
                    num_tokens=granted,
                    page_ids=self.cache_manager.get_request_pages(request.request_id),
                    cache_group=request.cache_state.cache_group,
                    prefix_cache_hit=request.cache_state.prefix_cache_hit,
                )
            )
            self.cache_manager.touch_request(request.request_id)

        output = SchedulerStepOutput(
            scheduled=scheduled,
            preempted_request_ids=preempted_ids,
            newly_running_request_ids=newly_running_ids,
            num_scheduled=len(scheduled),
            num_waiting=len(self.waiting),
            num_running=len(self.running),
            token_budget_remaining=budget.available,
            decode_only=bool(scheduled) and all(not entry.is_prefill for entry in scheduled),
            pending_structured_output_tokens=pending_structured,
            timestamp=time.time(),
        )
        return output

    def step(self) -> SchedulerStepOutput:
        """Compatibility alias for :meth:`schedule`.

        Returns:
            The :class:`SchedulerStepOutput` for this step.
        """
        return self.schedule()

    def _finalize_request(self, request: EngineRequest, *, reason: str | None = None) -> None:
        """Finalize a request: release resources and update status.

        Optionally caches the prompt prefix for future reuse before
        releasing pages and the row.

        Args:
            request: The request to finalize.
            reason: Finish reason (``"canceled"``, ``"error"``, or a
                normal finish string like ``"eos"`` / ``"length"``).
        """
        self.running.pop(request.request_id, None)
        if (
            reason not in {"canceled", "error"}
            and self._prefix_caching_enabled
            and request.total_prompt_tokens > 0
            and request.cache_state.prefix_hash
        ):
            prompt_pages = self.cache_manager.pages_required(request.total_prompt_tokens)
            if prompt_pages > 0:
                self.cache_manager.cache_prefix(
                    request.cache_state.prefix_hash,
                    request.request_id,
                    max_pages=prompt_pages,
                )
        self.cache_manager.release_request(request.request_id)
        self._release_row(request.request_id)
        if reason == "canceled":
            request.mark_canceled()
        elif reason == "error":
            if request.failure_reason is None:
                request.mark_failed("unknown")
        else:
            request.mark_finished(reason)

    def update_from_model_output(
        self,
        step_output: SchedulerStepOutput,
        *,
        sampled_token_ids: dict[str, list["int"]] | None = None,
        stop_reasons: dict[str, str | None] | None = None,
        failed_requests: dict[str, str] | None = None,
    ) -> list[EngineRequest]:
        """Apply model output to the scheduler state and finalize requests.

        For each scheduled request, updates computed-token counts and
        generated tokens.  Checks for EOS tokens, generation-budget
        exhaustion, stop reasons, and failures, finalizing requests as
        appropriate.

        Args:
            step_output: The scheduling decision that produced this
                output.
            sampled_token_ids: Mapping of request IDs to sampled token
                ID lists.
            stop_reasons: Mapping of request IDs to stop-reason strings.
            failed_requests: Mapping of request IDs to failure
                descriptions.

        Returns:
            List of :class:`EngineRequest` objects that finished during
            this update.
        """
        sampled_token_ids = sampled_token_ids or {}
        stop_reasons = stop_reasons or {}
        failed_requests = failed_requests or {}
        finished: list[EngineRequest] = []

        for scheduled in step_output.scheduled:
            request = self.requests.get(scheduled.request_id)
            if request is None or request.is_finished:
                continue
            request.consume_computed_tokens(scheduled.num_tokens)

            for token in sampled_token_ids.get(request.request_id, []):
                request.append_generated_token(token)

            if request.request_id in failed_requests:
                request.mark_failed(failed_requests[request.request_id])
                self._finalize_request(request, reason="error")
                finished.append(request)
                continue

            stop_reason = stop_reasons.get(request.request_id)
            if stop_reason:
                request.mark_finished(stop_reason)
                self._finalize_request(request, reason=stop_reason)
                finished.append(request)
                continue

            eos_token = request.eos_token_id
            if eos_token is None and request.sampling_params is not None:
                eos_token = request.sampling_params.eos_token_id
            generated = sampled_token_ids.get(request.request_id, [])
            if generated and eos_token is not None and int(generated[-1]) == int(eos_token):
                self._finalize_request(request, reason="eos")
                finished.append(request)
                continue

            if request.remaining_prefill_tokens == 0 and request.remaining_generation_budget <= 0:
                self._finalize_request(request, reason="length")
                finished.append(request)
                continue

        if finished:
            logger.debug("Finished %d request(s): %s", len(finished), [r.request_id for r in finished])
        step_output.finished_request_ids.extend(request.request_id for request in finished)
        return finished

    def pause(self) -> None:
        """Pause all running requests by setting their status to ``PAUSED``."""
        for request in self.running.values():
            request.mark_paused()

    def resume(self) -> None:
        """Resume all paused running requests by setting status to ``RUNNING``."""
        for request in self.running.values():
            request.mark_running()

    def reset(self) -> None:
        """Reset the scheduler to its initial empty state.

        Clears all queues, running sets, row mappings, and the cache
        manager.
        """
        self.waiting = create_request_queue(self.policy)
        self.running.clear()
        self.requests.clear()
        self._row_for_request.clear()
        self._free_rows = deque(range(self.max_num_running_reqs))
        self.cache_manager.reset()

    def has_pending_work(self) -> bool:
        """Return whether any requests are waiting or running.

        Returns:
            ``True`` if the waiting queue or running set is non-empty.
        """
        return bool(self.waiting) or bool(self.running)

    def row_index_for_request(self, request_id: str) -> int | None:
        """Look up the row index assigned to a request.

        Args:
            request_id: The unique request identifier.

        Returns:
            The row index, or ``None`` if the request has no row.
        """
        return self._row_for_request.get(request_id)

    def remap_rows(self, moves: list[tuple[int, int]]) -> None:
        """Update the row-index mapping after a compaction operation.

        Args:
            moves: List of ``(from_index, to_index)`` row moves.
        """
        if not moves:
            return
        by_old_row = {row_index: request_id for request_id, row_index in self._row_for_request.items()}
        for _from_index, to_index in moves:
            request_id = by_old_row.get(_from_index)
            if request_id is None:
                continue
            self._row_for_request[request_id] = to_index
            by_old_row[to_index] = request_id
            by_old_row.pop(_from_index, None)
        occupied = set(self._row_for_request.values())
        self._free_rows = deque(row_index for row_index in range(self.max_num_running_reqs) if row_index not in occupied)


class LocalScheduler(Scheduler):
    """Backward-compatible scheduler name used by imports/tests.

    Inherits all behaviour from :class:`Scheduler` without modification.
    """
