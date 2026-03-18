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

"""Focused tests for eSurge request/core/scheduler runtime layers."""

from __future__ import annotations

from easymlx.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easymlx.inference.esurge.core.manager import CacheManager
from easymlx.inference.esurge.core.page_pool import PagePool
from easymlx.inference.esurge.request import EngineRequest, EngineRequestStatus
from easymlx.inference.esurge.sampling_params import SamplingParams
from easymlx.inference.esurge.scheduler import AsyncScheduler, Scheduler


def _request(request_id: str, prompt_ids: list[int], *, max_tokens: int = 2, priority: int = 0) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt=f"prompt-{request_id}",
        prompt_token_ids=list(prompt_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        priority=priority,
    )


def test_engine_request_state_and_counters() -> None:
    request = _request("r1", [1, 2, 3], max_tokens=2)
    assert request.remaining_prefill_tokens == 3
    request.consume_computed_tokens(2)
    assert request.remaining_prefill_tokens == 1
    request.append_generated_token(9)
    assert request.num_generated_tokens == 1
    assert request.remaining_generation_budget == 1
    request.mark_preempted()
    assert request.status == EngineRequestStatus.PREEMPTED
    assert request.preemptions == 1
    request.mark_finished("length")
    assert request.is_finished is True
    assert request.finished_reason == "length"


def test_page_pool_allocate_release_roundtrip() -> None:
    pool = PagePool(capacity=3)
    pages = pool.allocate(2)
    assert pages == [0, 1]
    assert pool.free_count() == 1
    released = pool.release(pages)
    assert released == [0, 1]
    assert pool.free_count() == 3


def test_cache_manager_prefix_and_eviction() -> None:
    cache = CacheManager(num_pages=3, page_size=2, enable_prefix_caching=True)
    request1 = _request("r1", [1, 2, 3, 4], max_tokens=2)
    request2 = _request("r2", [1, 2, 3, 4], max_tokens=2)

    prefix_hash = cache.prefix_hash_from_tokens(request1.prompt_token_ids)
    result1 = cache.acquire_for_request(request1, required_tokens=6, prefix_hash=prefix_hash)
    assert len(result1.page_ids) == 3
    cache.cache_prefix(prefix_hash, request1.request_id)
    cache.release_request(request1.request_id)

    result2 = cache.acquire_for_request(request2, required_tokens=6, prefix_hash=prefix_hash)
    assert result2.prefix_cache_hit is True
    assert len(result2.page_ids) == 3

    cache.release_request(request2.request_id)
    request3 = _request("r3", [9, 9, 9, 9], max_tokens=2)
    cache.acquire_for_request(request3, required_tokens=6, prefix_hash=None)
    stats = cache.stats()
    assert stats["evictions"] >= 1


def test_scheduler_prefill_then_decode_then_finish() -> None:
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            max_model_len=16,
            chunked_prefill_enabled=True,
            long_prefill_token_threshold=2,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=2, enable_prefix_caching=True),
    )
    scheduler = Scheduler(config)
    request = _request("r1", [1, 2], max_tokens=2)
    scheduler.add_request(request)

    first = scheduler.schedule()
    assert first.num_scheduled == 1
    assert first.scheduled[0].is_prefill is True
    scheduler.update_from_model_output(first)

    second = scheduler.schedule()
    assert second.scheduled[0].is_prefill is False
    scheduler.update_from_model_output(second, sampled_token_ids={"r1": [9]})

    third = scheduler.schedule()
    assert third.scheduled[0].is_prefill is False
    finished = scheduler.update_from_model_output(third, sampled_token_ids={"r1": [8]})
    assert len(finished) == 1
    assert finished[0].finished_reason == "length"


def test_scheduler_preemption_on_cache_pressure() -> None:
    config = Config(
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, max_model_len=16, policy="fcfs"),
        cache_config=CacheConfig(num_pages=2, page_size=2, enable_prefix_caching=False),
    )
    scheduler = Scheduler(config)
    low = _request("low", [1, 2], max_tokens=2, priority=0)
    high = _request("high", [4, 5], max_tokens=2, priority=10)
    scheduler.add_request(low)
    scheduler.add_request(high)

    step = scheduler.schedule()
    assert "low" in step.preempted_request_ids
    assert "high" in step.newly_running_request_ids
    assert scheduler.get_request("low") is not None
    assert scheduler.get_request("low").status in {EngineRequestStatus.PREEMPTED, EngineRequestStatus.WAITING}


def test_scheduler_cancel_queued_and_running() -> None:
    config = Config(
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, max_model_len=16),
        cache_config=CacheConfig(num_pages=16, page_size=2, enable_prefix_caching=False),
    )
    scheduler = Scheduler(config)
    first = _request("r1", [1, 2], max_tokens=1)
    second = _request("r2", [3, 4], max_tokens=1)
    scheduler.add_request(first)
    scheduler.add_request(second)

    assert scheduler.cancel_request("r2") is True
    assert scheduler.get_request("r2").status == EngineRequestStatus.CANCELED

    scheduler.schedule()
    assert scheduler.cancel_request("r1") is True
    assert scheduler.get_request("r1").status == EngineRequestStatus.CANCELED


def test_scheduler_determinism_for_identical_inputs() -> None:
    config = Config(
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=6, max_model_len=32, policy="priority"),
        cache_config=CacheConfig(num_pages=32, page_size=2, enable_prefix_caching=False),
    )
    scheduler_a = Scheduler(config)
    scheduler_b = Scheduler(config)
    requests = [
        _request("a", [1, 2, 3], max_tokens=2, priority=1),
        _request("b", [4, 5], max_tokens=2, priority=5),
        _request("c", [6, 7, 8], max_tokens=2, priority=1),
    ]
    for request in requests:
        scheduler_a.add_request(_request(request.request_id, request.prompt_token_ids, max_tokens=2, priority=request.priority))
        scheduler_b.add_request(_request(request.request_id, request.prompt_token_ids, max_tokens=2, priority=request.priority))

    out_a = scheduler_a.schedule()
    out_b = scheduler_b.schedule()
    sig_a = [(entry.request_id, entry.is_prefill, entry.num_tokens) for entry in out_a.scheduled]
    sig_b = [(entry.request_id, entry.is_prefill, entry.num_tokens) for entry in out_b.scheduled]
    assert sig_a == sig_b


def test_async_scheduler_placeholder_accounting() -> None:
    config = Config(
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, max_model_len=16),
        cache_config=CacheConfig(num_pages=16, page_size=2, enable_prefix_caching=False),
    )
    scheduler = AsyncScheduler(config)
    request = _request("r1", [1], max_tokens=1)
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    scheduler.update_from_model_output(prefill)
    decode = scheduler.schedule()
    assert scheduler.get_request("r1").num_output_placeholders == 1
    scheduler.update_from_model_output(decode, sampled_token_ids={"r1": [2]})
    assert scheduler.get_request("r1").num_output_placeholders == 0
