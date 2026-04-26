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

"""Refactor-focused runtime tests for the scheduler/runner eSurge engine."""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import numpy as np
import pytest
from easymlx.inference.esurge import CacheConfig, Config, SamplingParams, SchedulerConfig, eSurge
from easymlx.inference.esurge.request import EngineRequest
from easymlx.inference.esurge.runners import ModelRunner, SequenceBuffer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "out": 6,
        "x": 7,
        "y": 8,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


class TracePagedCache:
    def __init__(self, num_seqs: int, *, blocks_per_seq: int = 16, block_size: int = 2):
        self.num_seqs = num_seqs
        self.block_size = block_size
        self.block_tables = np.full((num_seqs, blocks_per_seq), -1, dtype=np.int32)
        for seq_idx in range(num_seqs):
            start = seq_idx * blocks_per_seq
            self.block_tables[seq_idx, :] = np.arange(start, start + blocks_per_seq, dtype=np.int32)
        self.kv_lens = np.zeros((num_seqs,), dtype=np.int32)
        shape = (num_seqs * blocks_per_seq, block_size, 1, 1)
        self.key_cache = np.zeros(shape, dtype=np.float16)
        self.value_cache = np.zeros_like(self.key_cache)

    @property
    def cache(self):
        """Compatibility property: operations may access ``view.cache``."""
        return self

    def reset(self, seq_idx: int) -> None:
        self.kv_lens[int(seq_idx)] = 0
        self.block_tables[int(seq_idx), :] = -1


class LoggingPagedModel:
    vocab_size = 9

    def __init__(self, *, sleep_s: float = 0.0):
        self.sleep_s = float(sleep_s)
        self.call_log: list[dict[str, object]] = []
        self.multimodal_log: list[object] = []

    def init_paged_cache(self, *, num_seqs: int, **_kwargs):
        return [TracePagedCache(num_seqs)]

    def __call__(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        query_lens=None,
        slot_ids=None,
        multimodal=None,
        multimodal_inputs=None,
        return_dict=True,
        **_kwargs,
    ):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        arr = np.array(input_ids, dtype=np.int32)
        self.call_log.append(
            {
                "input_ids": arr.copy(),
                "query_lens": list(query_lens) if query_lens else [],
                "slot_ids": list(slot_ids) if slot_ids else [],
            }
        )
        self.multimodal_log.append(multimodal if multimodal is not None else multimodal_inputs)

        num_seqs = len(query_lens) if query_lens else arr.shape[0]
        logits = np.full((num_seqs, self.vocab_size), -1e9, dtype=np.float32)
        offset = 0
        for row in range(num_seqs):
            qlen = int(query_lens[row]) if query_lens else (arr.shape[1] if arr.ndim > 1 else arr.shape[0])
            sid = int(slot_ids[row]) if slot_ids else row
            if arr.ndim == 1:
                last_token = int(arr[offset + qlen - 1])
            else:
                last_token = int(arr[row, qlen - 1])
            offset += qlen
            if cache_views:
                cache_views[0].cache.kv_lens[sid] += qlen
            next_token = 6
            if last_token == 6:
                next_token = 1
            logits[row, next_token] = 0.0
        if return_dict:
            return type("DummyOutput", (), {"logits": logits})()
        return logits


class OverlapProbePagedModel(LoggingPagedModel):
    def __init__(self, *, sleep_s: float = 0.05, fail: bool = False):
        super().__init__(sleep_s=sleep_s)
        self.fail = bool(fail)
        self.in_call = threading.Event()
        self.future_pending = threading.Event()
        self.schedule_calls_while_running = 0

    def __call__(self, *args, **kwargs):
        self.in_call.set()
        try:
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)
            if self.fail:
                raise RuntimeError("probe failure")
            original_sleep = self.sleep_s
            self.sleep_s = 0.0
            try:
                return super().__call__(*args, **kwargs)
            finally:
                self.sleep_s = original_sleep
        finally:
            self.in_call.clear()


class _WarmupStateModule:
    def __init__(self) -> None:
        self.reset_count = 0
        self.state = 0

    def reset_state(self, batch_size: int = 1) -> None:
        del batch_size
        self.reset_count += 1
        self.state = 0


class WarmupStatePagedModel(LoggingPagedModel):
    def __init__(self) -> None:
        super().__init__()
        self.state_module = _WarmupStateModule()
        self.layers = [type("Layer", (), {"linear_attn": self.state_module})()]

    def __call__(self, *args, **kwargs):
        self.state_module.state = 1
        output = super().__call__(*args, **kwargs)
        logits = mx.array(output.logits if hasattr(output, "logits") else output)
        return type("DummyOutput", (), {"logits": logits})()


class FakeDistributedController:
    enabled = True

    def __init__(self) -> None:
        self.started = 0
        self.dispatched = 0
        self.verified = 0
        self.shutdowns = 0

    def start(self) -> None:
        self.started += 1

    def dispatch_step(self, scheduler_output) -> dict[str, int]:
        self.dispatched += 1
        return {"count": len(scheduler_output.scheduled)}

    def verify_step(self, dispatch, model_output) -> None:
        assert dispatch["count"] >= 1
        assert model_output is not None
        self.verified += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


class FakeMultimodalPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def prepare_batch(self, *, requests, runtime_states, step_output):
        self.calls += 1
        assert len(requests) == len(runtime_states) == len(step_output.scheduled)
        return {"kind": "fake-mm", "request_ids": [request.request_id for request in requests]}


def _engine_config(*, max_num_seqs: int = 4, prefix: bool = False) -> Config:
    return Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=4,
            max_model_len=32,
            chunked_prefill_enabled=True,
            long_prefill_token_threshold=2,
            max_num_seq_buckets=(1, 2, 4, 8),
        ),
        cache_config=CacheConfig(num_pages=64, page_size=2, enable_prefix_caching=prefix),
    )


def _overlap_config() -> Config:
    return Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=1,
            max_model_len=16,
            chunked_prefill_enabled=True,
            long_prefill_token_threshold=1,
        ),
        cache_config=CacheConfig(num_pages=16, page_size=2, enable_prefix_caching=False),
    )


def _record_schedule_during_model_call(engine: eSurge, model: OverlapProbePagedModel) -> None:
    assert engine._scheduler is not None
    assert engine._execution_manager is not None
    original_schedule = engine._scheduler.schedule
    original_execute_async = engine._execution_manager.execute_async
    original_collect = engine._execution_manager.collect

    def wrapped_execute_async(*args, **kwargs):
        pending = original_execute_async(*args, **kwargs)
        model.future_pending.set()
        return pending

    def wrapped_collect(*args, **kwargs):
        try:
            return original_collect(*args, **kwargs)
        finally:
            model.future_pending.clear()

    def wrapped_schedule(*args, **kwargs):
        if model.future_pending.is_set():
            model.schedule_calls_while_running += 1
        return original_schedule(*args, **kwargs)

    engine._execution_manager.execute_async = wrapped_execute_async
    engine._execution_manager.collect = wrapped_collect
    engine._scheduler.schedule = wrapped_schedule


def test_esurge_overlap_prefetches_scheduler_step_while_runner_future_is_pending() -> None:
    tokenizer = _build_tokenizer()
    model = OverlapProbePagedModel(sleep_s=0.05)
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_overlap_config(),
        reserve_tokens=2,
        compile_runner=False,
    )
    _record_schedule_during_model_call(engine, model)
    try:
        outputs = engine.generate(["a", "a"], SamplingParams(max_tokens=1))
    finally:
        engine.close()

    assert len(outputs) == 2
    assert model.schedule_calls_while_running >= 1


def test_esurge_overlap_drains_prefetched_scheduler_state_on_runner_exception() -> None:
    tokenizer = _build_tokenizer()
    model = OverlapProbePagedModel(sleep_s=0.05, fail=True)
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_overlap_config(),
        reserve_tokens=2,
        compile_runner=False,
    )
    _record_schedule_during_model_call(engine, model)
    try:
        with pytest.raises(RuntimeError, match="probe failure"):
            engine.generate(["a", "a"], SamplingParams(max_tokens=1))
    finally:
        engine.close()

    assert model.schedule_calls_while_running >= 1


def test_esurge_accepts_easydel_constructor_surface_as_mlx_runtime_config() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel()
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_engine_config(max_num_seqs=2),
        reserve_tokens=2,
        hbm_utilization=0.5,
        min_input_pad=1,
        min_token_pad=1,
        async_scheduling=False,
        overlap_execution=False,
        compile_runner=False,
        use_aot_forward=False,
        bind_graphstate_for_aot=True,
        auto_shard_model=False,
        sharding_axis_dims=(1,),
        sampler_metrics=True,
        data_parallelism_axis="tp",
        esurge_name="probe",
    )
    try:
        assert engine.hbm_utilization == 0.5
        assert engine.memory_utilization == 0.5
        assert engine.min_input_pad == 1
        assert engine.min_token_pad == 1
        assert engine.async_scheduling is False
        assert engine.overlap_execution is False
        assert engine.compile_runner is False
        assert engine.esurge_name == "probe"
    finally:
        engine.close()


def test_esurge_resets_recurrent_state_after_compile_warmup() -> None:
    tokenizer = _build_tokenizer()
    model = WarmupStatePagedModel()
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_engine_config(max_num_seqs=2),
        reserve_tokens=2,
        compile_runner=True,
    )
    try:
        assert model.state_module.reset_count >= 2
        assert model.state_module.state == 0
    finally:
        engine.close()


def test_esurge_rejects_overlap_execution_in_distributed_lockstep_mode() -> None:
    tokenizer = _build_tokenizer()
    controller = FakeDistributedController()
    with pytest.raises(ValueError, match="overlap_execution=True"):
        eSurge(
            LoggingPagedModel(),
            tokenizer=tokenizer,
            config=_engine_config(max_num_seqs=2),
            reserve_tokens=2,
            distributed_controller=controller,
        )


def test_esurge_mixed_prefill_decode_continuous_batching() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel()
    engine = eSurge(model, tokenizer=tokenizer, config=_engine_config(max_num_seqs=2), reserve_tokens=2)
    try:
        outputs = engine.generate(["a", "a b c d"], SamplingParams(max_tokens=3))
        assert len(outputs) == 2
        mixed_step_seen = False
        for call in model.call_log:
            query_lens = call["query_lens"]
            if len(query_lens) >= 2 and any(ql == 1 for ql in query_lens) and any(ql > 1 for ql in query_lens):
                mixed_step_seen = True
                break
        assert mixed_step_seen is True
    finally:
        engine.close()


def test_esurge_prefix_cache_hits_on_repeated_prompt() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel()
    engine = eSurge(model, tokenizer=tokenizer, config=_engine_config(prefix=True), reserve_tokens=2)
    try:
        engine.generate("a b", SamplingParams(max_tokens=1))
        engine.generate("a b", SamplingParams(max_tokens=1))
        assert engine._scheduler is not None
        stats = engine._scheduler.cache_manager.stats()
        assert stats["prefix_hits"] >= 1
    finally:
        engine.close()


def test_esurge_pause_resume_blocks_and_recovers_generation() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel(sleep_s=0.02)
    engine = eSurge(model, tokenizer=tokenizer, config=_engine_config(), reserve_tokens=2)
    result: dict[str, object] = {}

    try:
        engine.pause()

        def _run() -> None:
            result["output"] = engine.generate("a", SamplingParams(max_tokens=1))

        thread = threading.Thread(target=_run)
        thread.start()
        time.sleep(0.05)
        assert thread.is_alive() is True
        assert model.call_log == []

        engine.resume()
        thread.join(timeout=2.0)
        assert thread.is_alive() is False
        assert result["output"][0].finished is True
    finally:
        engine.close()


def test_esurge_abort_running_request() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel(sleep_s=0.005)
    engine = eSurge(model, tokenizer=tokenizer, config=_engine_config(), reserve_tokens=2)
    try:
        iterator = engine.stream("a", SamplingParams(max_tokens=4))
        first = next(iterator)
        assert first.request_id
        assert engine.abort_request(first.request_id) is True
        rest = list(iterator)
        assert rest
        assert rest[-1].finished is True
        assert rest[-1].outputs[0].finish_reason == "canceled"
    finally:
        engine.close()


def test_esurge_wires_distributed_and_multimodal_seams() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel()
    controller = FakeDistributedController()
    multimodal = FakeMultimodalPreprocessor()
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_engine_config(),
        reserve_tokens=2,
        distributed_controller=controller,
        multimodal_preprocessor=multimodal,
        overlap_execution=False,
    )
    try:
        engine.generate("a", SamplingParams(max_tokens=1))
    finally:
        engine.close()
    assert controller.started == 1
    assert controller.dispatched >= 1
    assert controller.verified >= 1
    assert controller.shutdowns >= 1
    assert multimodal.calls >= 1
    assert model.multimodal_log[-1] is not None


def test_model_runner_compaction_updates_cache_rows_and_bucket_selection() -> None:
    cache = TracePagedCache(num_seqs=4, blocks_per_seq=4, block_size=2)
    model = LoggingPagedModel()
    buffer = SequenceBuffer(max_num_rows=4)
    runner = ModelRunner(
        model,
        sequence_buffer=buffer,
        kv_caches=[cache],
        max_num_seq_buckets=(1, 2, 4),
        seed=0,
    )

    req_a = EngineRequest(
        request_id="req-a",
        prompt="a",
        prompt_token_ids=[3],
        sampling_params=SamplingParams(max_tokens=1),
    )
    req_a.assign_cache_pages([10, 11])
    runner.bind_request(req_a, row_index=0)

    req_b = EngineRequest(
        request_id="req-b",
        prompt="b",
        prompt_token_ids=[4],
        sampling_params=SamplingParams(max_tokens=1),
    )
    req_b.assign_cache_pages([20, 21])
    runner.bind_request(req_b, row_index=2)

    runner.detach_request("req-a")
    moves = runner.compact_rows()

    assert moves == [(2, 0)]
    assert buffer.get_row_index("req-b") == 0
    assert cache.block_tables[0, 0] == 20
    assert cache.block_tables[2, 0] == -1
    assert runner.select_bucket_size(3) == 4


def test_model_runner_bind_avoids_clearing_assigned_pages() -> None:
    cache = TracePagedCache(num_seqs=1, blocks_per_seq=4, block_size=2)
    cache.key_cache[1] = 7
    cache.value_cache[1] = 9
    runner = ModelRunner(LoggingPagedModel(), kv_caches=[cache], seed=0)
    request = EngineRequest(
        request_id="req-a",
        prompt="a",
        prompt_token_ids=[3],
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.assign_cache_pages([1, 2])

    runner.bind_request(request, row_index=0)

    assert cache.block_tables[0, :2].tolist() == [1, 2]
    assert cache.kv_lens[0] == 0
    np.testing.assert_array_equal(cache.key_cache[1], np.full_like(cache.key_cache[1], 7))
    np.testing.assert_array_equal(cache.value_cache[1], np.full_like(cache.value_cache[1], 9))


def test_esurge_batched_runtime_beats_naive_sequential_baseline() -> None:
    tokenizer = _build_tokenizer()
    model = LoggingPagedModel(sleep_s=0.01)
    engine = eSurge(model, tokenizer=tokenizer, config=_engine_config(max_num_seqs=8), reserve_tokens=2)
    prompts = ["a", "a", "a", "a"]
    params = SamplingParams(max_tokens=4)

    def _naive_sequential(prompt_count: int, *, prompt_steps: int, decode_steps: int, sleep_s: float) -> float:
        started = time.perf_counter()
        for _ in range(prompt_count):
            for _ in range(prompt_steps + decode_steps):
                time.sleep(sleep_s)
        return time.perf_counter() - started

    try:
        started = time.perf_counter()
        outputs = engine.generate(prompts, params)
        batched_elapsed = time.perf_counter() - started
        assert len(outputs) == len(prompts)
    finally:
        engine.close()

    baseline_elapsed = _naive_sequential(len(prompts), prompt_steps=1, decode_steps=4, sleep_s=model.sleep_s)
    assert batched_elapsed < baseline_elapsed * 0.75
