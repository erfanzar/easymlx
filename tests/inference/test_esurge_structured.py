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

"""Structured-output and convenience-loading tests for easymlx eSurge."""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from easymlx.inference.esurge import CacheConfig, Config, SamplingParams, SchedulerConfig, eSurge
from easymlx.inference.esurge.esurge_engine import _MemoryUtilizationSummary
from easymlx.inference.esurge.runners.model_runner import ModelRunner
from easymlx.modules.llama import LlamaConfig, LlamaForCausalLM


class _DummyPagedCache:
    def __init__(self, num_seqs: int):
        self.num_seqs = num_seqs
        self.kv_lens = np.zeros((num_seqs,), dtype=np.int32)

    @property
    def cache(self):
        return self

    def reset(self, seq_idx: int) -> None:
        self.kv_lens[int(seq_idx)] = 0


class DummyModel:
    """Paged-compatible dummy that emits a fixed token sequence then EOS."""

    def __init__(
        self,
        append_tokens: list[int],
        *,
        vocab_size: int = 16,
        eos_token_id: int = 1,
        sleep_s: float = 0.0,
        return_mx_logits: bool = False,
    ):
        self.append_tokens = list(append_tokens)
        self.vocab_size = vocab_size
        self._eos = eos_token_id
        self._step: dict[int, int] = {}
        self.sleep_s = float(sleep_s)
        self.return_mx_logits = bool(return_mx_logits)
        self.multimodal_log: list[object | None] = []

    def init_paged_cache(self, *, num_seqs: int, **_kwargs):
        return [_DummyPagedCache(num_seqs)]

    def __call__(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        query_lens=None,
        slot_ids=None,
        return_dict=True,
        **_kwargs,
    ):
        if self.sleep_s > 0.0:
            time.sleep(self.sleep_s)
        arr = np.array(input_ids, dtype=np.int32)
        num_seqs = len(query_lens) if query_lens else arr.shape[0]
        logits = np.full((num_seqs, self.vocab_size), -1e9, dtype=np.float32)
        self.multimodal_log.append(_kwargs.get("multimodal") or _kwargs.get("multimodal_inputs"))
        offset = 0
        for row in range(num_seqs):
            qlen = int(query_lens[row]) if query_lens else (arr.shape[1] if arr.ndim > 1 else arr.shape[0])
            sid = int(slot_ids[row]) if slot_ids else row
            if cache_views and int(cache_views[0].cache.kv_lens[sid]) == 0:
                self._step[sid] = 0
            if cache_views:
                cache_views[0].cache.kv_lens[sid] += qlen
            step = self._step.get(sid, 0)
            if step < len(self.append_tokens):
                next_token = self.append_tokens[step]
            else:
                next_token = self._eos
            self._step[sid] = step + 1
            logits[row, next_token] = 0.0
            offset += qlen
        result_logits = mx.array(logits) if self.return_mx_logits else logits
        if return_dict:
            return type("Out", (), {"logits": result_logits})()
        return result_logits


class _DecodeStepOnlyModel(DummyModel):
    def __init__(self, append_tokens: list[int], *, vocab_size: int = 16, eos_token_id: int = 1):
        super().__init__(append_tokens, vocab_size=vocab_size, eos_token_id=eos_token_id, return_mx_logits=True)
        self.decode_calls = 0
        self.forward_calls = 0

    def __call__(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("decode_step path should bypass __call__")

    def decode_step(self, input_ids, *, cache_views=None, cache_metadata=None):
        del cache_views, cache_metadata
        self.decode_calls += 1
        arr = np.array(input_ids, dtype=np.int32)
        batch_size = arr.shape[0] if arr.ndim > 1 else 1
        logits = np.full((batch_size, self.vocab_size), -1e9, dtype=np.float32)
        logits[:, self.append_tokens[0]] = 0.0
        return mx.array(logits)


class FakeDistributedController:
    def __init__(self) -> None:
        self.enabled = True
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


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "hello": 3,
        "<think>plan</think>": 4,
        "answer": 5,
        '<tool_call>{"name":"lookup","arguments":{"city":"paris"}}</tool_call>': 6,
        "tok3": 7,
        "tok4": 8,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    mx.eval(model.parameters())
    return model


def _flatten_params(model: LlamaForCausalLM) -> dict[str, mx.array]:
    return tree_flatten(model.parameters(), destination={})


def _sync_test_config(*, prefix: bool = False) -> Config:
    return Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=8,
            max_model_len=32,
            chunked_prefill_enabled=True,
            long_prefill_token_threshold=2,
        ),
        cache_config=CacheConfig(num_pages=32, page_size=2, enable_prefix_caching=prefix),
    )


def test_esurge_extracts_reasoning_content():
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([4, 5]),
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        enable_background=False,
        reasoning_parser="deepseek_r1",
    )

    output = engine.generate("hello", SamplingParams(max_tokens=2))[0]

    assert output.outputs[0].text == "answer"
    assert output.outputs[0].reasoning_content == "plan"
    assert output.reasoning_content == "plan"


def test_model_runner_exposes_decode_step_helper():
    model = _DecodeStepOnlyModel([5])
    runner = ModelRunner(
        model,
        max_model_len=16,
        max_num_batched_tokens=8,
        max_num_seqs=1,
        use_compiled_forward=True,
    )
    raw_logits = runner._call_decode_step(input_ids=mx.array([[3]], dtype=mx.int32))

    assert callable(runner._decode_step_fn)
    assert model.decode_calls == 1
    assert model.forward_calls == 0
    np.testing.assert_array_equal(np.asarray(raw_logits).argmax(axis=-1), np.array([5]))


def test_esurge_logs_reduced_memory_utilization_when_runtime_cap_is_lower(monkeypatch):
    import easymlx.inference.esurge.esurge_engine as esurge_engine_module

    messages: list[str] = []
    monkeypatch.setattr(
        esurge_engine_module.logger,
        "info",
        lambda msg, *args: messages.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        esurge_engine_module.eSurge,
        "_build_memory_utilization_summary",
        lambda self, _kv_caches: _MemoryUtilizationSummary(
            requested_utilization=0.5,
            allocated_utilization=0.2352941176,
            estimated_token_capacity=17_000,
            runtime_token_capacity=8_000,
            requested_budget_bytes=17_000,
            allocated_cache_bytes=8_000,
            available_memory_bytes=34_000,
            runtime_capped=True,
        ),
    )

    engine = eSurge(
        DummyModel([5]),
        tokenizer=_build_tokenizer(),
        max_model_len=8_000,
        max_num_seqs=1,
        reserve_tokens=2,
    )
    try:
        summary = "\n".join(messages)
        assert "Memory util   : requested=50% | allocated~24% | est_capacity~17,000 tok" in summary
        assert "runtime cap 8,000 tok (8,000 x 1)" in summary
        assert "reduced to the runtime cap" in summary
        assert engine.effective_memory_utilization == pytest.approx(0.2352941176)
        assert engine.estimated_memory_token_capacity == 17_000
    finally:
        engine.close()


def test_esurge_logs_approximate_sequence_capacity_when_memory_target_is_tighter(monkeypatch):
    import easymlx.inference.esurge.esurge_engine as esurge_engine_module

    messages: list[str] = []
    monkeypatch.setattr(
        esurge_engine_module.logger,
        "info",
        lambda msg, *args: messages.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        esurge_engine_module.eSurge,
        "_build_memory_utilization_summary",
        lambda self, _kv_caches: _MemoryUtilizationSummary(
            requested_utilization=0.5,
            allocated_utilization=0.67,
            estimated_token_capacity=6_000,
            runtime_token_capacity=8_000,
            requested_budget_bytes=6_000,
            allocated_cache_bytes=8_000,
            available_memory_bytes=12_000,
            approximate_sequence_capacity=True,
        ),
    )

    engine = eSurge(
        DummyModel([5]),
        tokenizer=_build_tokenizer(),
        max_model_len=4_000,
        max_num_seqs=2,
        reserve_tokens=2,
    )
    try:
        summary = "\n".join(messages)
        assert "Memory util   : requested=50% | allocated~67% | est_capacity~6,000 tok" in summary
        assert "below the runtime cap of 8,000" in summary
        assert "not an exact per-sequence limit" in summary
        assert "depends on concurrent sequence lengths and page rounding" in summary
        assert engine.effective_memory_utilization == pytest.approx(0.67)
        assert engine.estimated_memory_token_capacity == 6_000
    finally:
        engine.close()


def test_esurge_extracts_tool_calls():
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([6]),
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        enable_background=False,
        tool_parser="hermes",
    )

    output = engine.generate("hello", SamplingParams(max_tokens=1))[0]

    assert output.outputs[0].finish_reason == "tool_calls"
    assert output.outputs[0].text == ""
    assert output.outputs[0].tool_calls is not None
    assert output.outputs[0].tool_calls[0]["function"]["name"] == "lookup"


def test_esurge_sync_path_handles_stop_strings() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5]),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
    )

    try:
        output = engine.generate("hello", SamplingParams(max_tokens=2, stop=["answer"]))[0]
    finally:
        engine.close()

    assert output.outputs[0].finish_reason == "stop"
    assert output.outputs[0].token_ids == [5]
    assert output.outputs[0].text == ""


def test_esurge_sync_path_greedy_samples_mx_logits() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5], return_mx_logits=True),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
    )

    try:
        output = engine.generate("hello", SamplingParams(max_tokens=1))[0]
    finally:
        engine.close()

    assert output.finished is True
    assert output.outputs[0].token_ids == [5]
    assert output.outputs[0].text.strip() == "answer"


def test_esurge_sync_path_respects_pause_resume() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5], sleep_s=0.02),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
    )
    result: dict[str, object] = {}

    try:
        engine.pause()

        def _run() -> None:
            result["output"] = engine.generate("hello", SamplingParams(max_tokens=1))

        thread = threading.Thread(target=_run)
        thread.start()
        time.sleep(0.05)
        assert thread.is_alive() is True

        engine.resume()
        thread.join(timeout=2.0)
        assert thread.is_alive() is False
        assert result["output"][0].finished is True
    finally:
        engine.close()


def test_esurge_sync_path_prefix_cache_hits_on_repeated_prompt() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5]),
        tokenizer=tokenizer,
        config=_sync_test_config(prefix=True),
        reserve_tokens=2,
    )

    try:
        engine.generate("hello", SamplingParams(max_tokens=1))
        engine.generate("hello", SamplingParams(max_tokens=1))
        assert engine._scheduler is not None
        stats = engine._scheduler.cache_manager.stats()
    finally:
        engine.close()

    assert stats["prefix_hits"] >= 1


def test_esurge_sync_stream_is_incremental() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5]),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
    )

    try:
        outputs = list(engine.stream("hello", SamplingParams(max_tokens=2)))
    finally:
        engine.close()

    assert len(outputs) >= 1
    assert outputs[0].delta_text.strip() == "answer"
    assert outputs[-1].finished is True


def test_esurge_autodetected_tool_parser_is_disabled_without_tools() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([4, 5]),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
        reasoning_parser="deepseek_r1",
    )

    try:
        gen_kwargs = engine._resolve_generation_kwargs(SamplingParams(max_tokens=2))
        gen_kwargs["max_new_tokens"] = 2
        states = engine._build_requests(["hello"], gen_kwargs, max_new_tokens=2)
        outputs = list(engine.stream("hello", SamplingParams(max_tokens=2)))
    finally:
        engine.close()

    assert engine.tool_parser_name == "hermes"
    assert states[0].tool_parser_instance is None
    assert len(outputs) == 2
    assert outputs[0].delta_reasoning_content == "plan"
    assert outputs[0].delta_text == ""
    assert outputs[1].delta_text == "answer"
    assert outputs[-1].reasoning_content == "plan"


def test_esurge_sync_stream_abort_request() -> None:
    tokenizer = _build_tokenizer()
    engine = eSurge(
        DummyModel([5, 5, 5], sleep_s=0.01),
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
    )

    try:
        iterator = engine.stream("hello", SamplingParams(max_tokens=4))
        first = next(iterator)
        assert first.request_id
        assert engine.abort_request(first.request_id) is True
        rest = list(iterator)
    finally:
        engine.close()

    assert rest
    assert rest[-1].finished is True
    assert rest[-1].outputs[0].finish_reason == "canceled"


def test_esurge_sync_path_wires_distributed_and_multimodal_seams() -> None:
    tokenizer = _build_tokenizer()
    model = DummyModel([5])
    controller = FakeDistributedController()
    multimodal = FakeMultimodalPreprocessor()
    engine = eSurge(
        model,
        tokenizer=tokenizer,
        config=_sync_test_config(),
        reserve_tokens=2,
        distributed_controller=controller,
        multimodal_preprocessor=multimodal,
        multimodal_payload={"kind": "fake"},
    )

    try:
        output = engine.generate("hello", SamplingParams(max_tokens=1))[0]
    finally:
        engine.close()

    assert output.finished is True
    assert controller.started == 1
    assert controller.dispatched >= 1
    assert controller.verified >= 1
    assert controller.shutdowns == 1
    assert multimodal.calls >= 1
    assert model.multimodal_log[-1] is not None
