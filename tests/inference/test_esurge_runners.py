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

"""Focused tests for the refactored eSurge runner stack."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from easymlx.inference.esurge.runners import (
    ExecutionManager,
    ExecutionRequest,
    ModelRunner,
    ScheduledSequence,
    SequenceBuffer,
)
from easymlx.inference.esurge.core.sampler import argmax_token
from easymlx.inference.esurge.runners.model_runner import _as_mx_logits
from easymlx.inference.esurge.sampling_params import SamplingParams


class DummyPagedModel:
    """Deterministic paged model used for runner tests."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size

    def __call__(self, input_ids=None, *, positions=None, slot_ids=None, **_kwargs):
        del positions
        logits = np.full((len(input_ids), self.vocab_size), -50.0, dtype=np.float32)
        for index, token_ids in enumerate(input_ids):
            target = (int(sum(token_ids)) + int(slot_ids[index])) % self.vocab_size
            logits[index, target] = 10.0
        return {"logits": logits}


class DummyCompiledPagedModel:
    """MLX-native paged model that can run through the compiled runner path."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size

    def __call__(self, input_ids=None, *, cache_views=None, cache_metadata=None, return_dict=True):
        slot_ids = cache_metadata.slot_ids or ()
        slot_array = mx.array(list(slot_ids), dtype=mx.int32)
        cache = cache_views[0]
        cache.kv_lens[slot_array] = cache.kv_lens[slot_array] + 1

        last_indices = (cache_metadata.query_start_loc[1:] - 1).astype(mx.int32)
        last_tokens = mx.take(input_ids, last_indices, axis=0).astype(mx.int32)
        logits = mx.take(mx.eye(self.vocab_size, dtype=mx.float32), last_tokens % self.vocab_size, axis=0)

        if return_dict:
            return {"logits": logits}
        return logits


class DummyDecodeStepPagedModel:
    """Model that exposes a single-token decode fast path for paged serving."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.call_count = 0
        self.decode_step_count = 0

    def __call__(self, input_ids=None, *, cache_views=None, cache_metadata=None, return_dict=True, **_kwargs):
        del input_ids, cache_views, cache_metadata
        self.call_count += 1
        logits = mx.zeros((1, self.vocab_size), dtype=mx.float32)
        if return_dict:
            return {"logits": logits}
        return logits

    def decode_step(self, input_ids, *, cache_views=None, cache_metadata=None):
        del input_ids, cache_views, cache_metadata
        self.decode_step_count += 1
        return mx.zeros((1, self.vocab_size), dtype=mx.float32)


def test_sequence_buffer_row_ops_and_page_table() -> None:
    buffer = SequenceBuffer(max_num_rows=6)
    row0 = buffer.begin_sequence("req-0", [1, 2], page_ids=[10, 11], row_index=0)
    row2 = buffer.begin_sequence("req-2", [3], page_ids=[20], row_index=2)
    row1 = buffer.begin_sequence("req-1", [4], page_ids=[30, 31])
    assert (row0, row1, row2) == (0, 1, 2)

    removed = buffer.remove_sequence("req-1")
    assert removed.request_id == "req-1"

    moved = buffer.compact_holes(0, 3)
    assert moved == [(2, 1)]
    assert buffer.get_row_index("req-2") == 1

    buffer.swap_rows(0, 1)
    assert buffer.get_row_index("req-0") == 1
    assert buffer.get_row_index("req-2") == 0

    buffer.move_row(1, 3)
    assert buffer.get_row_index("req-0") == 3

    page_table = buffer.page_table()
    assert isinstance(page_table, mx.array)
    assert page_table.shape == (4, 2)
    assert int(page_table[0, 0].item()) == 20
    assert int(page_table[3, 0].item()) == 10


def test_model_runner_normalizes_step_updates_and_mutates_buffer() -> None:
    model = DummyPagedModel(vocab_size=7)
    buffer = SequenceBuffer()
    buffer.begin_sequence("req-a", [10], row_index=0)
    buffer.begin_sequence("req-b", [20], row_index=1)
    runner = ModelRunner(model, sequence_buffer=buffer, seed=0)

    request = ExecutionRequest(
        step_id=42,
        mode="mixed",
        sequences=[
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[10], num_computed_tokens=1),
            ScheduledSequence(request_id="req-b", row_index=1, token_ids=[20], num_computed_tokens=1),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False),
            "req-b": SamplingParams(max_tokens=1, do_sample=False),
        },
    )

    result = runner.run(request)
    assert result.step_id == 42
    assert result.req_ids == ["req-a", "req-b"]
    assert result.metadata["mode"] == "mixed"
    assert result.sampled_token_ids == [[3], [0]]
    assert buffer.get_row("req-a").output_token_ids == [3]
    assert buffer.get_row("req-b").output_token_ids == [0]


def test_as_mx_logits_handles_mlx_bfloat16() -> None:
    logits = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4).astype(mx.bfloat16)
    array = _as_mx_logits(logits)
    assert isinstance(array, mx.array)
    assert array.shape == (1, 4)
    assert array.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(array), np.arange(8, 12, dtype=np.float32).reshape(1, 4))


def test_model_runner_pad_batch_token_ids_stays_mlx_and_compiles() -> None:
    runner = ModelRunner(DummyPagedModel(vocab_size=5), seed=0)

    batch, query_lens = runner._pad_batch_token_ids(
        [
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[1, 2, 3], num_computed_tokens=0),
            ScheduledSequence(request_id="req-b", row_index=1, token_ids=[4], num_computed_tokens=0),
        ]
    )

    assert isinstance(batch, mx.array)
    assert query_lens == [3, 1]
    assert batch.shape == (2, 3)
    assert len(runner._compiled_batch_padders) == 1
    np.testing.assert_array_equal(np.asarray(batch), np.array([[1, 2, 3], [4, 0, 0]], dtype=np.int32))


def test_argmax_token_accepts_mlx_logits() -> None:
    token_id = argmax_token(mx.array([-1.0, 5.0, 2.0], dtype=mx.float32))
    assert token_id == 1


def test_sampling_params_to_generation_kwargs_includes_penalties() -> None:
    params = SamplingParams(max_tokens=4, presence_penalty=0.7, repetition_penalty=1.4)

    kwargs = params.to_generation_kwargs()

    assert kwargs["presence_penalty"] == pytest.approx(0.7)
    assert kwargs["repetition_penalty"] == pytest.approx(1.4)


def test_model_runner_uses_compiled_forward_for_stable_paged_models(monkeypatch) -> None:
    from easymlx.inference.esurge.runners import model_runner as model_runner_module
    cache_module = pytest.importorskip("easymlx.caching")
    PageCacheView = cache_module.PageCacheView

    model = DummyCompiledPagedModel(vocab_size=6)
    cache = PageCacheView.allocate(
        num_seqs=2,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=1,
        block_size=2,
        dtype=mx.float32,
    )
    runner = model_runner_module.ModelRunner(model, kv_caches=[cache], seed=0, use_compiled_forward=True)

    request = model_runner_module.ExecutionRequest(
        step_id=11,
        sequences=[
            model_runner_module.ScheduledSequence(request_id="req-a", row_index=0, token_ids=[1], num_computed_tokens=0),
            model_runner_module.ScheduledSequence(request_id="req-b", row_index=1, token_ids=[2], num_computed_tokens=0),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False),
            "req-b": SamplingParams(max_tokens=1, do_sample=False),
        },
    )

    _raw_output, logits = runner._forward_step(request)
    assert len(runner._compiled_forwards) == 1
    assert logits.shape == (2, 6)
    np.testing.assert_array_equal(np.asarray(cache.kv_lens), np.array([1, 1], dtype=np.int32))

    _raw_output, logits = runner._forward_step(request)
    assert len(runner._compiled_forwards) == 1
    assert logits.shape == (2, 6)
    np.testing.assert_array_equal(np.asarray(cache.kv_lens), np.array([2, 2], dtype=np.int32))


def test_model_runner_prefers_decode_step_for_single_token_paged_decode() -> None:
    cache_module = pytest.importorskip("easymlx.caching")
    PageCacheView = cache_module.PageCacheView

    model = DummyDecodeStepPagedModel(vocab_size=4)
    cache = PageCacheView.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=1,
        block_size=2,
        dtype=mx.float32,
    )
    runner = ModelRunner(model, kv_caches=[cache], seed=0, use_compiled_forward=True)

    request = ExecutionRequest(
        step_id=17,
        sequences=[
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[1], num_computed_tokens=1),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False),
        },
    )

    _raw_output, logits = runner._forward_step(request)

    assert logits.shape == (1, 4)
    assert model.decode_step_count == 1
    assert model.call_count == 0
    assert len(runner._compiled_forwards) == 0


def test_model_runner_run_samples_from_mlx_logits_without_numpy_materialization() -> None:
    class DummyMxModel:
        def __call__(self, input_ids=None, **_kwargs):
            logits = mx.array(
                [
                    [1.0, 5.0, -1.0],
                    [0.5, -2.0, 4.0],
                ],
                dtype=mx.float32,
            )
            return {"logits": logits}

    model = DummyMxModel()
    runner = ModelRunner(model, seed=0)

    request = ExecutionRequest(
        step_id=5,
        sequences=[
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[1], num_computed_tokens=0),
            ScheduledSequence(request_id="req-b", row_index=1, token_ids=[2], num_computed_tokens=0),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False),
            "req-b": SamplingParams(max_tokens=1, do_sample=False),
        },
    )

    result = runner.run(request)

    assert result.sampled_token_ids == [[1], [2]]
    assert isinstance(result.logits, mx.array)
    assert result.logits.shape == (2, 3)


def test_model_runner_mlx_sampling_applies_top_k_top_p_before_categorical(monkeypatch) -> None:
    runner = ModelRunner(DummyPagedModel(vocab_size=5), seed=0)
    logits = mx.array([[0.5, -4.0, 10.0, 9.0, 1.0]], dtype=mx.float32)
    sampling_params = SamplingParams(max_tokens=1, do_sample=True, temperature=1.0, top_k=2, top_p=0.6)

    def fake_categorical(filtered_logits, axis=-1, key=None):
        del axis, key
        filtered = np.asarray(filtered_logits)
        assert filtered.shape == (1, 2)
        assert filtered[0, 0] > 0.0
        assert filtered[0, 1] < -1e20
        return mx.array([0], dtype=mx.uint32)

    monkeypatch.setattr(mx, "compile", lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(mx.random, "categorical", fake_categorical)
    sampled = runner._sample_next_tokens_mx(logits, [sampling_params])

    assert sampled == [2]


def test_model_runner_reuses_compiled_sampler_for_same_shape_and_params(monkeypatch) -> None:
    compile_calls = 0
    original_compile = mx.compile

    def tracked_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(mx, "compile", tracked_compile)

    runner = ModelRunner(DummyPagedModel(vocab_size=5), seed=0)
    logits = mx.array([[0.5, -4.0, 10.0, 9.0, 1.0]], dtype=mx.float32)
    sampling_params = SamplingParams(max_tokens=1, do_sample=True, temperature=1.0, top_k=2, top_p=0.6)

    runner._sample_next_tokens_mx(logits, [sampling_params])
    runner._sample_next_tokens_mx(logits, [sampling_params])

    assert compile_calls == 1


def test_model_runner_does_not_advance_rng_for_greedy_groups(monkeypatch) -> None:
    runner = ModelRunner(DummyPagedModel(vocab_size=5), seed=0)
    logits = mx.array(
        [
            [0.5, -4.0, 10.0, 9.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=mx.float32,
    )
    sampling_params = [
        SamplingParams(max_tokens=1, do_sample=False),
        SamplingParams(max_tokens=1, do_sample=True, temperature=1.0, top_k=2, top_p=0.6),
    ]

    key_requests = 0

    def fake_next_sample_key():
        nonlocal key_requests
        key_requests += 1
        return mx.random.key(key_requests)

    def fake_get_compiled_sampler(_shape, *, do_sample, temperature, top_k, top_p):
        del temperature, top_k, top_p

        def sampler(group_logits, rng_key):
            del rng_key
            if do_sample:
                return mx.full((group_logits.shape[0],), 4, dtype=mx.int32)
            return mx.argmax(group_logits, axis=-1).astype(mx.int32)

        return sampler

    monkeypatch.setattr(runner, "_next_sample_key", fake_next_sample_key)
    monkeypatch.setattr(runner, "_get_compiled_sampler", fake_get_compiled_sampler)

    sampled = runner._sample_next_tokens_mx(logits, sampling_params)

    assert sampled == [2, 4]
    assert key_requests == 1


def test_model_runner_run_applies_repetition_penalty_from_token_history() -> None:
    class FixedLogitModel:
        def __call__(self, input_ids=None, **_kwargs):
            batch = np.asarray(input_ids).shape[0]
            logits = mx.repeat(
                mx.array([[9.0, 10.0, -1.0]], dtype=mx.float32),
                repeats=batch,
                axis=0,
            )
            return {"logits": logits}

    buffer = SequenceBuffer()
    buffer.begin_sequence("req-a", [1], row_index=0)
    runner = ModelRunner(FixedLogitModel(), sequence_buffer=buffer, seed=0)

    request = ExecutionRequest(
        step_id=23,
        sequences=[
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[7], num_computed_tokens=1),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False, repetition_penalty=2.0),
        },
    )

    result = runner.run(request)

    assert result.sampled_token_ids == [[0]]


def test_model_runner_sample_next_token_applies_presence_penalty_only_to_generated_history() -> None:
    runner = ModelRunner(DummyPagedModel(vocab_size=3), seed=0)
    logits = mx.array([9.0, 10.0, -1.0], dtype=mx.float32)
    sampling_params = SamplingParams(max_tokens=1, do_sample=False, presence_penalty=2.0)

    prompt_only = runner.sample_next_token(
        logits,
        sampling_params=sampling_params,
        prompt_token_ids=[1],
        generated_token_ids=[],
    )
    generated_repeat = runner.sample_next_token(
        logits,
        sampling_params=sampling_params,
        prompt_token_ids=[],
        generated_token_ids=[1],
    )

    assert prompt_only == 1
    assert generated_repeat == 0


def test_execution_manager_async_collect() -> None:
    model = DummyPagedModel(vocab_size=5)
    runner = ModelRunner(model, seed=0)
    manager = ExecutionManager(runner)
    request = ExecutionRequest(
        step_id=7,
        sequences=[ScheduledSequence(request_id="req-x", row_index=0, token_ids=[2], num_computed_tokens=0)],
        sampling_by_request={"req-x": SamplingParams(max_tokens=1, do_sample=False)},
    )

    pending = manager.execute_async(request)
    result = manager.collect(pending, timeout=1.0)
    assert result.step_id == 7
    assert result.req_ids == ["req-x"]
    assert result.sampled_token_ids == [[2]]
    manager.close()
