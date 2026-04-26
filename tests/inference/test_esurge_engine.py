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

"""Tests for the easymlx eSurge engine (paged attention only)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from easymlx.caching import PageCacheView
from easymlx.inference.esurge import (
    CacheConfig,
    Config,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
    eSurge,
    esurge_engine,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


class DummyPagedModel:
    """Minimal model using real PageCacheView for eSurge tests."""

    vocab_size = 6

    def init_paged_cache(self, *, num_seqs: int, max_seq_len: int = 16, page_size: int = 16, **_kwargs):
        return [
            PageCacheView.allocate(
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                num_kv_heads=1,
                head_dim=4,
                block_size=page_size,
                dtype=mx.float16,
            )
        ]

    def __call__(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        query_lens=None,
        slot_ids=None,
        positions=None,
        return_dict=True,
        **_kwargs,
    ):
        arr = np.array(input_ids, dtype=np.int32)
        if arr.ndim == 1:
            arr.shape[0]
            if query_lens and slot_ids:
                num_seqs = len(slot_ids)
                logits = np.full((num_seqs, self.vocab_size), -1e9, dtype=np.float32)
                offset = 0
                for row, (qlen, sid) in enumerate(zip(query_lens, slot_ids, strict=False)):
                    qlen = int(qlen)
                    last_token = int(arr[offset + qlen - 1])
                    offset += qlen
                    if cache_views:
                        cache_views[0].cache.kv_lens[sid] += qlen
                    next_token = 5 if last_token != 5 else 1
                    logits[row, next_token] = 0.0
            else:
                logits = np.full((1, self.vocab_size), -1e9, dtype=np.float32)
                logits[0, 5] = 0.0
        else:
            batch_size = arr.shape[0]
            logits = np.full((batch_size, self.vocab_size), -1e9, dtype=np.float32)
            for row in range(batch_size):
                qlen = int(query_lens[row]) if query_lens else arr.shape[1]
                last_token = int(arr[row, qlen - 1])
                if cache_views and slot_ids:
                    cache_views[0].cache.kv_lens[int(slot_ids[row])] += qlen
                next_token = 5 if last_token != 5 else 1
                logits[row, next_token] = 0.0

        if return_dict:
            return type("DummyOutput", (), {"logits": logits})()
        return logits


class DtypeRecordingPagedModel(DummyPagedModel):
    """Paged dummy that records the dtype requested by eSurge."""

    def __init__(self, parameter_dtype=mx.bfloat16):
        self.config = type(
            "DummyConfig",
            (),
            {
                "model_type": "dummy",
                "vocab_size": self.vocab_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "hidden_size": 4,
            },
        )()
        self._parameter = mx.zeros((1,), dtype=parameter_dtype)
        self.recorded_dtype = None

    def parameters(self):
        return {"weight": self._parameter}

    def init_paged_cache(
        self,
        *,
        num_seqs: int,
        max_seq_len: int = 16,
        page_size: int = 16,
        dtype=mx.float16,
        **_kwargs,
    ):
        self.recorded_dtype = dtype
        return [
            PageCacheView.allocate(
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                num_kv_heads=1,
                head_dim=4,
                block_size=page_size,
                dtype=dtype,
            )
        ]


def _build_tokenizer():
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "hello": 3,
        "world": 4,
        "foo": 5,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


class PlannedPagedModel:
    """Minimal paged model that emits a fixed token progression."""

    vocab_size = 8

    def init_paged_cache(self, *, num_seqs: int, max_seq_len: int = 16, page_size: int = 16, **_kwargs):
        return [
            PageCacheView.allocate(
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                num_kv_heads=1,
                head_dim=4,
                block_size=page_size,
                dtype=mx.float16,
            )
        ]

    def __call__(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        query_lens=None,
        slot_ids=None,
        positions=None,
        return_dict=True,
        **_kwargs,
    ):
        del cache_metadata, positions
        arr = np.array(input_ids, dtype=np.int32)
        logits: np.ndarray
        next_token_by_last = {
            2: 3,
            3: 4,
            4: 5,
            5: 6,
        }

        if arr.ndim == 1:
            if query_lens and slot_ids:
                num_seqs = len(slot_ids)
                logits = np.full((num_seqs, self.vocab_size), -1e9, dtype=np.float32)
                offset = 0
                for row, (qlen, sid) in enumerate(zip(query_lens, slot_ids, strict=False)):
                    qlen = int(qlen)
                    last_token = int(arr[offset + qlen - 1])
                    offset += qlen
                    if cache_views:
                        cache_views[0].cache.kv_lens[sid] += qlen
                    logits[row, next_token_by_last.get(last_token, 6)] = 0.0
            else:
                logits = np.full((1, self.vocab_size), -1e9, dtype=np.float32)
                logits[0, next_token_by_last.get(int(arr[-1]), 6)] = 0.0
        else:
            batch_size = arr.shape[0]
            logits = np.full((batch_size, self.vocab_size), -1e9, dtype=np.float32)
            for row in range(batch_size):
                qlen = int(query_lens[row]) if query_lens else arr.shape[1]
                last_token = int(arr[row, qlen - 1])
                if cache_views and slot_ids:
                    cache_views[0].cache.kv_lens[int(slot_ids[row])] += qlen
                logits[row, next_token_by_last.get(last_token, 6)] = 0.0

        if return_dict:
            return type("DummyOutput", (), {"logits": logits})()
        return logits


def _build_partial_decode_tokenizer():
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "hello": 2,
        "alpha": 3,
        "dino": 4,
        "beta": 5,
        "robot": 6,
        "<unk>": 7,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    original_decode = tokenizer.decode
    partial_decodes = {
        (3,): "alpha\ufffd",
        (3, 4): "alpha🦕",
        (3, 4, 5): "alpha🦕beta\ufffd",
        (3, 4, 5, 6): "alpha🦕beta🤖",
    }

    def fake_decode(token_ids, skip_special_tokens=True, **kwargs):
        ids = tuple(int(token_id) for token_id in token_ids)
        if skip_special_tokens:
            special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
            ids = tuple(token_id for token_id in ids if token_id not in special_ids)
        if ids in partial_decodes:
            return partial_decodes[ids]
        return original_decode(list(ids), skip_special_tokens=False, **kwargs)

    tokenizer.decode = fake_decode
    return tokenizer


class FullContextSpeculativeModel:
    """Paged-capable test model that also returns full-context logits."""

    vocab_size = 9

    def __init__(self, *, overrides: dict[int, int] | None = None):
        self.overrides = dict(overrides or {})
        self.full_context_calls = 0
        self.paged_calls = 0

    def init_paged_cache(self, *, num_seqs: int, max_seq_len: int = 16, page_size: int = 16, **_kwargs):
        return [
            PageCacheView.allocate(
                num_seqs=num_seqs,
                max_seq_len=max_seq_len,
                num_kv_heads=1,
                head_dim=4,
                block_size=page_size,
                dtype=mx.float16,
            )
        ]

    def _next_token(self, token_id: int) -> int:
        if token_id in self.overrides:
            return int(self.overrides[token_id])
        return {
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 1,
        }.get(int(token_id), 1)

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
        del cache_metadata
        arr = np.array(input_ids, dtype=np.int32)
        if query_lens and slot_ids:
            self.paged_calls += 1
            logits = np.full((len(slot_ids), self.vocab_size), -1e9, dtype=np.float32)
            offset = 0
            for row, (qlen, sid) in enumerate(zip(query_lens, slot_ids, strict=False)):
                qlen = int(qlen)
                last_token = int(arr[offset + qlen - 1] if arr.ndim == 1 else arr[row, qlen - 1])
                offset += qlen
                if cache_views:
                    cache_views[0].cache.kv_lens[int(sid)] += qlen
                logits[row, self._next_token(last_token)] = 0.0
        else:
            self.full_context_calls += 1
            row = arr[0] if arr.ndim == 2 else arr
            logits = np.full((1, len(row), self.vocab_size), -1e9, dtype=np.float32)
            for pos, token_id in enumerate(row):
                logits[0, pos, self._next_token(int(token_id))] = 0.0

        if return_dict:
            return type("DummyOutput", (), {"logits": logits})()
        return logits


class CachedSpeculativeModel(FullContextSpeculativeModel):
    """Speculative test model exposing init_operations_cache for fast path."""

    def __init__(self, *, overrides: dict[int, int] | None = None):
        super().__init__(overrides=overrides)
        self.cached_calls = 0

    def init_operations_cache(
        self,
        *,
        batch_size: int,
        max_length: int,
        page_size: int = 16,
        dtype=mx.float16,
        cache_type: str | None = None,
        **_kwargs,
    ):
        del cache_type
        return [
            PageCacheView.allocate(
                num_seqs=batch_size,
                max_seq_len=max_length,
                num_kv_heads=1,
                head_dim=4,
                block_size=page_size,
                dtype=dtype,
            )
        ]

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
        if cache_metadata is None:
            return super().__call__(
                input_ids,
                cache_views=cache_views,
                cache_metadata=cache_metadata,
                query_lens=query_lens,
                slot_ids=slot_ids,
                return_dict=return_dict,
                **_kwargs,
            )

        self.cached_calls += 1
        arr = np.array(input_ids, dtype=np.int32)
        row = arr[0] if arr.ndim == 2 else arr
        logits = np.full((1, len(row), self.vocab_size), -1e9, dtype=np.float32)
        for pos, token_id in enumerate(row):
            logits[0, pos, self._next_token(int(token_id))] = 0.0
        if cache_views:
            cache_views[0].kv_lens[0] = int(cache_views[0].kv_lens[0].item()) + len(row)

        if return_dict:
            return type("DummyOutput", (), {"logits": logits})()
        return logits


class CachedDFlashTargetModel(CachedSpeculativeModel):
    """Cached target exposing hidden features for DFlash tests."""

    def _hidden(self, row) -> tuple[mx.array]:
        values = np.array([[[float(token_id), float(token_id + 10)] for token_id in row]], dtype=np.float32)
        return (mx.array(values),)

    def __call__(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        query_lens=None,
        slot_ids=None,
        return_dict=True,
        output_hidden_states=False,
        **kwargs,
    ):
        output = super().__call__(
            input_ids,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
            query_lens=query_lens,
            slot_ids=slot_ids,
            return_dict=True,
            **kwargs,
        )
        if output_hidden_states:
            row = np.array(input_ids, dtype=np.int32)
            if row.ndim == 2:
                row = row[0]
            output.hidden_states = self._hidden(row)
        if return_dict:
            return output
        return output.logits

    def decode_step_with_state_and_hidden(
        self,
        input_ids,
        *,
        cache_views=None,
        cache_metadata=None,
        decode_state=None,
        feature_layer_indices=None,
        **_kwargs,
    ):
        del decode_state, feature_layer_indices
        output = self(
            input_ids,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
            return_dict=True,
            output_hidden_states=True,
        )
        return mx.array(output.logits), None, output.hidden_states


class FakeDFlashAdapter:
    speculative_kind = "dflash"
    target_feature_layer_indices = (1,)

    def __init__(self):
        self.calls = 0
        self.cache_ids: list[int] = []
        self.hidden_shapes: list[tuple[int, ...]] = []

    def make_cache(self):
        return [{"updates": 0}]

    def _next_token(self, token_id: int) -> int:
        return {
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 1,
        }.get(int(token_id), 1)

    def dflash_logits(self, *, first_token, hidden_states, max_tokens, cache=None, **_kwargs):
        self.calls += 1
        if cache is not None:
            self.cache_ids.append(id(cache))
            cache[0]["updates"] += 1
        hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        self.hidden_shapes.append(tuple(int(dim) for dim in hidden.shape))
        logits = np.full((int(max_tokens), 9), -1e9, dtype=np.float32)
        token = int(first_token)
        for idx in range(int(max_tokens)):
            token = self._next_token(token)
            logits[idx, token] = 0.0
        return mx.array(logits)


class WrongDFlashAdapter(FakeDFlashAdapter):
    """DFlash adapter that always proposes the wrong continuation token."""

    def _next_token(self, token_id: int) -> int:
        del token_id
        return 7


class RaisingDraftModel:
    def __call__(self, **_kwargs):
        raise RuntimeError("draft is unavailable")


class EAGLE3FeatureTargetModel(FullContextSpeculativeModel):
    """Target model exposing hidden-state features for EAGLE3 tests."""

    def __init__(self, *, overrides: dict[int, int] | None = None):
        super().__init__(overrides=overrides)
        self.hidden_state_calls = 0

    def eagle3_hidden_states(self, input_ids, *, feature_layer_indices=None, **_kwargs):
        del feature_layer_indices
        self.hidden_state_calls += 1
        row = np.array(input_ids, dtype=np.int32)
        if row.ndim == 2:
            row = row[0]
        values = np.array([[[float(token_id), float(token_id + 10)] for token_id in row]], dtype=np.float32)
        hidden = mx.array(values)
        return (hidden, hidden + 1.0, hidden + 2.0)


class FakeEAGLE3Adapter:
    """Linear EAGLE3 adapter used to exercise the eSurge integration point."""

    def __init__(self, *, overrides: dict[int, int] | None = None):
        self.overrides = dict(overrides or {})
        self.calls = 0
        self.feature_shapes: list[tuple[int, ...]] = []

    def _next_token(self, token_id: int) -> int:
        if token_id in self.overrides:
            return int(self.overrides[token_id])
        return {
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 1,
        }.get(int(token_id), 1)

    def propose_eagle3(self, token_ids, *, eagle3_features, max_tokens, **_kwargs):
        assert eagle3_features is not None
        self.calls += 1
        self.feature_shapes.append(tuple(int(dim) for dim in eagle3_features.shape))
        context = [int(token_id) for token_id in token_ids]
        draft_tokens: list[int] = []
        for _ in range(int(max_tokens)):
            token_id = self._next_token(context[-1])
            draft_tokens.append(token_id)
            context.append(token_id)
        return draft_tokens


def test_esurge_generate_basic():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello world", SamplingParams(max_tokens=2))
    assert len(outputs) == 1
    out = outputs[0]
    assert out.outputs[0].token_ids == [5, 1]
    assert out.outputs[0].text.strip() == "foo"


def test_esurge_speculative_decoding_accepts_draft_tokens():
    tokenizer = _build_partial_decode_tokenizer()
    target = FullContextSpeculativeModel()
    draft = FullContextSpeculativeModel()
    config = Config(
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_model_len=16),
        cache_config=CacheConfig(page_size=4),
        speculative_config=SpeculativeConfig(num_speculative_tokens=2, speculative_model=draft),
    )
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        config=config,
        reserve_tokens=2,
        compile_runner=False,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=4))[0]

        assert output.outputs[0].token_ids == [3, 4, 5, 6]
        assert output.outputs[0].text == "alpha🦕beta🤖"
        assert target.full_context_calls == 2
        assert target.paged_calls == 0
        assert output.metrics is not None
        assert output.metrics["speculative_decoding"] is True
        assert output.metrics["speculative_draft_tokens"] == 3
        assert output.metrics["speculative_accepted_tokens"] == 3
        assert output.metrics["speculative_rejected_tokens"] == 0
        assert output.metrics["speculative_target_steps"] == 2
    finally:
        engine.close()


def test_esurge_speculative_decoding_rejects_wrong_draft_token():
    tokenizer = _build_partial_decode_tokenizer()
    target = FullContextSpeculativeModel()
    draft = FullContextSpeculativeModel(overrides={3: 7})
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=draft,
        num_speculative_tokens=2,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=2))[0]

        assert output.outputs[0].token_ids == [3, 4]
        assert output.metrics is not None
        assert output.metrics["speculative_accepted_tokens"] == 1
        assert output.metrics["speculative_rejected_tokens"] == 1
    finally:
        engine.close()


def test_esurge_speculative_decoding_falls_back_when_draft_is_unavailable():
    tokenizer = _build_partial_decode_tokenizer()
    target = FullContextSpeculativeModel()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=RaisingDraftModel(),
        num_speculative_tokens=2,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=2))[0]

        assert output.outputs[0].token_ids == [3, 4]
        assert output.metrics is not None
        assert "speculative_decoding" not in output.metrics
        assert target.paged_calls > 0
    finally:
        engine.close()


def test_esurge_speculative_decoding_uses_cached_fast_path_when_available():
    tokenizer = _build_partial_decode_tokenizer()
    target = CachedSpeculativeModel()
    draft = CachedSpeculativeModel()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=draft,
        num_speculative_tokens=2,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=4))[0]

        assert output.outputs[0].token_ids == [3, 4, 5, 6]
        assert target.full_context_calls == 0
        assert draft.full_context_calls == 0
        assert target.cached_calls > 0
        assert draft.cached_calls > 0
        assert output.metrics is not None
        assert output.metrics["speculative_method"] == "draft"
        assert output.metrics["speculative_accepted_tokens"] == 3
        assert output.metrics["speculative_target_steps"] == 3
        assert target.cached_calls <= 4
    finally:
        engine.close()


def test_esurge_dflash_speculative_decoding_uses_cached_fast_path_when_available():
    tokenizer = _build_partial_decode_tokenizer()
    target = CachedDFlashTargetModel()
    adapter = FakeDFlashAdapter()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=adapter,
        speculative_method="dflash",
        num_speculative_tokens=3,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=4))[0]

        assert output.outputs[0].token_ids == [3, 4, 5, 6]
        assert target.full_context_calls <= 1
        assert target.cached_calls > 0
        assert adapter.calls == 1
        assert len(set(adapter.cache_ids)) == 1
        assert adapter.hidden_shapes[0] == (1, 1, 2)
        assert output.metrics is not None
        assert output.metrics["speculative_method"] == "dflash"
        assert output.metrics["speculative_accepted_tokens"] == 4
        assert output.metrics["speculative_rejected_tokens"] == 0
    finally:
        engine.close()


def test_esurge_stream_uses_dflash_speculative_decoding_when_greedy():
    tokenizer = _build_partial_decode_tokenizer()
    target = CachedDFlashTargetModel()
    adapter = FakeDFlashAdapter()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=adapter,
        speculative_method="dflash",
        num_speculative_tokens=3,
    )
    try:
        outputs = list(engine.stream("hello", SamplingParams(max_tokens=4, do_sample=False)))

        assert outputs
        assert outputs[-1].finished is True
        assert outputs[-1].outputs[0].token_ids == [3, 4, 5, 6]
        assert outputs[-1].accumulated_text == "alpha🦕beta🤖"
        assert target.full_context_calls <= 1
        assert target.cached_calls > 0
        assert adapter.calls == 1
        assert outputs[-1].metrics is not None
        assert outputs[-1].metrics["speculative_decoding"] is True
        assert outputs[-1].metrics["speculative_method"] == "dflash"
        assert outputs[-1].metrics["speculative_accepted_tokens"] == 4
    finally:
        engine.close()


def test_esurge_dflash_adaptively_falls_back_on_low_acceptance():
    class AlternatingDFlashTargetModel(CachedDFlashTargetModel):
        def _next_token(self, token_id: int) -> int:
            return 4 if int(token_id) == 3 else 3

    tokenizer = _build_partial_decode_tokenizer()
    target = AlternatingDFlashTargetModel()
    adapter = WrongDFlashAdapter()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=96,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=adapter,
        speculative_method="dflash",
        num_speculative_tokens=3,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=40, do_sample=False))[0]

        assert output.outputs[0].token_ids == [3 if idx % 2 == 0 else 4 for idx in range(40)]
        assert output.metrics is not None
        assert output.metrics["speculative_method"] == "dflash"
        assert output.metrics["speculative_fallback_triggered"] is True
        assert output.metrics["speculative_fallback_tokens"] > 0
        assert output.metrics["speculative_acceptance_rate"] < 0.70
        assert adapter.calls < 25
    finally:
        engine.close()


def test_esurge_eagle3_speculative_decoding_accepts_adapter_tokens():
    tokenizer = _build_partial_decode_tokenizer()
    target = EAGLE3FeatureTargetModel()
    adapter = FakeEAGLE3Adapter()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=adapter,
        speculative_method="eagle3",
        num_speculative_tokens=2,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=4))[0]

        assert output.outputs[0].token_ids == [3, 4, 5, 6]
        assert adapter.calls > 0
        assert target.hidden_state_calls > 0
        assert adapter.feature_shapes[0][-1] == 6
        assert output.metrics is not None
        assert output.metrics["speculative_decoding"] is True
        assert output.metrics["speculative_method"] == "eagle3"
        assert output.metrics["speculative_accepted_tokens"] == 3
        assert output.metrics["speculative_rejected_tokens"] == 0
    finally:
        engine.close()


def test_esurge_eagle3_speculative_decoding_falls_back_without_hidden_states():
    tokenizer = _build_partial_decode_tokenizer()
    target = FullContextSpeculativeModel()
    adapter = FakeEAGLE3Adapter()
    engine = eSurge(
        target,
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        speculative_model=adapter,
        speculative_method="eagle3",
        num_speculative_tokens=2,
    )
    try:
        output = engine.generate("hello", SamplingParams(max_tokens=2))[0]

        assert output.outputs[0].token_ids == [3, 4]
        assert adapter.calls == 0
        assert output.metrics is not None
        assert "speculative_decoding" not in output.metrics
        assert target.paged_calls > 0
    finally:
        engine.close()


def test_esurge_infers_bfloat16_cache_dtype_from_model_parameters():
    model = DtypeRecordingPagedModel(parameter_dtype=mx.bfloat16)
    engine = eSurge(
        model,
        tokenizer=_build_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        async_scheduling=False,
        overlap_execution=False,
    )
    try:
        assert engine.runtime_dtype == mx.bfloat16
        assert engine.kv_cache_dtype == mx.bfloat16
        assert model.recorded_dtype == mx.bfloat16
        assert engine._runner is not None
        assert engine._runner.kv_caches[0].key_cache.dtype == mx.bfloat16
    finally:
        engine.close()


def test_esurge_runtime_dtype_prefers_explicit_model_runtime_attr():
    model = DtypeRecordingPagedModel(parameter_dtype=mx.bfloat16)
    model.runtime_dtype = mx.float16
    engine = eSurge(
        model,
        tokenizer=_build_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        async_scheduling=False,
        overlap_execution=False,
    )
    try:
        assert engine.runtime_dtype == mx.float16
        assert engine.kv_cache_dtype == mx.float16
        assert model.recorded_dtype == mx.float16
    finally:
        engine.close()


def test_esurge_runtime_dtype_prefers_text_config_dtype_over_top_level_dtype():
    model = DtypeRecordingPagedModel(parameter_dtype=mx.float16)
    model.config.dtype = "float16"
    model.config.text_config = type("TextConfig", (), {"dtype": "bfloat16"})()
    engine = eSurge(
        model,
        tokenizer=_build_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=1,
        compile_runner=False,
        async_scheduling=False,
        overlap_execution=False,
    )
    try:
        assert engine.runtime_dtype == mx.bfloat16
        assert engine.kv_cache_dtype == mx.bfloat16
        assert model.recorded_dtype == mx.bfloat16
    finally:
        engine.close()


def test_esurge_stop_strings():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello", SamplingParams(max_tokens=2, stop=["foo"]))
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].text == ""


def test_esurge_stream():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    chunks = list(engine.stream("hello", SamplingParams(max_tokens=2)))
    assert any(c.finished for c in chunks)


def test_esurge_chat_accepts_explicit_request_id():
    engine = eSurge(
        DummyPagedModel(),
        tokenizer=_build_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=4,
        compile_runner=False,
    )
    try:
        output = engine.chat(
            [{"role": "user", "content": "hello"}],
            SamplingParams(max_tokens=1),
            request_id="chat-req-1",
        )
        assert output.request_id == "chat-req-1"
    finally:
        engine.close()


def test_esurge_chat_without_tokenizer_template_uses_quiet_plain_prompt(monkeypatch: pytest.MonkeyPatch):
    engine = eSurge(
        DummyPagedModel(),
        tokenizer=_build_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=4,
        compile_runner=False,
    )
    monkeypatch.setattr(
        esurge_engine.logger,
        "warning",
        lambda *args, **kwargs: pytest.fail(f"unexpected warning: {args[0] if args else ''}"),
    )
    try:
        prompt = engine._format_chat_prompt([{"role": "user", "content": "hello"}])
        assert prompt == "user: hello\nassistant:"
    finally:
        engine.close()


def test_esurge_chat_uses_tokenizer_template_when_available():
    tokenizer = _build_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}={{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant={% endif %}"
    )
    engine = eSurge(
        DummyPagedModel(),
        tokenizer=tokenizer,
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=4,
        compile_runner=False,
    )
    try:
        prompt = engine._format_chat_prompt([{"role": "user", "content": "hello"}])
        assert prompt == "user=hello\nassistant="
    finally:
        engine.close()


def test_esurge_paged_generate_basic():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello world", SamplingParams(max_tokens=2))
    assert len(outputs) == 1
    assert outputs[0].outputs[0].token_ids == [5, 1]
    assert outputs[0].outputs[0].text.strip() == "foo"


def test_esurge_paged_stop_strings():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello", SamplingParams(max_tokens=2, stop=["foo"]))
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].text == ""


def test_esurge_paged_stream_is_incremental():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = list(engine.stream("hello", SamplingParams(max_tokens=2)))
    assert len(outputs) >= 1
    assert outputs[0].delta_text.strip() == "foo"
    assert outputs[-1].finished is True


def test_esurge_stream_ignores_transient_replacement_suffixes():
    engine = eSurge(
        PlannedPagedModel(),
        tokenizer=_build_partial_decode_tokenizer(),
        max_model_len=16,
        reserve_tokens=2,
        max_num_seqs=4,
    )

    outputs = list(engine.stream("hello", SamplingParams(max_tokens=4)))

    assert [chunk.delta_text for chunk in outputs] == ["alpha", "🦕", "beta", "🤖"]
    assert [chunk.accumulated_text for chunk in outputs] == [
        "alpha",
        "alpha🦕",
        "alpha🦕beta",
        "alpha🦕beta🤖",
    ]
    reconstructed = ""
    for chunk in outputs:
        reconstructed += chunk.delta_text
        assert chunk.accumulated_text == reconstructed
    assert outputs[-1].finished is True
