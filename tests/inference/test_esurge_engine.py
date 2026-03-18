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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from easymlx.caching import PagedKVCache
from easymlx.inference.esurge import SamplingParams, eSurge


class DummyPagedModel:
    """Minimal model using real PagedKVCache for eSurge tests."""

    vocab_size = 6

    def init_paged_cache(self, *, num_seqs: int, max_seq_len: int = 16, page_size: int = 16, **_kwargs):
        return [
            PagedKVCache.allocate(
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
            num_tokens = arr.shape[0]
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


def test_esurge_generate_basic():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello world", SamplingParams(max_tokens=2))
    assert len(outputs) == 1
    out = outputs[0]
    assert out.outputs[0].token_ids == [5, 1]
    assert out.outputs[0].text.strip() == "foo"


def test_esurge_stop_strings():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    outputs = engine.generate("hello", SamplingParams(max_tokens=2, stop=["foo"]))
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].text == ""


def test_esurge_stream():
    engine = eSurge(DummyPagedModel(), tokenizer=_build_tokenizer(), max_model_len=16, reserve_tokens=2, max_num_seqs=4)
    chunks = list(engine.stream("hello", SamplingParams(max_tokens=2)))
    assert any(c.finished for c in chunks)


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
