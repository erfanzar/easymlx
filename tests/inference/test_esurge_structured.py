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

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from easymlx.inference.esurge import SamplingParams, eSurge
from easymlx.modules.llama import LlamaConfig, LlamaForCausalLM


class _DummyPagedCache:
    def __init__(self, num_seqs: int):
        self.num_seqs = num_seqs
        self.kv_lens = np.zeros((num_seqs,), dtype=np.int32)

    def reset(self, seq_idx: int) -> None:
        self.kv_lens[int(seq_idx)] = 0


class DummyModel:
    """Paged-compatible dummy that emits a fixed token sequence then EOS."""

    def __init__(self, append_tokens: list[int], *, vocab_size: int = 16, eos_token_id: int = 1):
        self.append_tokens = list(append_tokens)
        self.vocab_size = vocab_size
        self._eos = eos_token_id
        self._step: dict[int, int] = {}

    def init_paged_cache(self, *, num_seqs: int, **_kwargs):
        return [_DummyPagedCache(num_seqs)]

    def __call__(self, input_ids, *, cache_views=None, cache_metadata=None, query_lens=None, slot_ids=None, return_dict=True, **_kwargs):
        arr = np.array(input_ids, dtype=np.int32)
        num_seqs = len(query_lens) if query_lens else arr.shape[0]
        logits = np.full((num_seqs, self.vocab_size), -1e9, dtype=np.float32)
        offset = 0
        for row in range(num_seqs):
            qlen = int(query_lens[row]) if query_lens else (arr.shape[1] if arr.ndim > 1 else arr.shape[0])
            sid = int(slot_ids[row]) if slot_ids else row
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
        if return_dict:
            return type("Out", (), {"logits": logits})()
        return logits


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


def _save_test_tokenizer(path: Path) -> None:
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        **{f"tok{i}": i for i in range(3, 32)},
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    hf_tok.save_pretrained(str(path))


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


def test_esurge_from_pretrained_auto_converts_local_repo(tmp_path: Path):
    model = _tiny_llama()
    source_dir = tmp_path / "raw-hf"
    cache_dir = tmp_path / "converted-cache"
    source_dir.mkdir()

    model.config.save_pretrained(str(source_dir))
    mx.save_safetensors(str(source_dir / "model.safetensors"), _flatten_params(model))
    _save_test_tokenizer(source_dir)

    engine = eSurge.from_pretrained(
        str(source_dir),
        converted_cache_dir=str(cache_dir),
        max_model_len=32,
        max_num_seqs=2,
        reserve_tokens=4,
    )
    output = engine.generate("tok3 tok4", SamplingParams(max_tokens=1))[0]

    assert engine.tokenizer is not None
    assert engine.model.name_or_path.startswith(str(cache_dir))
    assert output.finished is True
