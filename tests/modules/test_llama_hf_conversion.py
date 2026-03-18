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

"""Tests for raw Hugging Face -> easymlx Llama conversion."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from easymlx.inference.esurge import SamplingParams, eSurge
from easymlx.modules.llama import LlamaConfig, LlamaForCausalLM


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


def test_llama_convert_hf_checkpoint_round_trip(tmp_path: Path):
    model = _tiny_llama()
    source_dir = tmp_path / "raw-hf"
    out_dir = tmp_path / "converted"
    source_dir.mkdir()

    model.config.save_pretrained(str(source_dir))
    mx.save_safetensors(str(source_dir / "model.safetensors"), _flatten_params(model))

    LlamaForCausalLM.convert_hf_checkpoint(source_dir, save_directory=out_dir)
    converted = LlamaForCausalLM.from_pretrained(out_dir)

    original_params = _flatten_params(model)
    converted_params = _flatten_params(converted)
    assert original_params.keys() == converted_params.keys()
    for key in original_params:
        assert converted_params[key].dtype == mx.float16
        assert mx.array_equal(original_params[key].astype(mx.float16), converted_params[key]).item() is True


def test_llama_from_pretrained_auto_convert_local_repo(tmp_path: Path):
    model = _tiny_llama()
    source_dir = tmp_path / "raw-hf"
    cache_dir = tmp_path / "converted-cache"
    source_dir.mkdir()

    model.config.save_pretrained(str(source_dir))
    mx.save_safetensors(str(source_dir / "model.safetensors"), _flatten_params(model))

    loaded = LlamaForCausalLM.from_pretrained(
        source_dir,
        auto_convert_hf=True,
        converted_cache_dir=cache_dir,
    )
    cache_entries = list(cache_dir.iterdir())
    assert len(cache_entries) == 1
    assert (cache_entries[0] / "model.safetensors").exists()
    assert (cache_entries[0] / "easymlx_source.json").exists()
    params = _flatten_params(loaded)
    assert all(value.dtype == mx.float16 for value in params.values())
    assert isinstance(loaded, LlamaForCausalLM)


def test_llama_from_pretrained_auto_convert_feeds_esurge_without_tokenizer(tmp_path: Path):
    model = _tiny_llama()
    source_dir = tmp_path / "raw-hf"
    cache_dir = tmp_path / "converted-cache"
    source_dir.mkdir()

    model.config.save_pretrained(str(source_dir))
    mx.save_safetensors(str(source_dir / "model.safetensors"), _flatten_params(model))
    _save_test_tokenizer(source_dir)

    loaded = LlamaForCausalLM.from_pretrained(
        source_dir,
        auto_convert_hf=True,
        converted_cache_dir=cache_dir,
    )

    cache_entries = list(cache_dir.iterdir())
    assert len(cache_entries) == 1
    assert loaded.name_or_path == str(cache_entries[0])

    engine = eSurge(loaded, max_model_len=32, max_num_seqs=2, reserve_tokens=4)
    outputs = engine.generate("tok3 tok4", SamplingParams(max_tokens=1, do_sample=False))

    assert len(outputs) == 1
    assert outputs[0].finished is True
