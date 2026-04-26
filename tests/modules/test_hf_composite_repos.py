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

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from easymlx.inference.esurge import eSurge
from easymlx.infra.factory import TaskType
from easymlx.modules.auto import AutoEasyMLXConfig, AutoEasyMLXModelForCausalLM
from easymlx.modules.qwen3 import Qwen3Config, Qwen3ForCausalLM
from easymlx.utils.hf_composite import resolve_hf_composite_repo
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "hello": 4,
        "world": 5,
        "flux": 6,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _build_flux_style_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "flux-klein-style"
    repo.mkdir()

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(repo / "text_encoder")

    tokenizer = _build_tokenizer()
    tokenizer.save_pretrained(str(repo / "tokenizer"))

    (repo / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return repo


def test_resolve_hf_composite_repo_detects_text_encoder_and_tokenizer(tmp_path: Path) -> None:
    repo = _build_flux_style_repo(tmp_path)

    resolved = resolve_hf_composite_repo(repo, task_type=TaskType.CAUSAL_LM)

    assert resolved.model_subfolder == "text_encoder"
    assert resolved.tokenizer_subfolder == "tokenizer"
    assert resolved.composite_model_type is None


def test_direct_model_load_from_flux_style_repo(tmp_path: Path) -> None:
    repo = _build_flux_style_repo(tmp_path)

    model = Qwen3ForCausalLM.from_pretrained(repo, auto_convert_hf=False)
    logits = model(mx.array([[1, 4, 5]], dtype=mx.int32), return_dict=False)

    assert logits.shape == (1, 3, model.config.vocab_size)
    assert Path(model.name_or_path).name == "text_encoder"
    assert Path(model.tokenizer_name_or_path).name == "tokenizer"


def test_auto_loaders_and_esurge_tokenizer_support_flux_style_repo(tmp_path: Path) -> None:
    repo = _build_flux_style_repo(tmp_path)

    config = AutoEasyMLXConfig.from_pretrained(str(repo))
    model = AutoEasyMLXModelForCausalLM.from_pretrained(str(repo), auto_convert_hf=False)

    assert isinstance(config, Qwen3Config)
    assert isinstance(model, Qwen3ForCausalLM)
    assert Path(model.name_or_path).name == "text_encoder"
    assert Path(model.tokenizer_name_or_path).name == "tokenizer"

    engine = eSurge.__new__(eSurge)
    engine.model = model
    tokenizer = engine._resolve_tokenizer(None)
    assert tokenizer.pad_token_id == 0
