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

import mlx.core as mx
import mlx.nn as nn
from easymlx.inference.esurge.dflash import DFlashConfig, DFlashDraftModel, load_dflash_draft_model
from easymlx.inference.esurge.esurge_engine import eSurge
from mlx.utils import tree_flatten


class FakeTarget:
    def __init__(self, vocab_size: int, hidden_size: int):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def get_embedding(self):
        return self.embed_tokens

    def compute_lm_logits(self, hidden_states):
        return self.embed_tokens.as_linear(hidden_states)


def write_dflash_checkpoint(path, *, hidden_size: int = 8, vocab_size: int = 32) -> None:
    config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden_size // 2,
        "intermediate_size": hidden_size * 2,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "block_size": 4,
        "num_target_layers": 1,
        "dflash_config": {
            "target_layer_ids": [0],
            "mask_token_id": vocab_size - 1,
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    draft_config = DFlashConfig.from_json(path / "config.json")
    model = DFlashDraftModel(draft_config)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))


def test_dflash_loader_accepts_dflash_key_layout(tmp_path):
    write_dflash_checkpoint(tmp_path)

    draft = load_dflash_draft_model(str(tmp_path), local_files_only=True)

    assert draft.target_feature_layer_indices == (1,)
    logits = draft.dflash_logits(
        first_token=1,
        hidden_states=(mx.zeros((1, 3, draft.config.hidden_size)),),
        max_tokens=2,
        target_model=FakeTarget(draft.config.vocab_size, draft.config.hidden_size),
    )
    mx.eval(logits)
    assert logits.shape == (2, draft.config.vocab_size)


def test_dflash_loader_accepts_fixed_quantization(tmp_path):
    write_dflash_checkpoint(tmp_path, hidden_size=32)

    draft = load_dflash_draft_model(
        str(tmp_path),
        local_files_only=True,
        quantization="mxfp4",
    )

    logits = draft.dflash_logits(
        first_token=1,
        hidden_states=(mx.zeros((1, 3, draft.config.hidden_size)),),
        max_tokens=2,
        target_model=FakeTarget(draft.config.vocab_size, draft.config.hidden_size),
    )
    mx.eval(logits)
    assert logits.shape == (2, draft.config.vocab_size)


def test_esurge_detects_dflash_draft_ids():
    assert eSurge._should_load_dflash_speculative_model(
        "z-lab/Qwen3.5-9B-DFlash",
        speculative_method="draft",
        model_kwargs={},
    )
    assert eSurge._should_load_dflash_speculative_model(
        "local-draft",
        speculative_method="dflash",
        model_kwargs={},
    )
