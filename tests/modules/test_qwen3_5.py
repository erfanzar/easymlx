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

import mlx.core as mx
from easymlx.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from mlx.utils import tree_flatten


def _small_qwen35_text_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"],
    )


def test_qwen35_text_sanitize_accepts_language_model_prefixes():
    model = Qwen3_5ForCausalLM(_small_qwen35_text_config())
    local_weights = dict(tree_flatten(model.parameters(), destination={}))
    upstream_weights = {}
    for key, value in local_weights.items():
        if key.startswith("model."):
            upstream_weights[f"language_model.{key}"] = value
        elif key.startswith("lm_head."):
            upstream_weights[f"language_model.{key}"] = value

    sanitized = model.sanitize(upstream_weights)

    assert set(sanitized) == set(local_weights)


def test_qwen35_text_sanitize_accepts_vl_language_model_prefixes_and_drops_vision():
    model = Qwen3_5ForCausalLM(_small_qwen35_text_config())
    local_weights = dict(tree_flatten(model.parameters(), destination={}))
    upstream_weights = {
        "model.vision_tower.blocks.0.norm1.weight": mx.ones((16,)),
        "model.visual.blocks.0.norm1.weight": mx.ones((16,)),
    }
    for key, value in local_weights.items():
        if key.startswith("model."):
            upstream_weights[f"model.language_model.{key.removeprefix('model.')}"] = value
        elif key.startswith("lm_head."):
            upstream_weights[f"model.language_model.{key}"] = value

    sanitized = model.sanitize(upstream_weights)

    assert set(sanitized) == set(local_weights)
    assert all("vision_tower" not in key and ".visual." not in key for key in sanitized)


def test_qwen35_text_sanitize_converts_mlx_lm_rmsnorm_scales():
    model = Qwen3_5ForCausalLM(_small_qwen35_text_config())
    weights = {
        "language_model.model.layers.0.input_layernorm.weight": mx.ones((16,)),
        "language_model.model.layers.0.linear_attn.norm.weight": mx.ones((8,)),
        "language_model.model.layers.1.self_attn.q_norm.weight": mx.ones((8,)) * 1.25,
        "language_model.model.norm.weight": mx.ones((16,)) * 2.0,
    }

    sanitized = model.sanitize(weights)

    assert float(mx.max(sanitized["model.layers.0.input_layernorm.weight"]).item()) == 0.0
    assert float(mx.max(sanitized["model.layers.1.self_attn.q_norm.weight"]).item()) == 0.25
    assert float(mx.max(sanitized["model.norm.weight"]).item()) == 1.0
    assert float(mx.max(sanitized["model.layers.0.linear_attn.norm.weight"]).item()) == 1.0
