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

"""Tests for Gemma3 model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gemma3 import Gemma3Config, Gemma3ForCausalLM, Gemma3Model

from .test_utils import CausalLMTester


class TestGemma3:
    """Test suite for Gemma3 model."""

    @pytest.fixture
    def gemma3_config(self, small_model_config):
        return Gemma3Config(
            text_config={
                "vocab_size": small_model_config["vocab_size"],
                "hidden_size": small_model_config["hidden_size"],
                "intermediate_size": small_model_config["intermediate_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "head_dim": small_model_config["head_dim"],
                "sliding_window_pattern": 2,
                "query_pre_attn_scalar": 16.0,
                "sliding_window": 8,
                "max_position_embeddings": small_model_config["max_position_embeddings"],
            },
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gemma3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gemma3")

        assert base_registration.module is Gemma3Model
        assert base_registration.config is Gemma3Config
        assert lm_registration.module is Gemma3ForCausalLM
        assert lm_registration.config is Gemma3Config

    def test_causal_lm(self, gemma3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma3",
            model_cls=Gemma3ForCausalLM,
            config=gemma3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, gemma3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma3",
            model_cls=Gemma3ForCausalLM,
            config=gemma3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Gemma3 generation failed: {result.error_message}"

    def test_sanitize_drops_vision_keys(self, gemma3_config):
        model = Gemma3ForCausalLM(gemma3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = {}
        for key, value in local_weights.items():
            if key == "lm_head.weight":
                upstream_weights["language_model.lm_head.weight"] = value
            else:
                upstream_weights[key.replace("model.language_model.", "language_model.", 1)] = value
        upstream_weights.update(
            {
                "vision_tower.blocks.0.weight": mx.ones((2, 2)),
                "multi_modal_projector.weight": mx.ones((2, 2)),
            }
        )

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.blocks.0.weight" not in sanitized
        assert "multi_modal_projector.weight" not in sanitized
        assert set(sanitized) == set(local_weights)
