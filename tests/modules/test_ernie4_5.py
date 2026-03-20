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

"""Tests for ERNIE 4.5 dense model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.ernie4_5 import Ernie45Config, Ernie45ForCausalLM, Ernie45Model

from .test_utils import CausalLMTester


class TestErnie45:
    """Test suite for ERNIE 4.5 dense model."""

    @pytest.fixture
    def ernie45_config(self, small_model_config):
        return Ernie45Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "ernie4_5")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "ernie4_5")

        assert base_registration.module is Ernie45Model
        assert base_registration.config is Ernie45Config
        assert lm_registration.module is Ernie45ForCausalLM
        assert lm_registration.config is Ernie45Config

    def test_causal_lm(self, ernie45_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="ernie4_5",
            model_cls=Ernie45ForCausalLM,
            config=ernie45_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"ERNIE 4.5 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, ernie45_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="ernie4_5",
            model_cls=Ernie45ForCausalLM,
            config=ernie45_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"ERNIE 4.5 generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, ernie45_config):
        model = Ernie45ForCausalLM(ernie45_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
