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

"""Tests for PLaMo model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.plamo import PlamoConfig, PlamoForCausalLM, PlamoModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestPlamo:
    """Test suite for PLaMo model."""

    @pytest.fixture
    def plamo_config(self, small_model_config):
        return PlamoConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            n_shared_head=2,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "plamo")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "plamo")

        assert base_registration.module is PlamoModel
        assert base_registration.config is PlamoConfig
        assert lm_registration.module is PlamoForCausalLM
        assert lm_registration.config is PlamoConfig

    def test_causal_lm(self, plamo_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="plamo",
            model_cls=PlamoForCausalLM,
            config=plamo_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"PLaMo CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, plamo_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="plamo",
            model_cls=PlamoForCausalLM,
            config=plamo_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"PLaMo generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, plamo_config):
        model = PlamoForCausalLM(plamo_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
