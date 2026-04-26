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

"""Tests for Exaone model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.exaone import ExaoneConfig, ExaoneForCausalLM, ExaoneModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestExaone:
    """Test suite for Exaone model."""

    @pytest.fixture
    def exaone_config(self, small_model_config):
        return ExaoneConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "exaone")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "exaone")

        assert base_registration.module is ExaoneModel
        assert base_registration.config is ExaoneConfig
        assert lm_registration.module is ExaoneForCausalLM
        assert lm_registration.config is ExaoneConfig

    def test_causal_lm(self, exaone_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="exaone",
            model_cls=ExaoneForCausalLM,
            config=exaone_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Exaone CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, exaone_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="exaone",
            model_cls=ExaoneForCausalLM,
            config=exaone_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Exaone generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, exaone_config):
        model = ExaoneForCausalLM(exaone_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["transformer.wte.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
