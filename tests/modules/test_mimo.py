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

"""Tests for MiMo model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.mimo import MiMoConfig, MiMoForCausalLM, MiMoModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestMiMo:
    """Test suite for MiMo model."""

    @pytest.fixture
    def mimo_config(self, small_model_config):
        return MiMoConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "mimo")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "mimo")

        assert base_registration.module is MiMoModel
        assert base_registration.config is MiMoConfig
        assert lm_registration.module is MiMoForCausalLM
        assert lm_registration.config is MiMoConfig

    def test_causal_lm(self, mimo_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="mimo",
            model_cls=MiMoForCausalLM,
            config=mimo_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MiMo CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, mimo_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mimo",
            model_cls=MiMoForCausalLM,
            config=mimo_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"MiMo generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, mimo_config):
        model = MiMoForCausalLM(mimo_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]
        upstream_weights["model.mtp_layers.0.weight"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
        assert "model.mtp_layers.0.weight" not in sanitized
        assert set(sanitized) == set(local_weights)
