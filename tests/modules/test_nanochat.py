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

"""Tests for Nanochat model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.nanochat import NanochatConfig, NanochatForCausalLM, NanochatModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestNanochat:
    """Test suite for Nanochat model."""

    @pytest.fixture
    def nanochat_config(self, small_model_config):
        return NanochatConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            logits_soft_cap=15.0,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "nanochat")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "nanochat")

        assert base_registration.module is NanochatModel
        assert base_registration.config is NanochatConfig
        assert lm_registration.module is NanochatForCausalLM
        assert lm_registration.config is NanochatConfig

    def test_causal_lm(self, nanochat_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="nanochat",
            model_cls=NanochatForCausalLM,
            config=nanochat_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Nanochat CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, nanochat_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="nanochat",
            model_cls=NanochatForCausalLM,
            config=nanochat_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Nanochat generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, nanochat_config):
        model = NanochatForCausalLM(nanochat_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.wte.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
