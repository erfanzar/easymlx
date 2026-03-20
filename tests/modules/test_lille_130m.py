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

"""Tests for Lille-130M model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.lille_130m import Lille130mConfig, Lille130mForCausalLM, Lille130mModel

from .test_utils import CausalLMTester


class TestLille130m:
    """Test suite for Lille-130M model."""

    @pytest.fixture
    def lille_130m_config(self, small_model_config):
        return Lille130mConfig(
            vocab_size=small_model_config["vocab_size"],
            n_embd=small_model_config["hidden_size"],
            n_head=small_model_config["num_attention_heads"],
            n_kv_heads=small_model_config["num_key_value_heads"],
            n_layer=small_model_config["num_hidden_layers"],
            block_size=small_model_config["max_position_embeddings"],
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "lille-130m")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "lille-130m")

        assert base_registration.module is Lille130mModel
        assert base_registration.config is Lille130mConfig
        assert lm_registration.module is Lille130mForCausalLM
        assert lm_registration.config is Lille130mConfig

    def test_causal_lm(self, lille_130m_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="lille_130m",
            model_cls=Lille130mForCausalLM,
            config=lille_130m_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Lille-130M CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, lille_130m_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="lille_130m",
            model_cls=Lille130mForCausalLM,
            config=lille_130m_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Lille-130M generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, lille_130m_config):
        model = Lille130mForCausalLM(lille_130m_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
