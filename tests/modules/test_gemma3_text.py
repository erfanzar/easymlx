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

"""Tests for Gemma3Text model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gemma3_text import Gemma3TextConfig, Gemma3TextForCausalLM, Gemma3TextModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestGemma3Text:
    """Test suite for Gemma3Text model."""

    @pytest.fixture
    def gemma3_text_config(self, small_model_config):
        return Gemma3TextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            sliding_window_pattern=2,
            query_pre_attn_scalar=16.0,
            sliding_window=8,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gemma3_text")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gemma3_text")

        assert base_registration.module is Gemma3TextModel
        assert base_registration.config is Gemma3TextConfig
        assert lm_registration.module is Gemma3TextForCausalLM
        assert lm_registration.config is Gemma3TextConfig

    def test_causal_lm(self, gemma3_text_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma3_text",
            model_cls=Gemma3TextForCausalLM,
            config=gemma3_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma3Text CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, gemma3_text_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma3_text",
            model_cls=Gemma3TextForCausalLM,
            config=gemma3_text_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Gemma3Text generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, gemma3_text_config):
        model = Gemma3TextForCausalLM(gemma3_text_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
