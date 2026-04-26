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

"""Tests for Telechat3 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.telechat3 import Telechat3Config, Telechat3ForCausalLM, Telechat3Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestTelechat3:
    """Test suite for Telechat3 model."""

    @pytest.fixture
    def telechat3_config(self, small_model_config):
        return Telechat3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            head_dim=small_model_config["head_dim"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "telechat3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "telechat3")

        assert base_registration.module is Telechat3Model
        assert base_registration.config is Telechat3Config
        assert lm_registration.module is Telechat3ForCausalLM
        assert lm_registration.config is Telechat3Config

    def test_causal_lm(self, telechat3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="telechat3",
            model_cls=Telechat3ForCausalLM,
            config=telechat3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Telechat3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, telechat3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="telechat3",
            model_cls=Telechat3ForCausalLM,
            config=telechat3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Telechat3 generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, telechat3_config):
        model = Telechat3ForCausalLM(telechat3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
