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

"""Tests for Phi3Small model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.phi3small import Phi3SmallConfig, Phi3SmallForCausalLM, Phi3SmallModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestPhi3Small:
    """Test suite for Phi3Small model."""

    @pytest.fixture
    def phi3small_config(self, small_model_config):
        return Phi3SmallConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            ff_intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            dense_attention_every_n_layers=2,
            gegelu_limit=20.0,
            mup_attn_multiplier=1.0,
            mup_embedding_multiplier=10.0,
            mup_use_scaling=True,
            mup_width_multiplier=8.0,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "phi3small")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "phi3small")

        assert base_registration.module is Phi3SmallModel
        assert base_registration.config is Phi3SmallConfig
        assert lm_registration.module is Phi3SmallForCausalLM
        assert lm_registration.config is Phi3SmallConfig

    def test_causal_lm(self, phi3small_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="phi3small",
            model_cls=Phi3SmallForCausalLM,
            config=phi3small_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Phi3Small CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, phi3small_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phi3small",
            model_cls=Phi3SmallForCausalLM,
            config=phi3small_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Phi3Small generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, phi3small_config):
        model = Phi3SmallForCausalLM(phi3small_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
