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

"""Tests for Bailing MoE model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.bailing_moe import BailingMoeConfig, BailingMoeForCausalLM, BailingMoeModel

from .test_utils import CausalLMTester


class TestBailingMoe:
    """Test suite for Bailing MoE model."""

    @pytest.fixture
    def bailing_moe_config(self, small_model_config):
        return BailingMoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config["num_experts"],
            num_shared_experts=1,
            num_experts_per_tok=small_model_config["num_experts_per_tok"],
            first_k_dense_replace=0,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "bailing_moe")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "bailing_moe")

        assert base_registration.module is BailingMoeModel
        assert base_registration.config is BailingMoeConfig
        assert lm_registration.module is BailingMoeForCausalLM
        assert lm_registration.config is BailingMoeConfig

    def test_causal_lm(self, bailing_moe_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="bailing_moe",
            model_cls=BailingMoeForCausalLM,
            config=bailing_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Bailing MoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, bailing_moe_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="bailing_moe",
            model_cls=BailingMoeForCausalLM,
            config=bailing_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Bailing MoE generation failed: {result.error_message}"

    def test_sanitize_stacks_experts(self, bailing_moe_config):
        model = BailingMoeForCausalLM(bailing_moe_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.word_embeddings.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
