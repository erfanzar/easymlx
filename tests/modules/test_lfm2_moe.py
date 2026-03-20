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

"""Tests for LFM2-MoE model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.lfm2_moe import Lfm2MoeConfig, Lfm2MoeForCausalLM, Lfm2MoeModel

from .test_utils import CausalLMTester


class TestLfm2Moe:
    """Test suite for LFM2-MoE model."""

    @pytest.fixture
    def lfm2_moe_config(self, small_model_config):
        return Lfm2MoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=64,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_experts=small_model_config["num_experts"],
            num_experts_per_tok=small_model_config["num_experts_per_tok"],
            norm_topk_prob=True,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            use_expert_bias=False,
            num_dense_layers=0,
            conv_L_cache=4,
            conv_bias=True,
            full_attn_idxs=[0],
            layer_types=["full_attention", "short_conv"],
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "lfm2_moe")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "lfm2_moe")

        assert base_registration.module is Lfm2MoeModel
        assert base_registration.config is Lfm2MoeConfig
        assert lm_registration.module is Lfm2MoeForCausalLM
        assert lm_registration.config is Lfm2MoeConfig

    def test_causal_lm(self, lfm2_moe_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="lfm2_moe",
            model_cls=Lfm2MoeForCausalLM,
            config=lfm2_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"LFM2-MoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, lfm2_moe_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="lfm2_moe",
            model_cls=Lfm2MoeForCausalLM,
            config=lfm2_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"LFM2-MoE generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, lfm2_moe_config):
        model = Lfm2MoeForCausalLM(lfm2_moe_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        sanitized = model.sanitize(upstream_weights)
        assert set(sanitized) == set(local_weights)
