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

"""Tests for MiMo-V2-Flash model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.mimo_v2_flash import MiMoV2FlashConfig, MiMoV2FlashForCausalLM, MiMoV2FlashModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestMiMoV2Flash:
    """Test suite for MiMo-V2-Flash model."""

    @pytest.fixture
    def mimo_v2_flash_config(self, small_model_config):
        return MiMoV2FlashConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=64,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            num_experts_per_tok=2,
            hybrid_layer_pattern=[0, 0],
            moe_layer_freq=[0, 1],
            sliding_window_size=32,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            v_head_dim=small_model_config["head_dim"],
            swa_head_dim=small_model_config["head_dim"],
            swa_v_head_dim=small_model_config["head_dim"],
            swa_num_attention_heads=small_model_config["num_attention_heads"],
            swa_num_key_value_heads=small_model_config["num_key_value_heads"],
            partial_rotary_factor=1,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "mimo_v2_flash")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "mimo_v2_flash")

        assert base_registration.module is MiMoV2FlashModel
        assert base_registration.config is MiMoV2FlashConfig
        assert lm_registration.module is MiMoV2FlashForCausalLM
        assert lm_registration.config is MiMoV2FlashConfig

    def test_causal_lm(self, mimo_v2_flash_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="mimo_v2_flash",
            model_cls=MiMoV2FlashForCausalLM,
            config=mimo_v2_flash_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MiMo-V2-Flash CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, mimo_v2_flash_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mimo_v2_flash",
            model_cls=MiMoV2FlashForCausalLM,
            config=mimo_v2_flash_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"MiMo-V2-Flash generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, mimo_v2_flash_config):
        model = MiMoV2FlashForCausalLM(mimo_v2_flash_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        sanitized = model.sanitize(upstream_weights)
        assert set(sanitized) == set(local_weights)
