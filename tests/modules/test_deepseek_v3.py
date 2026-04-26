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

"""Tests for DeepSeek V3 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM, DeepseekV3Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestDeepseekV3:
    """Test suite for DeepSeek V3 model."""

    @pytest.fixture
    def deepseek_v3_config(self, small_model_config):
        return DeepseekV3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=64,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            topk_method="noaux_tc",
            max_position_embeddings=small_model_config["max_position_embeddings"],
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "deepseek_v3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "deepseek_v3")

        assert base_registration.module is DeepseekV3Model
        assert base_registration.config is DeepseekV3Config
        assert lm_registration.module is DeepseekV3ForCausalLM
        assert lm_registration.config is DeepseekV3Config

    def test_causal_lm(self, deepseek_v3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="deepseek_v3",
            model_cls=DeepseekV3ForCausalLM,
            config=deepseek_v3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DeepSeek V3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, deepseek_v3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="deepseek_v3",
            model_cls=DeepseekV3ForCausalLM,
            config=deepseek_v3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"DeepSeek V3 generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, deepseek_v3_config):
        model = DeepseekV3ForCausalLM(deepseek_v3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
