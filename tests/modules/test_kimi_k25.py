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

"""Tests for Kimi K2.5 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.kimi_k25 import KimiK25Config, KimiK25ForCausalLM, KimiK25Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestKimiK25:
    """Test suite for Kimi K2.5 model."""

    @pytest.fixture
    def kimi_k25_config(self, small_model_config):
        return KimiK25Config(
            text_config={
                "vocab_size": small_model_config["vocab_size"],
                "hidden_size": small_model_config["hidden_size"],
                "intermediate_size": small_model_config["intermediate_size"],
                "moe_intermediate_size": small_model_config["intermediate_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "max_position_embeddings": small_model_config["max_position_embeddings"],
                "kv_lora_rank": 32,
                "q_lora_rank": 32,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 8,
                "n_routed_experts": None,
                "n_shared_experts": None,
            },
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "kimi_k25")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "kimi_k25")

        assert base_registration.module is KimiK25Model
        assert base_registration.config is KimiK25Config
        assert lm_registration.module is KimiK25ForCausalLM
        assert lm_registration.config is KimiK25Config

    def test_causal_lm(self, kimi_k25_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="kimi_k25",
            model_cls=KimiK25ForCausalLM,
            config=kimi_k25_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Kimi K2.5 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, kimi_k25_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="kimi_k25",
            model_cls=KimiK25ForCausalLM,
            config=kimi_k25_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Kimi K2.5 generation failed: {result.error_message}"

    def test_sanitize_removes_vision_keys(self, kimi_k25_config):
        model = KimiK25ForCausalLM(kimi_k25_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["vision_tower.dummy"] = local_weights["model.language_model.embed_tokens.weight"]
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.language_model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.dummy" not in sanitized
        assert "rotary_emb.inv_freq" not in sanitized
