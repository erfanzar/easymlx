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

"""Tests for Ministral3 model."""

import pytest

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.ministral3 import Ministral3Config, Ministral3ForCausalLM, Ministral3Model

from .test_utils import CausalLMTester


class TestMinistral3:
    @pytest.fixture
    def ministral3_config(self, small_model_config):
        return Ministral3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "rope_theta": 10000.0,
                "original_max_position_embeddings": 128,
                "llama_4_scaling_beta": 0.1,
            },
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "ministral3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "ministral3")

        assert base_registration.module is Ministral3Model
        assert base_registration.config is Ministral3Config
        assert lm_registration.module is Ministral3ForCausalLM
        assert lm_registration.config is Ministral3Config

    def test_config_extracts_rope_parameters(self, ministral3_config):
        assert ministral3_config.model_type == "ministral3"
        assert ministral3_config.llama_4_scaling_beta == 0.1
        assert ministral3_config.original_max_position_embeddings == 128
        assert ministral3_config.layer_types == ["sliding_attention", "full_attention"]

    def test_causal_lm(self, ministral3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="ministral3",
            model_cls=Ministral3ForCausalLM,
            config=ministral3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Ministral3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, ministral3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="ministral3",
            model_cls=Ministral3ForCausalLM,
            config=ministral3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Ministral3 generation failed: {result.error_message}"
