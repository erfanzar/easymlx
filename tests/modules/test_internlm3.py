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

"""Tests for InternLM3 model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.internlm3 import InternLM3Config, InternLM3ForCausalLM, InternLM3Model

from .test_utils import CausalLMTester


class TestInternLM3:
    """Test suite for InternLM3 model."""

    @pytest.fixture
    def internlm3_config(self, small_model_config):
        return InternLM3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            bias=False,
            qkv_bias=False,
            rope_theta=10000,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "internlm3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "internlm3")

        assert base_registration.module is InternLM3Model
        assert base_registration.config is InternLM3Config
        assert lm_registration.module is InternLM3ForCausalLM
        assert lm_registration.config is InternLM3Config

    def test_causal_lm(self, internlm3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="internlm3",
            model_cls=InternLM3ForCausalLM,
            config=internlm3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"InternLM3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, internlm3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="internlm3",
            model_cls=InternLM3ForCausalLM,
            config=internlm3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"InternLM3 generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, internlm3_config):
        model = InternLM3ForCausalLM(internlm3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]
        upstream_weights["model.layers.0.attention.rope.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert "model.layers.0.attention.rope.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
