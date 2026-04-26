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

"""Tests for GPT-BigCode model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gpt_bigcode import GPTBigCodeConfig, GPTBigCodeForCausalLM, GPTBigCodeModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestGPTBigCode:
    """Test suite for GPT-BigCode model."""

    @pytest.fixture
    def gpt_bigcode_config(self, small_model_config):
        """Create a tiny GPT-BigCode config."""
        return GPTBigCodeConfig(
            vocab_size=small_model_config["vocab_size"],
            n_embd=small_model_config["hidden_size"],
            n_layer=small_model_config["num_hidden_layers"],
            n_head=small_model_config["num_attention_heads"],
            n_positions=small_model_config["max_position_embeddings"],
            n_inner=small_model_config["intermediate_size"],
            multi_query=True,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        """GPT-BigCode should register under the exact HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gpt_bigcode")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gpt_bigcode")

        assert base_registration.module is GPTBigCodeModel
        assert base_registration.config is GPTBigCodeConfig
        assert lm_registration.module is GPTBigCodeForCausalLM
        assert lm_registration.config is GPTBigCodeConfig

    def test_causal_lm(self, gpt_bigcode_config, small_model_config):
        """Test GPT-BigCode causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt_bigcode",
            model_cls=GPTBigCodeForCausalLM,
            config=gpt_bigcode_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT-BigCode CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, gpt_bigcode_config, small_model_config):
        """Test GPT-BigCode generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gpt_bigcode",
            model_cls=GPTBigCodeForCausalLM,
            config=gpt_bigcode_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GPT-BigCode generation failed: {result.error_message}"

    def test_multi_query_config(self):
        """Multi-query attention should default to 1 KV head."""
        config = GPTBigCodeConfig(multi_query=True, n_head=8)
        assert config.num_key_value_heads == 1

    def test_multi_head_config(self):
        """Non-multi-query should default to n_head KV heads."""
        config = GPTBigCodeConfig(multi_query=False, n_head=8)
        assert config.num_key_value_heads == 8

    def test_sanitize_removes_attn_bias(self, gpt_bigcode_config):
        """Sanitize should remove attention bias/masked_bias keys."""
        model = GPTBigCodeModel(gpt_bigcode_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["layers.0.self_attn.attn.bias"] = local_weights["wte.weight"]
        upstream_weights["layers.0.self_attn.attn.masked_bias"] = local_weights["wte.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "layers.0.self_attn.attn.bias" not in sanitized
        assert "layers.0.self_attn.attn.masked_bias" not in sanitized
