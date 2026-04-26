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

"""Tests for Qwen3 MoE model."""

import pytest
from easymlx.modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from .test_utils import CausalLMTester


class TestQwen3Moe:
    """Test suite for Qwen3 MoE model."""

    @pytest.fixture
    def qwen3_moe_config(self, small_model_config):
        """Create Qwen3 MoE-specific config."""
        return Qwen3MoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config["num_experts"],
            num_experts_per_tok=small_model_config["num_experts_per_tok"],
            moe_intermediate_size=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, qwen3_moe_config, small_model_config):
        """Test Qwen3MoeForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_moe",
            model_cls=Qwen3MoeForCausalLM,
            config=qwen3_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3 MoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, qwen3_moe_config, small_model_config):
        """Test Qwen3 MoE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_moe",
            model_cls=Qwen3MoeForCausalLM,
            config=qwen3_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Qwen3 MoE generation failed: {result.error_message}"
