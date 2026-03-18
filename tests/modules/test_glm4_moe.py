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

"""Tests for GLM-4 MoE model."""

import pytest

from easymlx.modules.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM

from .test_utils import CausalLMTester


class TestGlm4Moe:
    """Test suite for GLM-4 MoE model."""

    @pytest.fixture
    def glm4_moe_config(self, small_model_config):
        """Create GLM-4 MoE-specific config."""
        return Glm4MoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            moe_intermediate_size=32,
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_routed_experts=4,
            first_k_dense_replace=1,
        )

    def test_causal_lm(self, glm4_moe_config, small_model_config):
        """Test Glm4MoeForCausalLM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4_moe",
            model_cls=Glm4MoeForCausalLM,
            config=glm4_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-4 MoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, glm4_moe_config, small_model_config):
        """Test GLM-4 MoE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4_moe",
            model_cls=Glm4MoeForCausalLM,
            config=glm4_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GLM-4 MoE generation failed: {result.error_message}"
