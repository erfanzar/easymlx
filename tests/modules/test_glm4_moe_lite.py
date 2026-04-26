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

"""Tests for GLM-4 MoE Lite model."""

import pytest
from easymlx.modules.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from .test_utils import CausalLMTester


class TestGlm4MoeLite:
    """Test suite for GLM-4 MoE Lite model."""

    @pytest.fixture
    def glm4_moe_lite_config(self, small_model_config):
        """Create GLM-4 MoE Lite-specific config."""
        return Glm4MoeLiteConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            kv_lora_rank=8,
            q_lora_rank=None,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=8,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            mlp_layer_types=["dense", "dense"],
        )

    def test_causal_lm(self, glm4_moe_lite_config, small_model_config):
        """Test Glm4MoeLiteForCausalLM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4_moe_lite",
            model_cls=Glm4MoeLiteForCausalLM,
            config=glm4_moe_lite_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-4 MoE Lite CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, glm4_moe_lite_config, small_model_config):
        """Test GLM-4 MoE Lite text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4_moe_lite",
            model_cls=Glm4MoeLiteForCausalLM,
            config=glm4_moe_lite_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GLM-4 MoE Lite generation failed: {result.error_message}"
