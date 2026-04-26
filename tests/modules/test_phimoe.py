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

"""Tests for PhiMoE model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.phimoe import PhimoeConfig, PhimoeForCausalLM, PhimoeModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestPhimoe:
    """Test suite for PhiMoE model."""

    @pytest.fixture
    def phimoe_config(self, small_model_config):
        """Create PhiMoE-specific config."""
        return PhimoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            num_local_experts=4,
            num_experts_per_tok=2,
        )

    def test_registry(self):
        """Test that PhiMoE is properly registered."""
        base_reg = registry.get_module_registration(TaskType.BASE_MODULE, "phimoe")
        lm_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "phimoe")
        assert base_reg.module is PhimoeModel
        assert lm_reg.module is PhimoeForCausalLM

    def test_causal_lm(self, phimoe_config, small_model_config):
        """Test PhimoeForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phimoe",
            model_cls=PhimoeForCausalLM,
            config=phimoe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"PhiMoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, phimoe_config, small_model_config):
        """Test PhiMoE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phimoe",
            model_cls=PhimoeForCausalLM,
            config=phimoe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"PhiMoE generation failed: {result.error_message}"

    def test_sanitize(self, phimoe_config):
        """Test weight sanitization."""
        model = PhimoeForCausalLM(phimoe_config)
        local_weights = dict(tree_flatten(model.parameters()))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.ones((8,))
        sanitized = model.sanitize(upstream_weights)
        assert "rotary_emb.inv_freq" not in sanitized
