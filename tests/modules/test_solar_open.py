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

"""Tests for Solar Open model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.solar_open import SolarOpenConfig, SolarOpenForCausalLM, SolarOpenModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestSolarOpen:
    """Test suite for Solar Open model."""

    @pytest.fixture
    def solar_open_config(self, small_model_config):
        """Create a tiny Solar Open config."""
        return SolarOpenConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            moe_intermediate_size=small_model_config["intermediate_size"],
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_routed_experts=4,
            first_k_dense_replace=1,
        )

    def test_registry(self):
        """Solar Open should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "solar_open")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "solar_open")

        assert base_registration.module is SolarOpenModel
        assert base_registration.config is SolarOpenConfig
        assert lm_registration.module is SolarOpenForCausalLM
        assert lm_registration.config is SolarOpenConfig

    def test_causal_lm(self, solar_open_config, small_model_config):
        """Test Solar Open causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="solar_open",
            model_cls=SolarOpenForCausalLM,
            config=solar_open_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Solar Open CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, solar_open_config, small_model_config):
        """Test Solar Open generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="solar_open",
            model_cls=SolarOpenForCausalLM,
            config=solar_open_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Solar Open generation failed: {result.error_message}"

    def test_sanitize_stacks_experts(self, solar_open_config):
        """Ensure Solar Open sanitize preserves the GLM-4 MoE expert stacking rules."""
        model = SolarOpenForCausalLM(solar_open_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        prefix = "model.layers.1"
        for suffix in ["weight", "scales", "biases"]:
            for dst in ["gate_proj", "down_proj", "up_proj"]:
                stacked_key = f"{prefix}.mlp.switch_mlp.{dst}.{suffix}"
                if stacked_key in upstream_weights:
                    tensor = upstream_weights.pop(stacked_key)
                    for expert_idx in range(tensor.shape[0]):
                        upstream_weights[f"{prefix}.mlp.experts.{expert_idx}.{dst}.{suffix}"] = tensor[expert_idx]

        sanitized = model.sanitize(upstream_weights)

        assert set(sanitized) == set(local_weights)
