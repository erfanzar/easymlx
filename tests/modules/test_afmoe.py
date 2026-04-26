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

"""Tests for AFMoE model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.afmoe import AfmoeConfig, AfmoeForCausalLM, AfmoeModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestAfmoe:
    """Test suite for AFMoE model."""

    @pytest.fixture
    def afmoe_config(self, small_model_config):
        """Create a tiny AFMoE config."""
        return AfmoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            num_dense_layers=1,
            layer_types=["full_attention", "full_attention"],
            mup_enabled=False,
            n_group=1,
            topk_group=1,
        )

    def test_registry(self):
        """AFMoE should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "afmoe")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "afmoe")

        assert base_registration.module is AfmoeModel
        assert base_registration.config is AfmoeConfig
        assert lm_registration.module is AfmoeForCausalLM
        assert lm_registration.config is AfmoeConfig

    def test_causal_lm(self, afmoe_config, small_model_config):
        """Test AFMoE causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="afmoe",
            model_cls=AfmoeForCausalLM,
            config=afmoe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"AFMoE CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, afmoe_config, small_model_config):
        """Test AFMoE generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="afmoe",
            model_cls=AfmoeForCausalLM,
            config=afmoe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"AFMoE generation failed: {result.error_message}"

    def test_sanitize_stacks_experts(self, afmoe_config):
        """Ensure AFMoE sanitize stacks per-expert weights correctly."""
        model = AfmoeForCausalLM(afmoe_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        prefix = "model.layers.1"
        for dst in ["gate_proj", "down_proj", "up_proj"]:
            stacked_key = f"{prefix}.mlp.experts.{dst}.weight"
            if stacked_key in upstream_weights:
                tensor = upstream_weights.pop(stacked_key)
                for expert_idx in range(tensor.shape[0]):
                    upstream_weights[f"{prefix}.mlp.experts.{expert_idx}.{dst}.weight"] = tensor[expert_idx]

        sanitized = model.sanitize(upstream_weights)

        assert set(sanitized) == set(local_weights)
