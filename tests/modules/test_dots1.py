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

"""Tests for Dots1 model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.dots1 import Dots1Config, Dots1ForCausalLM, Dots1Model

from .test_utils import CausalLMTester


class TestDots1:
    """Test suite for Dots1 model."""

    @pytest.fixture
    def dots1_config(self, small_model_config):
        """Create a tiny Dots1 config."""
        return Dots1Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
        )

    def test_registry(self):
        """Dots1 should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "dots1")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "dots1")

        assert base_registration.module is Dots1Model
        assert base_registration.config is Dots1Config
        assert lm_registration.module is Dots1ForCausalLM
        assert lm_registration.config is Dots1Config

    def test_causal_lm(self, dots1_config, small_model_config):
        """Test Dots1 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="dots1",
            model_cls=Dots1ForCausalLM,
            config=dots1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Dots1 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, dots1_config, small_model_config):
        """Test Dots1 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="dots1",
            model_cls=Dots1ForCausalLM,
            config=dots1_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Dots1 generation failed: {result.error_message}"

    def test_sanitize_stacks_experts(self, dots1_config):
        """Ensure Dots1 sanitize stacks per-expert weights correctly."""
        model = Dots1ForCausalLM(dots1_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        # Mimic upstream shard names for MoE layer (layer 1, since first_k_dense_replace=1)
        prefix = "model.layers.1"
        for dst in ["gate_proj", "down_proj", "up_proj"]:
            stacked_key = f"{prefix}.mlp.switch_mlp.{dst}.weight"
            if stacked_key in upstream_weights:
                tensor = upstream_weights.pop(stacked_key)
                for expert_idx in range(tensor.shape[0]):
                    upstream_weights[f"{prefix}.mlp.experts.{expert_idx}.{dst}.weight"] = tensor[expert_idx]

        sanitized = model.sanitize(upstream_weights)

        assert set(sanitized) == set(local_weights)
