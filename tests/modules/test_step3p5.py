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

"""Tests for Step 3.5 model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.step3p5 import Step3p5Config, Step3p5ForCausalLM, Step3p5Model

from .test_utils import CausalLMTester


class TestStep3p5:
    """Test suite for Step 3.5 model."""

    @pytest.fixture
    def step3p5_config(self, small_model_config):
        """Create a tiny Step 3.5 config with MoE on layer 1."""
        return Step3p5Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_attention_groups=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            sliding_window=32,
            moe_num_experts=4,
            moe_top_k=2,
            moe_intermediate_size=small_model_config["intermediate_size"],
            share_expert_dim=small_model_config["intermediate_size"],
            moe_layers_enum="1",
            use_head_wise_attn_gate=True,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """Step 3.5 should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "step3p5")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "step3p5")

        assert base_registration.module is Step3p5Model
        assert base_registration.config is Step3p5Config
        assert lm_registration.module is Step3p5ForCausalLM
        assert lm_registration.config is Step3p5Config

    def test_causal_lm(self, step3p5_config, small_model_config):
        """Test Step 3.5 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="step3p5",
            model_cls=Step3p5ForCausalLM,
            config=step3p5_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Step 3.5 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, step3p5_config, small_model_config):
        """Test Step 3.5 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="step3p5",
            model_cls=Step3p5ForCausalLM,
            config=step3p5_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Step 3.5 generation failed: {result.error_message}"

    def test_sanitize_remaps_moe_keys(self, step3p5_config):
        """Ensure sanitize handles Step 3.5 weight remapping."""
        model = Step3p5ForCausalLM(step3p5_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        # Add a rotary key that should be stripped
        upstream_weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((4,))

        sanitized = model.sanitize(upstream_weights)

        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
