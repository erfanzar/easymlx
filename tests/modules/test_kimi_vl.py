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

"""Tests for Kimi VL model (text backbone only)."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.kimi_vl import KimiVLConfig, KimiVLForCausalLM, KimiVLModel

from .test_utils import CausalLMTester


class TestKimiVL:
    """Test suite for Kimi VL text backbone."""

    @pytest.fixture
    def kimi_vl_config(self, small_model_config):
        """Create a tiny Kimi VL config (text backbone = DeepSeek V3)."""
        return KimiVLConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            kv_lora_rank=16,
            q_lora_rank=None,
            qk_rope_head_dim=8,
            v_head_dim=16,
            qk_nope_head_dim=8,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            head_dim=small_model_config["head_dim"],
            n_routed_experts=None,
            n_shared_experts=None,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """Kimi VL should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "kimi_vl")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "kimi_vl")

        assert base_registration.module is KimiVLModel
        assert base_registration.config is KimiVLConfig
        assert lm_registration.module is KimiVLForCausalLM
        assert lm_registration.config is KimiVLConfig

    def test_causal_lm(self, kimi_vl_config, small_model_config):
        """Test Kimi VL causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="kimi_vl",
            model_cls=KimiVLForCausalLM,
            config=kimi_vl_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Kimi VL CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, kimi_vl_config, small_model_config):
        """Test Kimi VL generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="kimi_vl",
            model_cls=KimiVLForCausalLM,
            config=kimi_vl_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Kimi VL generation failed: {result.error_message}"

    def test_sanitize_strips_vision_weights(self, kimi_vl_config):
        """Ensure sanitize strips vision_tower and multi_modal_projector weights."""
        model = KimiVLForCausalLM(kimi_vl_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        # Add fake vision weights that should be stripped
        upstream_weights["vision_tower.encoder.weight"] = mx.zeros((4, 4))
        upstream_weights["multi_modal_projector.linear.weight"] = mx.zeros((4, 4))

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.encoder.weight" not in sanitized
        assert "multi_modal_projector.linear.weight" not in sanitized
