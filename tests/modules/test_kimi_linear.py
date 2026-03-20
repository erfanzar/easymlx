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

"""Tests for Kimi Linear model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM, KimiLinearModel

from .test_utils import CausalLMTester


class TestKimiLinear:
    """Test suite for Kimi Linear model."""

    @pytest.fixture
    def kimi_linear_config(self, small_model_config):
        """Create a tiny Kimi Linear config (dense, no MoE, no KDA layers)."""
        return KimiLinearConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            num_experts=0,
            linear_attn_config={
                "kda_layers": [],
                "num_heads": small_model_config["num_attention_heads"],
                "head_dim": small_model_config["head_dim"],
                "short_conv_kernel_size": 4,
            },
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """Kimi Linear should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "kimi_linear")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "kimi_linear")

        assert base_registration.module is KimiLinearModel
        assert base_registration.config is KimiLinearConfig
        assert lm_registration.module is KimiLinearForCausalLM
        assert lm_registration.config is KimiLinearConfig

    def test_causal_lm(self, kimi_linear_config, small_model_config):
        """Test Kimi Linear causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="kimi_linear",
            model_cls=KimiLinearForCausalLM,
            config=kimi_linear_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Kimi Linear CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, kimi_linear_config, small_model_config):
        """Test Kimi Linear generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="kimi_linear",
            model_cls=KimiLinearForCausalLM,
            config=kimi_linear_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Kimi Linear generation failed: {result.error_message}"

    def test_sanitize_strips_rotary_keys(self, kimi_linear_config):
        """Ensure sanitize removes rotary_emb.inv_freq keys."""
        model = KimiLinearForCausalLM(kimi_linear_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((4,))

        sanitized = model.sanitize(upstream_weights)

        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
