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

"""Tests for LongcatFlash model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.longcat_flash import LongcatFlashConfig, LongcatFlashForCausalLM, LongcatFlashModel

from .test_utils import CausalLMTester


class TestLongcatFlash:
    """Test suite for LongcatFlash model."""

    @pytest.fixture
    def longcat_flash_config(self, small_model_config):
        """Create a tiny LongcatFlash config."""
        return LongcatFlashConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            ffn_hidden_size=small_model_config["intermediate_size"],
            expert_ffn_hidden_size=small_model_config["intermediate_size"],
            num_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            kv_lora_rank=32,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            moe_topk=2,
            n_routed_experts=4,
            zero_expert_num=1,
            zero_expert_type="identity",
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """LongcatFlash should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "longcat_flash")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "longcat_flash")

        assert base_registration.module is LongcatFlashModel
        assert base_registration.config is LongcatFlashConfig
        assert lm_registration.module is LongcatFlashForCausalLM
        assert lm_registration.config is LongcatFlashConfig

    def test_causal_lm(self, longcat_flash_config, small_model_config):
        """Test LongcatFlash causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="longcat_flash",
            model_cls=LongcatFlashForCausalLM,
            config=longcat_flash_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"LongcatFlash CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, longcat_flash_config, small_model_config):
        """Test LongcatFlash generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="longcat_flash",
            model_cls=LongcatFlashForCausalLM,
            config=longcat_flash_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"LongcatFlash generation failed: {result.error_message}"

    def test_sanitize(self, longcat_flash_config):
        """Ensure LongcatFlash sanitize removes rotary embedding keys."""
        model = LongcatFlashForCausalLM(longcat_flash_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.zeros((16,))

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
