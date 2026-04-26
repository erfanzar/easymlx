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

"""Tests for YouTu LLM model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.youtu_llm import YouTuLLMConfig, YouTuLLMForCausalLM, YouTuLLMModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestYouTuLLM:
    """Test suite for YouTu LLM model."""

    @pytest.fixture
    def youtu_llm_config(self, small_model_config):
        """Create a tiny YouTu LLM config with MLA."""
        return YouTuLLMConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            v_head_dim=16,
            qk_nope_head_dim=8,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        """YouTu LLM should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "youtu_llm")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "youtu_llm")

        assert base_registration.module is YouTuLLMModel
        assert base_registration.config is YouTuLLMConfig
        assert lm_registration.module is YouTuLLMForCausalLM
        assert lm_registration.config is YouTuLLMConfig

    def test_causal_lm(self, youtu_llm_config, small_model_config):
        """Test YouTu LLM causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="youtu_llm",
            model_cls=YouTuLLMForCausalLM,
            config=youtu_llm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"YouTu LLM CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, youtu_llm_config, small_model_config):
        """Test YouTu LLM generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="youtu_llm",
            model_cls=YouTuLLMForCausalLM,
            config=youtu_llm_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"YouTu LLM generation failed: {result.error_message}"

    def test_sanitize_strips_rotary_keys(self, youtu_llm_config):
        """Ensure sanitize removes rotary_emb.inv_freq and tied lm_head keys."""
        model = YouTuLLMForCausalLM(youtu_llm_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.zeros((4,))

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert "lm_head.weight" not in sanitized
