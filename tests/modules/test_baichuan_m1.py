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

"""Tests for Baichuan M1 model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.baichuan_m1 import BaichuanM1Config, BaichuanM1ForCausalLM, BaichuanM1Model

from .test_utils import CausalLMTester


class TestBaichuanM1:
    """Test suite for Baichuan M1 model."""

    @pytest.fixture
    def baichuan_m1_config(self, small_model_config):
        """Create a tiny Baichuan M1 config."""
        return BaichuanM1Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            rope_theta=10000.0,
            sliding_window=32,
            sliding_window_layers=[0],
            conv_window=2,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """Baichuan M1 should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "baichuan_m1")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "baichuan_m1")

        assert base_registration.module is BaichuanM1Model
        assert base_registration.config is BaichuanM1Config
        assert lm_registration.module is BaichuanM1ForCausalLM
        assert lm_registration.config is BaichuanM1Config

    def test_causal_lm(self, baichuan_m1_config, small_model_config):
        """Test Baichuan M1 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="baichuan_m1",
            model_cls=BaichuanM1ForCausalLM,
            config=baichuan_m1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Baichuan M1 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, baichuan_m1_config, small_model_config):
        """Test Baichuan M1 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="baichuan_m1",
            model_cls=BaichuanM1ForCausalLM,
            config=baichuan_m1_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Baichuan M1 generation failed: {result.error_message}"

    def test_sanitize_normalizes_lm_head(self, baichuan_m1_config):
        """Ensure sanitize normalizes lm_head weights."""
        model = BaichuanM1ForCausalLM(baichuan_m1_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)

        # Add a rotary key that should be stripped
        upstream_weights["rotary_emb.inv_freq"] = mx.zeros((4,))

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        # lm_head.weight should be present and normalized
        assert "lm_head.weight" in sanitized
