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

"""Tests for Qwen3 model."""

import pytest
from easymlx.modules.qwen3 import Qwen3Config, Qwen3ForCausalLM

from .test_utils import CausalLMTester


class TestQwen3:
    """Test suite for Qwen3 model."""

    @pytest.fixture
    def qwen3_config(self, small_model_config):
        """Create Qwen3-specific config."""
        return Qwen3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
        )

    def test_causal_lm(self, qwen3_config, small_model_config):
        """Test Qwen3ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3",
            model_cls=Qwen3ForCausalLM,
            config=qwen3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, qwen3_config, small_model_config):
        """Test Qwen3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3",
            model_cls=Qwen3ForCausalLM,
            config=qwen3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Qwen3 generation failed: {result.error_message}"

    def test_config_accepts_null_sliding_window(self):
        """HF Qwen3 configs may set `sliding_window` to null."""
        config = Qwen3Config(use_sliding_window=False, sliding_window=None)

        assert config.sliding_window is None
