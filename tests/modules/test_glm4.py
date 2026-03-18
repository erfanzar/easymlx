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

"""Tests for GLM-4 model."""

import pytest

from easymlx.modules.glm4 import Glm4Config, Glm4ForCausalLM

from .test_utils import CausalLMTester


class TestGlm4:
    """Test suite for GLM-4 model."""

    @pytest.fixture
    def glm4_config(self, small_model_config):
        """Create GLM-4-specific config."""
        return Glm4Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, glm4_config, small_model_config):
        """Test Glm4ForCausalLM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4",
            model_cls=Glm4ForCausalLM,
            config=glm4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-4 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, glm4_config, small_model_config):
        """Test GLM-4 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4",
            model_cls=Glm4ForCausalLM,
            config=glm4_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GLM-4 generation failed: {result.error_message}"
