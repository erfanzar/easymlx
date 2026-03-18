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

"""Tests for Llama4 model."""

import pytest

from easymlx.modules.llama4 import Llama4Config, Llama4ForConditionalGeneration

from .test_utils import CausalLMTester


class TestLlama4:
    """Test suite for Llama4 model."""

    @pytest.fixture
    def llama4_config(self, small_model_config):
        """Create Llama4-specific config."""
        vocab_size = small_model_config["vocab_size"]
        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": small_model_config["hidden_size"],
            "intermediate_size": small_model_config["intermediate_size"],
            "intermediate_size_mlp": small_model_config["intermediate_size"],
            "num_hidden_layers": small_model_config["num_hidden_layers"],
            "num_attention_heads": small_model_config["num_attention_heads"],
            "num_key_value_heads": small_model_config["num_key_value_heads"],
            "head_dim": small_model_config["head_dim"],
            "num_local_experts": small_model_config["num_local_experts"],
            "num_experts_per_tok": 1,
            "interleave_moe_layer_step": 2,
            "no_rope_layer_interval": 2,
        }
        vision_config = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vision_output_dim": small_model_config["hidden_size"],
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
        }
        return Llama4Config(
            text_config=text_config,
            vision_config=vision_config,
            boi_token_index=vocab_size - 3,
            eoi_token_index=vocab_size - 2,
            image_token_index=vocab_size - 1,
        )

    def test_causal_lm(self, llama4_config, small_model_config):
        """Test Llama4ForConditionalGeneration text-only forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="llama4",
            model_cls=Llama4ForConditionalGeneration,
            config=llama4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama4 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, llama4_config, small_model_config):
        """Test Llama4 text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama4",
            model_cls=Llama4ForConditionalGeneration,
            config=llama4_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Llama4 generation failed: {result.error_message}"
