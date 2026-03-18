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

"""Tests for Llama4 vision-language model."""

import pytest

from easymlx.modules.llama4 import Llama4Config, Llama4ForConditionalGeneration

from .test_utils import CausalLMTester, VisionLanguageTester


class TestLlama4Vision:
    """Test suite for Llama4 vision-language model."""

    @pytest.fixture
    def llama4_vision_config(self, small_model_config):
        """Create Llama4 vision config."""
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

    @pytest.fixture
    def vlm_config(self, llama4_vision_config, small_model_config):
        """Create VLM-specific config."""
        vision_config = llama4_vision_config.vision_config
        image_size = vision_config["image_size"]
        patch_size = vision_config["patch_size"]
        num_image_tokens = (image_size // patch_size) ** 2
        return {
            "image_token_id": llama4_vision_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (small_model_config["batch_size"], 3, image_size, image_size),
        }

    def test_vision_language(self, llama4_vision_config, small_model_config, vlm_config):
        """Test Llama4ForConditionalGeneration with vision inputs."""
        tester = VisionLanguageTester()
        result = tester.run(
            module_name="llama4",
            model_cls=Llama4ForConditionalGeneration,
            config=llama4_vision_config,
            small_model_config=small_model_config,
            vlm_config=vlm_config,
        )
        assert result.success, f"Llama4 VLM failed: {result.error_message}"

    def test_generation(self, llama4_vision_config, small_model_config):
        """Test Llama4 text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama4",
            model_cls=Llama4ForConditionalGeneration,
            config=llama4_vision_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Llama4 generation failed: {result.error_message}"
