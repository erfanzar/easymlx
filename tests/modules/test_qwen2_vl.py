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

"""Tests for Qwen2-VL model."""

import pytest

from easymlx.modules.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from .test_utils import CausalLMTester, VisionLanguageTester


class TestQwen2VL:
    """Test suite for Qwen2-VL vision-language model."""

    @pytest.fixture
    def qwen2_vl_config(self, small_model_config):
        """Create Qwen2-VL-specific config."""
        vocab_size = small_model_config["vocab_size"]
        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": small_model_config["hidden_size"],
            "intermediate_size": small_model_config["intermediate_size"],
            "num_hidden_layers": small_model_config["num_hidden_layers"],
            "num_attention_heads": small_model_config["num_attention_heads"],
            "num_key_value_heads": small_model_config["num_key_value_heads"],
            "max_position_embeddings": small_model_config["max_position_embeddings"],
        }
        vision_config = {
            "depth": 1,
            "embed_dim": small_model_config["hidden_size"],
            "hidden_size": small_model_config["hidden_size"],
            "num_heads": small_model_config["num_attention_heads"],
            "patch_size": 4,
            "in_channels": 3,
        }
        return Qwen2VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=vocab_size - 1,
            video_token_id=vocab_size - 2,
            vision_start_token_id=vocab_size - 3,
            vision_end_token_id=vocab_size - 4,
        )

    @pytest.fixture
    def vlm_config(self, qwen2_vl_config, small_model_config):
        """Create VLM-specific config for Qwen2-VL."""
        image_size = 8
        patch_size = qwen2_vl_config.vision_config.patch_size
        num_image_tokens = (image_size // patch_size) ** 2
        return {
            "image_token_id": qwen2_vl_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (small_model_config["batch_size"], 3, image_size, image_size),
        }

    def test_vision_language(self, qwen2_vl_config, small_model_config, vlm_config):
        """Test Qwen2VLForConditionalGeneration with vision inputs."""
        tester = VisionLanguageTester()
        result = tester.run(
            module_name="qwen2_vl",
            model_cls=Qwen2VLForConditionalGeneration,
            config=qwen2_vl_config,
            small_model_config=small_model_config,
            vlm_config=vlm_config,
        )
        assert result.success, f"Qwen2-VL VLM failed: {result.error_message}"

    def test_text_only(self, qwen2_vl_config, small_model_config):
        """Test Qwen2-VL text-only forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen2_vl",
            model_cls=Qwen2VLForConditionalGeneration,
            config=qwen2_vl_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen2-VL text-only failed: {result.error_message}"

    def test_generation(self, qwen2_vl_config, small_model_config):
        """Test Qwen2-VL text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen2_vl",
            model_cls=Qwen2VLForConditionalGeneration,
            config=qwen2_vl_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Qwen2-VL generation failed: {result.error_message}"
