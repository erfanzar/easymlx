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

"""Tests for Qwen3-VL-MoE model."""

import pytest
from easymlx.modules.qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

from .test_utils import CausalLMTester, VisionLanguageTester


class TestQwen3VLMoe:
    """Test suite for Qwen3-VL-MoE vision-language model."""

    @pytest.fixture
    def qwen3_vl_moe_config(self, small_model_config):
        """Create Qwen3-VL-MoE-specific config."""
        vocab_size = small_model_config["vocab_size"]
        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": small_model_config["hidden_size"],
            "intermediate_size": small_model_config["intermediate_size"],
            "num_hidden_layers": small_model_config["num_hidden_layers"],
            "num_attention_heads": small_model_config["num_attention_heads"],
            "num_key_value_heads": small_model_config["num_key_value_heads"],
            "head_dim": small_model_config["head_dim"],
            "max_position_embeddings": small_model_config["max_position_embeddings"],
            "num_experts": small_model_config["num_experts"],
            "num_experts_per_tok": small_model_config["num_experts_per_tok"],
            "moe_intermediate_size": small_model_config["intermediate_size"],
        }
        vision_config = {
            "depth": 1,
            "hidden_size": small_model_config["hidden_size"],
            "intermediate_size": small_model_config["intermediate_size"],
            "num_heads": small_model_config["num_attention_heads"],
            "patch_size": 4,
            "in_channels": 3,
            "out_hidden_size": small_model_config["hidden_size"],
        }
        return Qwen3VLMoeConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=vocab_size - 1,
            video_token_id=vocab_size - 2,
            vision_start_token_id=vocab_size - 3,
            vision_end_token_id=vocab_size - 4,
        )

    @pytest.fixture
    def vlm_config(self, qwen3_vl_moe_config, small_model_config):
        """Create VLM-specific config for Qwen3-VL-MoE."""
        image_size = 8
        patch_size = qwen3_vl_moe_config.vision_config.patch_size
        num_image_tokens = (image_size // patch_size) ** 2
        return {
            "image_token_id": qwen3_vl_moe_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (small_model_config["batch_size"], 3, image_size, image_size),
        }

    def test_vision_language(self, qwen3_vl_moe_config, small_model_config, vlm_config):
        """Test Qwen3VLMoeForConditionalGeneration with vision inputs."""
        tester = VisionLanguageTester()
        result = tester.run(
            module_name="qwen3_vl_moe",
            model_cls=Qwen3VLMoeForConditionalGeneration,
            config=qwen3_vl_moe_config,
            small_model_config=small_model_config,
            vlm_config=vlm_config,
        )
        assert result.success, f"Qwen3-VL-MoE VLM failed: {result.error_message}"

    def test_text_only(self, qwen3_vl_moe_config, small_model_config):
        """Test Qwen3-VL-MoE text-only forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_vl_moe",
            model_cls=Qwen3VLMoeForConditionalGeneration,
            config=qwen3_vl_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3-VL-MoE text-only failed: {result.error_message}"

    def test_generation(self, qwen3_vl_moe_config, small_model_config):
        """Test Qwen3-VL-MoE text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_vl_moe",
            model_cls=Qwen3VLMoeForConditionalGeneration,
            config=qwen3_vl_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Qwen3-VL-MoE generation failed: {result.error_message}"
