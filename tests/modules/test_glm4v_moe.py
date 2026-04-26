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

"""Tests for GLM-4V MoE model."""

import mlx.core as mx
import pytest
from easymlx.modules.glm4v_moe import Glm4VMoeConfig, Glm4VMoeForConditionalGeneration

from .test_utils import CausalLMTester


def _build_glm4v_vlm_inputs(config):
    vision_cfg = config.vision_config
    if not isinstance(vision_cfg, dict):
        vision_cfg = vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vision_cfg.__dict__

    image_size = int(vision_cfg["image_size"])
    patch_size = int(vision_cfg["patch_size"])
    spatial_merge_size = int(vision_cfg["spatial_merge_size"])
    temporal_patch_size = int(vision_cfg["temporal_patch_size"])
    in_channels = int(vision_cfg.get("in_channels", 3))

    grid_h = image_size // patch_size
    grid_w = image_size // patch_size
    t = 1
    num_patches = t * grid_h * grid_w
    num_image_tokens = num_patches // (spatial_merge_size**2)

    image_grid_thw = mx.array([[t, grid_h, grid_w]], dtype=mx.int32)
    pixel_values = mx.zeros((num_patches, in_channels, temporal_patch_size, patch_size, patch_size), dtype=mx.float32)
    return num_image_tokens, image_grid_thw, pixel_values


class TestGlm4VMoe:
    """Test suite for GLM-4V MoE model."""

    @pytest.fixture
    def glm4v_moe_config(self, small_model_config):
        """Create GLM-4V MoE-specific config."""
        vocab_size = small_model_config["vocab_size"]
        hidden_size = small_model_config["hidden_size"]
        num_heads = small_model_config["num_attention_heads"]
        rope_scaling = {"type": "default", "mrope_section": [2, 2, 4]}
        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": small_model_config["intermediate_size"],
            "num_hidden_layers": small_model_config["num_hidden_layers"],
            "num_attention_heads": num_heads,
            "num_key_value_heads": small_model_config["num_key_value_heads"],
            "head_dim": small_model_config["head_dim"],
            "partial_rotary_factor": 1.0,
            "rope_scaling": rope_scaling,
            "moe_intermediate_size": 32,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "first_k_dense_replace": small_model_config["num_hidden_layers"] + 1,
        }
        vision_config = {
            "model_type": "glm4v",
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "patch_size": 2,
            "image_size": 4,
            "in_channels": 3,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": hidden_size,
        }
        return Glm4VMoeConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=vocab_size - 1,
            video_token_id=vocab_size - 2,
            vision_start_token_id=vocab_size - 3,
            vision_end_token_id=vocab_size - 4,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

    def test_text_only(self, glm4v_moe_config, small_model_config):
        """Test GLM-4V MoE text-only forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4v_moe",
            model_cls=Glm4VMoeForConditionalGeneration,
            config=glm4v_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-4V MoE text-only failed: {result.error_message}"

    def test_vision_language(self, glm4v_moe_config, small_model_config):
        """Test GLM-4V MoE vision-language forward pass."""
        model = Glm4VMoeForConditionalGeneration(glm4v_moe_config)
        num_image_tokens, image_grid_thw, pixel_values = _build_glm4v_vlm_inputs(glm4v_moe_config)

        batch_size = 1
        vocab_size = small_model_config["vocab_size"]
        image_token_id = glm4v_moe_config.image_token_id

        prefix = mx.arange(batch_size, dtype=mx.int32).reshape(batch_size, 1)
        image_tokens = mx.full((batch_size, num_image_tokens), image_token_id, dtype=mx.int32)
        input_ids = mx.concatenate([prefix, image_tokens], axis=1)
        attention_mask = mx.ones((batch_size, input_ids.shape[1]), dtype=mx.int32)

        logits = model(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            return_dict=False,
        )
        assert logits.shape == (batch_size, input_ids.shape[1], vocab_size)

    def test_generation(self, glm4v_moe_config, small_model_config):
        """Test GLM-4V MoE text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4v_moe",
            model_cls=Glm4VMoeForConditionalGeneration,
            config=glm4v_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GLM-4V MoE generation failed: {result.error_message}"
