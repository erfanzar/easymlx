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

"""Tests for Qwen3 Omni MoE model."""

import pytest

from easymlx.modules.qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration

from .test_utils import EasyMLXOnlyTester


class TestQwen3OmniMoe:
    """Test suite for Qwen3 Omni MoE model."""

    @pytest.fixture
    def qwen3_omni_moe_config(self, small_model_config):
        """Create Qwen3 Omni MoE-specific config."""
        vocab_size = small_model_config["vocab_size"]
        thinker_config = {
            "audio_config": {},
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 4,
                "depth": 1,
                "out_hidden_size": small_model_config["hidden_size"],
                "patch_size": 4,
                "in_channels": 3,
            },
            "text_config": {
                "vocab_size": vocab_size,
                "hidden_size": small_model_config["hidden_size"],
                "intermediate_size": small_model_config["intermediate_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "head_dim": small_model_config["head_dim"],
                "num_experts": small_model_config["num_experts"],
                "num_experts_per_tok": small_model_config["num_experts_per_tok"],
            },
            "image_token_id": vocab_size - 1,
            "video_token_id": vocab_size - 2,
        }
        return Qwen3OmniMoeConfig(thinker_config=thinker_config)

    def test_causal_lm(self, qwen3_omni_moe_config, small_model_config):
        """Test Qwen3 Omni MoE text-only forward pass."""
        tester = EasyMLXOnlyTester()
        result = tester.run(
            module_name="qwen3_omni_moe",
            model_cls=Qwen3OmniMoeForConditionalGeneration,
            config=qwen3_omni_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3 Omni MoE forward failed: {result.error_message}"
