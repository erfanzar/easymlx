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

"""Tests for Pixtral model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.pixtral import PixtralConfig, PixtralForCausalLM, PixtralModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestPixtral:
    """Test suite for Pixtral model."""

    @pytest.fixture
    def pixtral_config(self, small_model_config):
        """Create a Pixtral-specific text config."""
        return PixtralConfig(
            text_config={
                "vocab_size": small_model_config["vocab_size"],
                "hidden_size": small_model_config["hidden_size"],
                "intermediate_size": small_model_config["intermediate_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "head_dim": small_model_config["head_dim"],
                "max_position_embeddings": small_model_config["max_position_embeddings"],
                "tie_word_embeddings": False,
            }
        )

    def test_registry_wiring(self):
        """Pixtral should register under the expected HF model type."""
        causal_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "pixtral")
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "pixtral")

        assert causal_registration.module is PixtralForCausalLM
        assert base_registration.module is PixtralModel
        assert PixtralConfig.model_type == "pixtral"

    def test_causal_lm(self, pixtral_config, small_model_config):
        """Test PixtralForCausalLM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="pixtral",
            model_cls=PixtralForCausalLM,
            config=pixtral_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Pixtral CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, pixtral_config, small_model_config):
        """Test Pixtral text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="pixtral",
            model_cls=PixtralForCausalLM,
            config=pixtral_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Pixtral generation failed: {result.error_message}"

    def test_sanitize_drops_vision_keys(self, pixtral_config):
        """Ensure Pixtral sanitize normalizes upstream checkpoint prefixes."""
        model = PixtralForCausalLM(pixtral_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = {}
        for key, value in local_weights.items():
            if key == "lm_head.weight":
                upstream_weights["language_model.lm_head.weight"] = value
            else:
                upstream_weights[key.replace("model.model.", "language_model.model.", 1)] = value
        upstream_weights.update(
            {
                "vision_tower.blocks.0.weight": mx.ones((2, 2)),
                "multi_modal_projector.weight": mx.ones((2, 2)),
            }
        )

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.blocks.0.weight" not in sanitized
        assert "multi_modal_projector.weight" not in sanitized
        assert set(sanitized) == set(local_weights)
