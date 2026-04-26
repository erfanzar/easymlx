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

"""Tests for SmolLM-3 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.smollm3 import SmolLM3Config, SmolLM3ForCausalLM, SmolLM3Model, SmolLM3NoPE
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestSmolLM3:
    """Test suite for SmolLM-3 model."""

    @pytest.fixture
    def smollm3_config(self, small_model_config):
        """Create a tiny SmolLM-3 config."""
        return SmolLM3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            no_rope_layer_interval=2,
        )

    def test_registry(self):
        """SmolLM-3 should register under the exact HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "smollm3")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "smollm3")

        assert base_registration.module is SmolLM3Model
        assert base_registration.config is SmolLM3Config
        assert lm_registration.module is SmolLM3ForCausalLM
        assert lm_registration.config is SmolLM3Config

    def test_causal_lm(self, smollm3_config, small_model_config):
        """Test SmolLM-3 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="smollm3",
            model_cls=SmolLM3ForCausalLM,
            config=smollm3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SmolLM-3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, smollm3_config, small_model_config):
        """Test SmolLM-3 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="smollm3",
            model_cls=SmolLM3ForCausalLM,
            config=smollm3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"SmolLM-3 generation failed: {result.error_message}"

    def test_no_rope_layers(self, smollm3_config):
        """Every second layer should disable RoPE when interval=2."""
        model = SmolLM3Model(smollm3_config)

        assert model.no_rope_layers == [1, 0]
        assert not isinstance(model.layers[0].self_attn.rope, SmolLM3NoPE)
        assert isinstance(model.layers[1].self_attn.rope, SmolLM3NoPE)

    def test_sanitize_maps_upstream_checkpoint_keys(self, smollm3_config):
        model = SmolLM3ForCausalLM(smollm3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = {key.replace("model.model.", "model.", 1): value for key, value in local_weights.items()}

        sanitized = model.sanitize(upstream_weights)

        assert set(sanitized) == set(local_weights)
