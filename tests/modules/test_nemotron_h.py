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

"""Tests for Nemotron-H model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.nemotron_h import NemotronHConfig, NemotronHForCausalLM, NemotronHModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestNemotronH:
    """Test suite for Nemotron-H model."""

    @pytest.fixture
    def nemotron_h_config(self, small_model_config):
        """Create a tiny Nemotron-H config with mixed block types."""
        return NemotronHConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=2,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            hybrid_override_pattern=["*", "-"],
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """Nemotron-H should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "nemotron_h")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "nemotron_h")

        assert base_registration.module is NemotronHModel
        assert base_registration.config is NemotronHConfig
        assert lm_registration.module is NemotronHForCausalLM
        assert lm_registration.config is NemotronHConfig

    def test_causal_lm(self, nemotron_h_config, small_model_config):
        """Test Nemotron-H causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="nemotron_h",
            model_cls=NemotronHForCausalLM,
            config=nemotron_h_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Nemotron-H CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, nemotron_h_config, small_model_config):
        """Test Nemotron-H generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="nemotron_h",
            model_cls=NemotronHForCausalLM,
            config=nemotron_h_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Nemotron-H generation failed: {result.error_message}"

    def test_sanitize(self, nemotron_h_config):
        """Ensure Nemotron-H sanitize removes rotary embedding keys."""
        model = NemotronHForCausalLM(nemotron_h_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.zeros((16,))

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
