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

"""Tests for OLMo model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.olmo import OlmoConfig, OlmoForCausalLM, OlmoModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestOlmo:
    """Test suite for OLMo model."""

    @pytest.fixture
    def olmo_config(self, small_model_config):
        """Create a tiny OLMo config."""
        return OlmoConfig(
            vocab_size=small_model_config["vocab_size"],
            d_model=small_model_config["hidden_size"],
            n_layers=small_model_config["num_hidden_layers"],
            n_heads=small_model_config["num_attention_heads"],
            mlp_hidden_size=small_model_config["intermediate_size"],
            weight_tying=False,
        )

    def test_registry(self):
        """OLMo should register under the exact HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "olmo")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "olmo")

        assert base_registration.module is OlmoModel
        assert base_registration.config is OlmoConfig
        assert lm_registration.module is OlmoForCausalLM
        assert lm_registration.config is OlmoConfig

    def test_causal_lm(self, olmo_config, small_model_config):
        """Test OLMo causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="olmo",
            model_cls=OlmoForCausalLM,
            config=olmo_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OLMo CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, olmo_config, small_model_config):
        """Test OLMo generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="olmo",
            model_cls=OlmoForCausalLM,
            config=olmo_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"OLMo generation failed: {result.error_message}"

    def test_mlp_ratio_default(self):
        """mlp_hidden_size should default to mlp_ratio * d_model."""
        config = OlmoConfig(d_model=64, mlp_ratio=4)
        assert config.mlp_hidden_size == 256

    def test_sanitize_removes_rotary_inv_freq(self, olmo_config):
        """Sanitize should remove rotary_emb.inv_freq keys."""
        model = OlmoModel(olmo_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
