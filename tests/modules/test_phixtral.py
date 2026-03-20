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

"""Tests for Phixtral model."""

import pytest

import mlx.core as mx
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.phixtral import PhixtralConfig, PhixtralForCausalLM, PhixtralModel

from .test_utils import CausalLMTester


class TestPhixtral:
    """Test suite for Phixtral model."""

    @pytest.fixture
    def phixtral_config(self):
        """Create Phixtral-specific config with small dimensions."""
        return PhixtralConfig(
            num_vocab=128,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            rotary_dim=8,
            num_local_experts=4,
            num_experts_per_tok=2,
        )

    def test_registry(self):
        """Test that Phixtral is properly registered."""
        base_reg = registry.get_module_registration(TaskType.BASE_MODULE, "phixtral")
        lm_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "phixtral")
        assert base_reg.module is PhixtralModel
        assert lm_reg.module is PhixtralForCausalLM

    def test_causal_lm(self, phixtral_config, small_model_config):
        """Test PhixtralForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phixtral",
            model_cls=PhixtralForCausalLM,
            config=phixtral_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Phixtral CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, phixtral_config, small_model_config):
        """Test Phixtral text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phixtral",
            model_cls=PhixtralForCausalLM,
            config=phixtral_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Phixtral generation failed: {result.error_message}"

    def test_sanitize(self, phixtral_config):
        """Test weight sanitization."""
        model = PhixtralForCausalLM(phixtral_config)
        local_weights = dict(tree_flatten(model.parameters()))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.ones((8,))
        sanitized = model.sanitize(upstream_weights)
        assert "rotary_emb.inv_freq" not in sanitized
