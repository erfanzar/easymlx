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

"""Tests for RecurrentGemma model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.recurrent_gemma import (
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM,
    RecurrentGemmaModel,
)

from .test_utils import CausalLMTester


class TestRecurrentGemma:
    """Test suite for RecurrentGemma (Griffin) model."""

    @pytest.fixture
    def recurrent_gemma_config(self):
        return RecurrentGemmaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            lru_width=64,
            conv1d_width=4,
            logits_soft_cap=30.0,
            attention_window_size=16,
            block_types=["attention", "recurrent"],
            embeddings_scale_by_sqrt_dim=True,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(
            TaskType.BASE_MODULE, "recurrent_gemma"
        )
        lm_registration = registry.get_module_registration(
            TaskType.CAUSAL_LM, "recurrent_gemma"
        )

        assert base_registration.module is RecurrentGemmaModel
        assert base_registration.config is RecurrentGemmaConfig
        assert lm_registration.module is RecurrentGemmaForCausalLM
        assert lm_registration.config is RecurrentGemmaConfig

    def test_causal_lm(self, recurrent_gemma_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="recurrent_gemma",
            model_cls=RecurrentGemmaForCausalLM,
            config=recurrent_gemma_config,
            small_model_config=small_model_config,
        )
        assert result.success, (
            f"RecurrentGemma CAUSAL_LM failed: {result.error_message}"
        )

    def test_logit_soft_cap(self, recurrent_gemma_config):
        """Verify logit soft capping is configured."""
        model = RecurrentGemmaForCausalLM(recurrent_gemma_config)
        assert model._logit_cap == 30.0

    def test_block_types(self, recurrent_gemma_config):
        """Verify block type assignment."""
        assert recurrent_gemma_config.block_types == [
            "attention",
            "recurrent",
        ]

    def test_sanitize(self, recurrent_gemma_config):
        import mlx.core as mx

        model = RecurrentGemmaForCausalLM(recurrent_gemma_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        # Simulate upstream conv_1d weight format
        upstream_weights = dict(local_weights)
        for k, v in upstream_weights.items():
            if "conv_1d.weight" in k and v.ndim == 3:
                upstream_weights[k] = v.moveaxis(1, 2)

        sanitized = model.sanitize(upstream_weights)
        for k, v in sanitized.items():
            if "conv_1d.weight" in k and v.ndim == 3:
                assert v.shape[-1] == 1, (
                    f"conv_1d weight {k} not transposed correctly"
                )
