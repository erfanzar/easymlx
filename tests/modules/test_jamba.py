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

"""Tests for Jamba model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.jamba import JambaConfig, JambaForCausalLM, JambaModel

from .test_utils import CausalLMTester


class TestJamba:
    """Test suite for Jamba hybrid attention/SSM model."""

    @pytest.fixture
    def jamba_config(self):
        return JambaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_layer_offset=0,
            attn_layer_period=2,
            expert_layer_offset=0,
            expert_layer_period=2,
            num_experts=4,
            num_experts_per_tok=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(
            TaskType.BASE_MODULE, "jamba"
        )
        lm_registration = registry.get_module_registration(
            TaskType.CAUSAL_LM, "jamba"
        )

        assert base_registration.module is JambaModel
        assert base_registration.config is JambaConfig
        assert lm_registration.module is JambaForCausalLM
        assert lm_registration.config is JambaConfig

    def test_causal_lm(self, jamba_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="jamba",
            model_cls=JambaForCausalLM,
            config=jamba_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Jamba CAUSAL_LM failed: {result.error_message}"

    def test_layer_types(self, jamba_config):
        """Verify layer type assignment matches config."""
        assert jamba_config.layers_block_type == [
            "attention",  # layer 0: 0 % 2 == 0
            "mamba",      # layer 1: 1 % 2 != 0
            "attention",  # layer 2: 2 % 2 == 0
            "mamba",      # layer 3: 3 % 2 != 0
        ]

    def test_sanitize(self, jamba_config):
        import mlx.core as mx

        model = JambaForCausalLM(jamba_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        # Simulate upstream conv1d weight format
        upstream_weights = dict(local_weights)
        for k, v in upstream_weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                upstream_weights[k] = v.moveaxis(1, 2)

        sanitized = model.sanitize(upstream_weights)
        for k, v in sanitized.items():
            if "conv1d.weight" in k and v.ndim == 3:
                assert v.shape[-1] == 1, (
                    f"conv1d weight {k} not transposed correctly"
                )
