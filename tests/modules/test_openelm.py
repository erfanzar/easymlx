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

"""Tests for OpenELM model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.openelm import OpenELMConfig, OpenELMForCausalLM, OpenELMModel

from .test_utils import CausalLMTester


class TestOpenELM:
    """Test suite for OpenELM model."""

    @pytest.fixture
    def openelm_config(self, small_model_config):
        return OpenELMConfig(
            vocab_size=small_model_config["vocab_size"],
            model_dim=small_model_config["hidden_size"],
            head_dim=small_model_config["head_dim"],
            num_transformer_layers=small_model_config["num_hidden_layers"],
            num_query_heads=[4, 4],
            num_kv_heads=[2, 2],
            ffn_multipliers=[2.0, 2.0],
            ffn_dim_divisor=8,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            share_input_output_layers=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "openelm")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "openelm")

        assert base_registration.module is OpenELMModel
        assert base_registration.config is OpenELMConfig
        assert lm_registration.module is OpenELMForCausalLM
        assert lm_registration.config is OpenELMConfig

    def test_causal_lm(self, openelm_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="openelm",
            model_cls=OpenELMForCausalLM,
            config=openelm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OpenELM CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, openelm_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="openelm",
            model_cls=OpenELMForCausalLM,
            config=openelm_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"OpenELM generation failed: {result.error_message}"

    def test_sanitize(self, openelm_config):
        model = OpenELMForCausalLM(openelm_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
