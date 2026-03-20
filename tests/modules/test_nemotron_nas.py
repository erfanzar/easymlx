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

"""Tests for NemotronNAS model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.nemotron_nas import NemotronNASConfig, NemotronNASForCausalLM, NemotronNASModel

from .test_utils import CausalLMTester


class TestNemotronNAS:
    """Test suite for NemotronNAS model."""

    @pytest.fixture
    def nemotron_nas_config(self, small_model_config):
        return NemotronNASConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            block_configs=[
                {
                    "attention": {"n_heads_in_group": 2},
                    "ffn": {"ffn_mult": 2.0},
                },
                {
                    "attention": {"no_op": True},
                    "ffn": {"ffn_mult": 2.0},
                },
            ],
            hidden_act="silu",
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "nemotron-nas")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "nemotron-nas")

        assert base_registration.module is NemotronNASModel
        assert base_registration.config is NemotronNASConfig
        assert lm_registration.module is NemotronNASForCausalLM
        assert lm_registration.config is NemotronNASConfig

    def test_causal_lm(self, nemotron_nas_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="nemotron-nas",
            model_cls=NemotronNASForCausalLM,
            config=nemotron_nas_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"NemotronNAS CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, nemotron_nas_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="nemotron-nas",
            model_cls=NemotronNASForCausalLM,
            config=nemotron_nas_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"NemotronNAS generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, nemotron_nas_config):
        model = NemotronNASForCausalLM(nemotron_nas_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
