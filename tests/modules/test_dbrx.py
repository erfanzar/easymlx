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

"""Tests for DBRX model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.dbrx import DBRXConfig, DBRXForCausalLM, DBRXModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestDBRX:
    """Test suite for DBRX model."""

    @pytest.fixture
    def dbrx_config(self, small_model_config):
        return DBRXConfig(
            vocab_size=small_model_config["vocab_size"],
            d_model=small_model_config["hidden_size"],
            n_layers=small_model_config["num_hidden_layers"],
            n_heads=small_model_config["num_attention_heads"],
            ffn_config={
                "ffn_hidden_size": small_model_config["intermediate_size"],
                "moe_num_experts": 4,
                "moe_top_k": 2,
                "moe_jitter_eps": 0.0,
            },
            attn_config={
                "kv_n_heads": small_model_config["num_key_value_heads"],
                "clip_qkv": 8.0,
                "rope_theta": 500000.0,
            },
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "dbrx")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "dbrx")

        assert base_registration.module is DBRXModel
        assert base_registration.config is DBRXConfig
        assert lm_registration.module is DBRXForCausalLM
        assert lm_registration.config is DBRXConfig

    def test_causal_lm(self, dbrx_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="dbrx",
            model_cls=DBRXForCausalLM,
            config=dbrx_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DBRX CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, dbrx_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="dbrx",
            model_cls=DBRXForCausalLM,
            config=dbrx_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"DBRX generation failed: {result.error_message}"

    def test_sanitize_matches_local_parameter_tree(self, dbrx_config):
        model = DBRXForCausalLM(dbrx_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
