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

"""Tests for Falcon-H1 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.falcon_h1 import FalconH1Config, FalconH1ForCausalLM, FalconH1Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestFalconH1:
    """Test suite for Falcon-H1 model."""

    @pytest.fixture
    def falcon_h1_config(self):
        return FalconH1Config(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            head_dim=16,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_n_groups=1,
            mamba_n_heads=4,
            mamba_d_ssm=64,
            mamba_d_head=16,
            mamba_chunk_size=16,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            embedding_multiplier=1.0,
            lm_head_multiplier=1.0,
            attention_in_multiplier=1.0,
            attention_out_multiplier=1.0,
            ssm_in_multiplier=1.0,
            ssm_out_multiplier=1.0,
            mlp_multipliers=[1.0, 1.0],
            ssm_multipliers=[1.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "falcon_h1")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "falcon_h1")

        assert base_registration.module is FalconH1Model
        assert base_registration.config is FalconH1Config
        assert lm_registration.module is FalconH1ForCausalLM
        assert lm_registration.config is FalconH1Config

    def test_causal_lm(self, falcon_h1_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="falcon_h1",
            model_cls=FalconH1ForCausalLM,
            config=falcon_h1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"FalconH1 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, falcon_h1_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="falcon_h1",
            model_cls=FalconH1ForCausalLM,
            config=falcon_h1_config,
            small_model_config=small_model_config,
            max_new_tokens=4,
        )
        assert result.success, f"FalconH1 generation failed: {result.error_message}"

    def test_sanitize(self, falcon_h1_config):
        model = FalconH1ForCausalLM(falcon_h1_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
