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

"""Tests for PLaMo2 model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.plamo2 import Plamo2Config, Plamo2ForCausalLM, Plamo2Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestPlamo2:
    """Test suite for PLaMo2 model."""

    @pytest.fixture
    def plamo2_config(self, small_model_config):
        return Plamo2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            hidden_size_per_head=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            mamba_enabled=False,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_num_heads=2,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "plamo2")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "plamo2")

        assert base_registration.module is Plamo2Model
        assert base_registration.config is Plamo2Config
        assert lm_registration.module is Plamo2ForCausalLM
        assert lm_registration.config is Plamo2Config

    def test_causal_lm(self, plamo2_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="plamo2",
            model_cls=Plamo2ForCausalLM,
            config=plamo2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"PLaMo2 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, plamo2_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="plamo2",
            model_cls=Plamo2ForCausalLM,
            config=plamo2_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"PLaMo2 generation failed: {result.error_message}"

    def test_sanitize_conv1d_weights(self, plamo2_config):
        model = Plamo2ForCausalLM(plamo2_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        fake_conv_key = "model.layers.0.mixer.conv1d.weight"
        upstream_weights = dict(local_weights)
        upstream_weights[fake_conv_key] = mx.ones((4, 8, 3))

        sanitized = model.sanitize(upstream_weights)

        if fake_conv_key in sanitized:
            assert sanitized[fake_conv_key].shape == (4, 3, 8)

    def test_sanitize_matches_local_parameter_tree(self, plamo2_config):
        model = Plamo2ForCausalLM(plamo2_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert set(sanitized) == set(local_weights)
