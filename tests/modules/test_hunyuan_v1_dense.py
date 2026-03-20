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

"""Tests for Hunyuan V1 Dense model."""

import pytest

import mlx.core as mx
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.hunyuan_v1_dense import (
    HunyuanV1DenseConfig,
    HunyuanV1DenseForCausalLM,
    HunyuanV1DenseModel,
)

from .test_utils import CausalLMTester


class TestHunyuanV1Dense:
    """Test suite for Hunyuan V1 Dense model."""

    @pytest.fixture
    def hunyuan_v1_dense_config(self, small_model_config):
        return HunyuanV1DenseConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            use_qk_norm=True,
            rope_scaling={"alpha": 1.0, "factor": 1.0, "type": "dynamic"},
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_reg = registry.get_module_registration(TaskType.BASE_MODULE, "hunyuan_v1_dense")
        lm_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "hunyuan_v1_dense")
        assert base_reg.module is HunyuanV1DenseModel
        assert base_reg.config is HunyuanV1DenseConfig
        assert lm_reg.module is HunyuanV1DenseForCausalLM
        assert lm_reg.config is HunyuanV1DenseConfig

    def test_causal_lm(self, hunyuan_v1_dense_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="hunyuan_v1_dense",
            model_cls=HunyuanV1DenseForCausalLM,
            config=hunyuan_v1_dense_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"HunyuanV1Dense CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, hunyuan_v1_dense_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="hunyuan_v1_dense",
            model_cls=HunyuanV1DenseForCausalLM,
            config=hunyuan_v1_dense_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"HunyuanV1Dense generation failed: {result.error_message}"

    def test_sanitize(self, hunyuan_v1_dense_config):
        model = HunyuanV1DenseForCausalLM(hunyuan_v1_dense_config)
        local_weights = dict(tree_flatten(model.parameters()))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.ones((8,))
        sanitized = model.sanitize(upstream_weights)
        assert "rotary_emb.inv_freq" not in sanitized
