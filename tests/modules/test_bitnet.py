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

"""Tests for BitNet model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.bitnet import BitNetConfig, BitNetForCausalLM, BitNetModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestBitNet:
    """Test suite for BitNet model."""

    @pytest.fixture
    def bitnet_config(self, small_model_config):
        return BitNetConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_reg = registry.get_module_registration(TaskType.BASE_MODULE, "bitnet")
        lm_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "bitnet")
        assert base_reg.module is BitNetModel
        assert base_reg.config is BitNetConfig
        assert lm_reg.module is BitNetForCausalLM
        assert lm_reg.config is BitNetConfig

    def test_causal_lm(self, bitnet_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="bitnet",
            model_cls=BitNetForCausalLM,
            config=bitnet_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"BitNet CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, bitnet_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="bitnet",
            model_cls=BitNetForCausalLM,
            config=bitnet_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"BitNet generation failed: {result.error_message}"

    def test_sanitize(self, bitnet_config):
        model = BitNetForCausalLM(bitnet_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]
        sanitized = model.sanitize(upstream_weights)
        assert "rotary_emb.inv_freq" not in sanitized
