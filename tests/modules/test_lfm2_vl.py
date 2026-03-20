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

"""Tests for LFM2-VL model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.lfm2_vl import Lfm2VlConfig, Lfm2VlForCausalLM, Lfm2VlModel

from .test_utils import CausalLMTester


class TestLfm2Vl:
    """Test suite for LFM2-VL model."""

    @pytest.fixture
    def lfm2_vl_config(self, small_model_config):
        return Lfm2VlConfig(
            text_config={
                "vocab_size": small_model_config["vocab_size"],
                "hidden_size": small_model_config["hidden_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "max_position_embeddings": small_model_config["max_position_embeddings"],
                "block_dim": small_model_config["hidden_size"],
                "block_ff_dim": small_model_config["intermediate_size"],
                "block_multiple_of": 64,
                "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": False,
                "conv_L_cache": 4,
                "conv_bias": True,
                "full_attn_idxs": [0],
                "layer_types": ["full_attention", "short_conv"],
                "tie_word_embeddings": False,
            },
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "lfm2-vl")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "lfm2-vl")

        assert base_registration.module is Lfm2VlModel
        assert base_registration.config is Lfm2VlConfig
        assert lm_registration.module is Lfm2VlForCausalLM
        assert lm_registration.config is Lfm2VlConfig

    def test_causal_lm(self, lfm2_vl_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="lfm2_vl",
            model_cls=Lfm2VlForCausalLM,
            config=lfm2_vl_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"LFM2-VL CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, lfm2_vl_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="lfm2_vl",
            model_cls=Lfm2VlForCausalLM,
            config=lfm2_vl_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"LFM2-VL generation failed: {result.error_message}"

    def test_sanitize_strips_vision_weights(self, lfm2_vl_config):
        model = Lfm2VlForCausalLM(lfm2_vl_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["vision_tower.weight"] = local_weights["model.language_model.embed_tokens.weight"]
        upstream_weights["multi_modal_projector.weight"] = local_weights["model.language_model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.weight" not in sanitized
        assert "multi_modal_projector.weight" not in sanitized
        assert set(sanitized) == set(local_weights)
