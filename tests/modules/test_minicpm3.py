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

"""Tests for MiniCPM3 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.minicpm3 import MiniCPM3Config, MiniCPM3ForCausalLM, MiniCPM3Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestMiniCPM3:
    """Test suite for MiniCPM3 model."""

    @pytest.fixture
    def minicpm3_config(self, small_model_config):
        return MiniCPM3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            q_lora_rank=32,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            kv_lora_rank=16,
            scale_depth=1.4,
            scale_emb=12.0,
            dim_model_base=256,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        base_reg = registry.get_module_registration(TaskType.BASE_MODULE, "minicpm3")
        lm_reg = registry.get_module_registration(TaskType.CAUSAL_LM, "minicpm3")
        assert base_reg.module is MiniCPM3Model
        assert base_reg.config is MiniCPM3Config
        assert lm_reg.module is MiniCPM3ForCausalLM
        assert lm_reg.config is MiniCPM3Config

    def test_causal_lm(self, minicpm3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="minicpm3",
            model_cls=MiniCPM3ForCausalLM,
            config=minicpm3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MiniCPM3 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, minicpm3_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="minicpm3",
            model_cls=MiniCPM3ForCausalLM,
            config=minicpm3_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"MiniCPM3 generation failed: {result.error_message}"

    def test_sanitize_creates_lm_head(self, minicpm3_config):
        model = MiniCPM3ForCausalLM(minicpm3_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights.pop("lm_head.weight", None)
        upstream_weights["rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
        assert "lm_head.weight" in sanitized
