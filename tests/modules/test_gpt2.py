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

"""Tests for GPT-2 model."""

import mlx.core as mx
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gpt2 import GPT2Config, GPT2ForCausalLM, GPT2Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestGPT2:
    """Test suite for GPT-2 model."""

    @pytest.fixture
    def gpt2_config(self, small_model_config):
        return GPT2Config(
            n_embd=small_model_config["hidden_size"],
            n_head=small_model_config["num_attention_heads"],
            n_layer=small_model_config["num_hidden_layers"],
            n_ctx=small_model_config["max_position_embeddings"],
            n_positions=small_model_config["max_position_embeddings"],
            vocab_size=small_model_config["vocab_size"],
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gpt2")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gpt2")

        assert base_registration.module is GPT2Model
        assert base_registration.config is GPT2Config
        assert lm_registration.module is GPT2ForCausalLM
        assert lm_registration.config is GPT2Config

    def test_causal_lm(self, gpt2_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt2",
            model_cls=GPT2ForCausalLM,
            config=gpt2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT2 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, gpt2_config, small_model_config):
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gpt2",
            model_cls=GPT2ForCausalLM,
            config=gpt2_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GPT2 generation failed: {result.error_message}"

    def test_sanitize(self, gpt2_config):
        model = GPT2ForCausalLM(gpt2_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        upstream_weights: dict[str, mx.array] = {}
        for key, value in local_weights.items():
            raw_key = key.removeprefix("model.")
            if raw_key.endswith(
                (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")
            ):
                upstream_weights[raw_key] = value.transpose(1, 0)
            else:
                upstream_weights[raw_key] = value

        upstream_weights["h.0.attn.bias"] = mx.zeros((1,))
        upstream_weights["h.0.attn.masked_bias"] = mx.zeros((1,))

        sanitized = model.sanitize(upstream_weights)

        assert "h.0.attn.bias" not in sanitized
        assert "h.0.attn.masked_bias" not in sanitized
        assert set(sanitized) == set(local_weights)
