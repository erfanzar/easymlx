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

"""Tests for GPT-NeoX model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestGPTNeoX:
    """Test suite for GPT-NeoX model."""

    @pytest.fixture
    def gpt_neox_config(self, small_model_config):
        """Create a tiny GPT-NeoX config."""
        return GPTNeoXConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            rotary_pct=0.25,
            rotary_emb_base=10000,
            use_parallel_residual=True,
            tie_word_embeddings=False,
        )

    def test_registry(self):
        """GPT-NeoX should register under the exact HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gpt_neox")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gpt_neox")

        assert base_registration.module is GPTNeoXModel
        assert base_registration.config is GPTNeoXConfig
        assert lm_registration.module is GPTNeoXForCausalLM
        assert lm_registration.config is GPTNeoXConfig

    def test_causal_lm(self, gpt_neox_config, small_model_config):
        """Test GPT-NeoX causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt_neox",
            model_cls=GPTNeoXForCausalLM,
            config=gpt_neox_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT-NeoX CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, gpt_neox_config, small_model_config):
        """Test GPT-NeoX generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gpt_neox",
            model_cls=GPTNeoXForCausalLM,
            config=gpt_neox_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"GPT-NeoX generation failed: {result.error_message}"

    def test_sanitize_removes_rotary_inv_freq(self, gpt_neox_config):
        """Sanitize should remove rotary_emb.inv_freq and attention bias keys."""
        model = GPTNeoXModel(gpt_neox_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["layers.0.self_attn.attention.rotary_emb.inv_freq"] = local_weights["embed_in.weight"]
        upstream_weights["layers.0.self_attn.attention.bias"] = local_weights["embed_in.weight"]
        upstream_weights["layers.0.self_attn.attention.masked_bias"] = local_weights["embed_in.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "layers.0.self_attn.attention.rotary_emb.inv_freq" not in sanitized
        assert "layers.0.self_attn.attention.bias" not in sanitized
        assert "layers.0.self_attn.attention.masked_bias" not in sanitized

    def test_sequential_residual(self, small_model_config):
        """Sequential residual mode should also work."""
        config = GPTNeoXConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            rotary_pct=0.25,
            rotary_emb_base=10000,
            use_parallel_residual=False,
        )
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt_neox",
            model_cls=GPTNeoXForCausalLM,
            config=config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT-NeoX sequential residual failed: {result.error_message}"
