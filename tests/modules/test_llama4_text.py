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

"""Tests for the Llama4 text-only model family."""

import pytest
from easymlx.modules.llama4_text import Llama4TextConfig, Llama4TextForCausalLM
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestLlama4Text:
    """Test suite for the Llama4 text-only model."""

    @pytest.fixture
    def llama4_text_config(self, small_model_config):
        """Create a tiny Llama4 text config."""
        config = Llama4TextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            intermediate_size_mlp=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["head_dim"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_local_experts=small_model_config["num_local_experts"],
            num_experts_per_tok=1,
            interleave_moe_layer_step=2,
            no_rope_layer_interval=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return config

    def test_config_model_type_and_defaults(self, llama4_text_config):
        """The text-only config should finalize under the llama4_text model type."""
        assert llama4_text_config.model_type == "llama4_text"
        assert llama4_text_config.no_rope_layers == [1, 0]
        assert llama4_text_config.layer_types == ["chunked_attention", "full_attention"]

    def test_causal_lm(self, llama4_text_config, small_model_config):
        """Test Llama4TextForCausalLM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="llama4_text",
            model_cls=Llama4TextForCausalLM,
            config=llama4_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama4 text forward failed: {result.error_message}"

    def test_generation(self, llama4_text_config, small_model_config):
        """Test Llama4 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama4_text",
            model_cls=Llama4TextForCausalLM,
            config=llama4_text_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Llama4 text generation failed: {result.error_message}"

    def test_sanitize_maps_upstream_checkpoint_keys(self, llama4_text_config):
        model = Llama4TextForCausalLM(llama4_text_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = {}
        for key, value in local_weights.items():
            if key == "lm_head.weight":
                upstream_weights["output.weight"] = value
            else:
                upstream_weights[key.replace("model.language_model.", "model.", 1)] = value

        sanitized = model.sanitize(upstream_weights)

        assert set(sanitized) == set(local_weights)
