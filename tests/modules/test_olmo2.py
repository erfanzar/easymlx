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

"""Tests for OLMo2 model."""

import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.olmo2 import Olmo2Config, Olmo2ForCausalLM, Olmo2Model

from .test_utils import CausalLMTester


class TestOlmo2:
    """Test suite for OLMo2 model."""

    @pytest.fixture
    def olmo2_config(self, small_model_config):
        """Create a tiny OLMo2 config."""
        return Olmo2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            tie_word_embeddings=True,
        )

    def test_registry(self):
        """OLMo2 should register under the exact HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "olmo2")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "olmo2")

        assert base_registration.module is Olmo2Model
        assert base_registration.config is Olmo2Config
        assert lm_registration.module is Olmo2ForCausalLM
        assert lm_registration.config is Olmo2Config

    def test_causal_lm(self, olmo2_config, small_model_config):
        """Test OLMo2 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="olmo2",
            model_cls=Olmo2ForCausalLM,
            config=olmo2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OLMo2 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, olmo2_config, small_model_config):
        """Test OLMo2 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="olmo2",
            model_cls=Olmo2ForCausalLM,
            config=olmo2_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"OLMo2 generation failed: {result.error_message}"

    def test_has_qk_norm(self, olmo2_config):
        """OLMo2 attention layers should have q_norm and k_norm."""
        model = Olmo2Model(olmo2_config)
        for layer in model.layers:
            assert hasattr(layer.self_attn, "q_norm")
            assert hasattr(layer.self_attn, "k_norm")

    def test_has_post_feedforward_layernorm(self, olmo2_config):
        """OLMo2 decoder layers should have post_feedforward_layernorm."""
        model = Olmo2Model(olmo2_config)
        for layer in model.layers:
            assert hasattr(layer, "post_feedforward_layernorm")

    def test_sanitize_removes_rotary_inv_freq(self, olmo2_config):
        """Sanitize should remove rotary_emb.inv_freq keys."""
        model = Olmo2ForCausalLM(olmo2_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["self_attn.rotary_emb.inv_freq"] = local_weights["model.embed_tokens.weight"]

        sanitized = model.sanitize(upstream_weights)

        assert "self_attn.rotary_emb.inv_freq" not in sanitized
