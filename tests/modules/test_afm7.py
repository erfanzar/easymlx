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

"""Tests for AFM7 model."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from easymlx.infra.factory import TaskType, registry
from easymlx.modules.afm7 import Afm7Config, Afm7ForCausalLM, Afm7Model

from .test_utils import CausalLMTester


class TestAfm7:
    """Test suite for AFM7 model."""

    @pytest.fixture
    def afm7_config(self, small_model_config):
        """Create a tiny AFM7 config."""
        return Afm7Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_dim=small_model_config["hidden_size"],
            num_layers=small_model_config["num_hidden_layers"],
            num_kv_reuse_layers=1,
            num_heads=small_model_config["num_attention_heads"],
            num_kv_heads=small_model_config["num_key_value_heads"],
            hidden_dim_scale_factor=2.0,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        """AFM7 should register under the expected HF model type."""
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "afm7")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "afm7")

        assert base_registration.module is Afm7Model
        assert base_registration.config is Afm7Config
        assert lm_registration.module is Afm7ForCausalLM
        assert lm_registration.config is Afm7Config

    def test_causal_lm(self, afm7_config, small_model_config):
        """Test AFM7 causal LM forward pass."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="afm7",
            model_cls=Afm7ForCausalLM,
            config=afm7_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"AFM7 CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, afm7_config, small_model_config):
        """Test AFM7 generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="afm7",
            model_cls=Afm7ForCausalLM,
            config=afm7_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"AFM7 generation failed: {result.error_message}"

    def test_sanitize(self, afm7_config):
        """Ensure AFM7 sanitize removes rotary embedding keys."""
        model = Afm7ForCausalLM(afm7_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        upstream_weights = dict(local_weights)
        upstream_weights["rotary_emb.inv_freq"] = mx.zeros((16,))

        sanitized = model.sanitize(upstream_weights)

        assert "rotary_emb.inv_freq" not in sanitized
