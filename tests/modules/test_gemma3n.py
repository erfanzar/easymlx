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

"""Tests for Gemma3N model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.gemma3n import Gemma3NConfig, Gemma3NForCausalLM, Gemma3NModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestGemma3N:
    """Test suite for Gemma3N model."""

    @pytest.fixture
    def gemma3n_config(self, small_model_config):
        return Gemma3NConfig(
            text_config={
                "vocab_size": small_model_config["vocab_size"],
                "hidden_size": small_model_config["hidden_size"],
                "intermediate_size": small_model_config["intermediate_size"],
                "num_hidden_layers": small_model_config["num_hidden_layers"],
                "num_attention_heads": small_model_config["num_attention_heads"],
                "num_key_value_heads": small_model_config["num_key_value_heads"],
                "head_dim": small_model_config["head_dim"],
                "rms_norm_eps": 1e-6,
                "sliding_window": 8,
                "max_position_embeddings": small_model_config["max_position_embeddings"],
                "layer_types": ["sliding_attention", "full_attention"],
                "num_kv_shared_layers": 0,
                "vocab_size_per_layer_input": small_model_config["vocab_size"],
                "hidden_size_per_layer_input": 32,
                "altup_num_inputs": 2,
                "altup_coef_clip": 1.0,
                "altup_correct_scale": False,
                "altup_active_idx": 0,
                "laurel_rank": 16,
                "activation_sparsity_pattern": [0.0, 0.0],
            },
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "gemma3n")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "gemma3n")

        assert base_registration.module is Gemma3NModel
        assert base_registration.config is Gemma3NConfig
        assert lm_registration.module is Gemma3NForCausalLM
        assert lm_registration.config is Gemma3NConfig

    def test_causal_lm(self, gemma3n_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma3n",
            model_cls=Gemma3NForCausalLM,
            config=gemma3n_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma3N CAUSAL_LM failed: {result.error_message}"

    def test_sanitize_drops_vision_audio_keys(self, gemma3n_config):
        model = Gemma3NForCausalLM(gemma3n_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))
        import mlx.core as mx

        upstream_weights = dict(local_weights)
        upstream_weights["vision_tower.weight"] = mx.ones((2, 2))
        upstream_weights["audio_tower.weight"] = mx.ones((2, 2))
        upstream_weights["embed_vision.weight"] = mx.ones((2, 2))
        upstream_weights["embed_audio.weight"] = mx.ones((2, 2))

        sanitized = model.sanitize(upstream_weights)

        assert "vision_tower.weight" not in sanitized
        assert "audio_tower.weight" not in sanitized
        assert "embed_vision.weight" not in sanitized
        assert "embed_audio.weight" not in sanitized
