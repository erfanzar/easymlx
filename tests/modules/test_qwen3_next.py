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

"""Tests for Qwen3Next model."""

import mlx.core as mx
import numpy as np
import pytest

from easymlx.modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from easymlx.operations.kernels.gated_delta_rule import _gdr_recurrent_step_metal, _l2_normalize

from .test_utils import CausalLMTester


class TestQwen3Next:
    """Test suite for Qwen3Next model."""

    @pytest.fixture
    def qwen3_next_config(self, small_model_config):
        """Create Qwen3Next-specific config."""
        return Qwen3NextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            num_experts=small_model_config["num_experts"],
            num_experts_per_tok=small_model_config["num_experts_per_tok"],
            linear_num_key_heads=small_model_config["num_key_value_heads"],
            linear_num_value_heads=small_model_config["num_attention_heads"],
            linear_key_head_dim=small_model_config["head_dim"],
            linear_value_head_dim=small_model_config["head_dim"],
        )

    def test_causal_lm(self, qwen3_next_config, small_model_config):
        """Test Qwen3NextForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_next",
            model_cls=Qwen3NextForCausalLM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3Next CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, qwen3_next_config, small_model_config):
        """Test Qwen3Next text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_next",
            model_cls=Qwen3NextForCausalLM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
            max_new_tokens=8,
        )
        assert result.success, f"Qwen3Next generation failed: {result.error_message}"

    def test_gdr_recurrent_metal_matches_math(self):
        """The Metal recurrent GDR step should match the reference math."""
        rng = np.random.default_rng(0)
        batch = 2
        num_heads = 3
        head_dim = 8
        d_state = 8

        query = mx.array(rng.standard_normal((batch, num_heads, head_dim)).astype(np.float32))
        key = mx.array(rng.standard_normal((batch, num_heads, head_dim)).astype(np.float32))
        value = mx.array(rng.standard_normal((batch, num_heads, d_state)).astype(np.float32))
        beta = mx.array(rng.standard_normal((batch, num_heads)).astype(np.float32))
        decay = mx.array(rng.standard_normal((batch, num_heads)).astype(np.float32))
        recurrent_state = mx.array(rng.standard_normal((batch, num_heads, head_dim, d_state)).astype(np.float32))

        query_n = _l2_normalize(query.astype(mx.float32)) * (head_dim**-0.5)
        key_n = _l2_normalize(key.astype(mx.float32))
        decay_scale = mx.exp(decay.astype(mx.float32))

        state_scaled = recurrent_state.astype(mx.float32) * decay_scale[:, :, None, None]
        kv_dot = mx.sum(state_scaled * key_n[:, :, :, None], axis=-2)
        delta = (value.astype(mx.float32) - kv_dot) * beta.astype(mx.float32)[:, :, None]
        expected_state = state_scaled + key_n[:, :, :, None] * delta[:, :, None, :]
        expected_output = mx.sum(expected_state * query_n[:, :, :, None], axis=-2)[:, None, :, :]

        try:
            actual_output, actual_state = _gdr_recurrent_step_metal(
                query=query_n,
                key=key_n,
                value=value.astype(mx.float32),
                beta=beta.astype(mx.float32),
                decay_scale=decay_scale,
                recurrent_state=recurrent_state.astype(mx.float32),
            )
        except RuntimeError as exc:
            pytest.skip(f"Metal recurrent GDR kernel unavailable: {exc}")

        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(actual_state), np.asarray(expected_state), atol=1e-5, rtol=1e-5)
