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
from easymlx.caching import PageMetadata
from easymlx.modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from easymlx.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextLinearAttention,
    Qwen3NextModel,
    _decode_depthwise_conv_state_update,
)
from easymlx.operations.kernels.gated_delta_rule import (
    _gdr_chunked_forward_metal,
    _gdr_chunked_forward_metal_grouped,
    _gdr_recurrent_step_metal,
    _gdr_recurrent_step_metal_fused_logdecay,
    _l2_normalize,
)

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

    def test_model_casts_embeddings_to_configured_activation_dtype(self):
        """Embedding output dtype should follow the config/runtime dtype."""
        config = Qwen3NextConfig(
            vocab_size=8,
            hidden_size=4,
            num_hidden_layers=0,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            head_dim=4,
            dtype="float16",
            cache_dtype="float16",
        )
        model = Qwen3NextModel(config)

        class _BfloatEmbedding:
            def __call__(self, input_ids):
                return mx.zeros((*input_ids.shape, config.hidden_size), dtype=mx.bfloat16)

        class _IdentityNorm:
            def __call__(self, hidden_states):
                return hidden_states

        model.embed_tokens = _BfloatEmbedding()
        model.norm = _IdentityNorm()

        output = model(mx.array([[1, 2]], dtype=mx.int32))
        mx.eval(output)

        assert output.dtype == mx.float16

    def test_linear_attention_mxfp4_quantized_decode_uses_floating_activations(self):
        """Quantized Qwen3-Next linear attention should keep activations in a real floating dtype."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(attn, name, getattr(attn, name).to_quantized(group_size=32, bits=4, mode="mxfp4"))

        attn.reset_state(batch_size=1)
        assert mx.issubdtype(attn._conv_state.dtype, mx.floating)

        hidden_states = mx.array(
            np.random.default_rng(42).standard_normal((1, 1, config.hidden_size)).astype(np.float16)
        )
        output = attn(hidden_states)
        mx.eval(output)

        assert mx.issubdtype(output.dtype, mx.floating)
        assert output.shape == hidden_states.shape

    def test_linear_attention_decode_cache_keeps_native_param_dtypes(self):
        """Decode cache should not widen learned tensors to fp32 outside recurrent math."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        attn.in_proj_qkv.weight = attn.in_proj_qkv.weight.astype(mx.float16)
        attn.in_proj_a.weight = attn.in_proj_a.weight.astype(mx.float16)
        attn.dt_bias = attn.dt_bias.astype(mx.float16)
        attn.A_log = attn.A_log.astype(mx.float16)
        if attn.conv1d is not None:
            attn.conv1d.weight = attn.conv1d.weight.astype(mx.float16)

        attn.reset_state(batch_size=1)

        assert attn._conv_state is not None
        assert attn._decode_dt_bias is not None
        assert attn._decode_conv_kernel is not None
        assert attn._decode_decay_base is not None
        assert attn._conv_state.dtype == mx.float16
        assert attn._decode_dt_bias.dtype == mx.float16
        assert attn._decode_conv_kernel.dtype == mx.float16
        assert attn._decode_decay_base.dtype == mx.float32

        hidden_states = mx.array(np.random.default_rng(0).standard_normal((1, 1, config.hidden_size)).astype(np.float16))
        output = attn(hidden_states)
        mx.eval(output, attn._conv_state)

        assert attn._conv_state.dtype == mx.float16
        assert mx.issubdtype(output.dtype, mx.floating)
        assert output.shape == hidden_states.shape

    def test_linear_attention_decode_step_with_state_matches_mutating_decode(self):
        """Explicit decode state should match the regular mutating decode path."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        attn.reset_state(batch_size=1)
        state = attn.get_decode_state()
        hidden_states = mx.array(
            np.random.default_rng(123).standard_normal((1, 1, config.hidden_size)).astype(np.float32)
        )

        stateful_output, new_state = attn.decode_step_with_state(hidden_states, decode_state=state)
        eager_output = attn(hidden_states)
        mx.eval(stateful_output, eager_output, new_state["conv_state"], new_state["recurrent_state"])

        np.testing.assert_allclose(np.asarray(stateful_output), np.asarray(eager_output), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            np.asarray(new_state["conv_state"]), np.asarray(attn._conv_state), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.asarray(new_state["recurrent_state"]),
            np.asarray(attn._recurrent_state),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_linear_attention_stateful_multi_token_matches_step_decode(self):
        """Stateful multi-token linear attention should use the vectorized path."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        attn.reset_state(batch_size=1)
        initial_state = attn.get_decode_state()
        hidden_states = mx.array(
            np.random.default_rng(321).standard_normal((1, 3, config.hidden_size)).astype(np.float32)
        )

        loop_state = initial_state
        loop_outputs = []
        for idx in range(hidden_states.shape[1]):
            output_t, loop_state = attn.decode_step_with_state(
                hidden_states[:, idx : idx + 1],
                decode_state=loop_state,
            )
            loop_outputs.append(output_t)
        expected_output = mx.concatenate(loop_outputs, axis=1)

        block_output, block_state = attn.decode_step_with_state(
            hidden_states,
            decode_state=initial_state,
        )
        attn.set_decode_state(initial_state)
        actual_output = attn(hidden_states)
        mx.eval(
            expected_output,
            block_output,
            actual_output,
            loop_state["conv_state"],
            loop_state["recurrent_state"],
            block_state["conv_state"],
            block_state["recurrent_state"],
            attn._conv_state,
            attn._recurrent_state,
        )

        np.testing.assert_allclose(np.asarray(block_output), np.asarray(expected_output), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            np.asarray(block_state["conv_state"]),
            np.asarray(loop_state["conv_state"]),
            rtol=2e-3,
            atol=2e-3,
        )
        np.testing.assert_allclose(
            np.asarray(block_state["recurrent_state"]),
            np.asarray(loop_state["recurrent_state"]),
            rtol=2e-3,
            atol=2e-3,
        )
        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            np.asarray(attn._conv_state), np.asarray(loop_state["conv_state"]), rtol=2e-3, atol=2e-3
        )
        np.testing.assert_allclose(
            np.asarray(attn._recurrent_state),
            np.asarray(loop_state["recurrent_state"]),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_linear_attention_ragged_single_token_decode_uses_slots(self):
        """Ragged eSurge decode should batch active slots without mixing states."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        attn.reset_state(batch_size=3)
        hidden_states = mx.array(np.random.default_rng(777).standard_normal((2, config.hidden_size)).astype(np.float32))
        metadata = PageMetadata(
            query_start_loc=mx.array([0, 1, 2], dtype=mx.int32),
            is_single_token_decode=True,
            slot_ids=(0, 2),
        )

        initial_conv_state = attn._conv_state
        initial_recurrent_state = attn._recurrent_state
        assert initial_conv_state is not None
        assert initial_recurrent_state is not None

        expected_outputs = []
        expected_conv_state = initial_conv_state
        expected_recurrent_state = initial_recurrent_state
        for row, slot in enumerate(metadata.slot_ids):
            slot_idx = mx.array([slot], dtype=mx.int32)
            state = {
                "conv_state": mx.take(initial_conv_state, slot_idx, axis=0),
                "recurrent_state": mx.take(initial_recurrent_state, slot_idx, axis=0),
            }
            out, new_state = attn.decode_step_with_state(hidden_states[row : row + 1], decode_state=state)
            expected_outputs.append(out[0])
            conv_indices = slot_idx.reshape(1, 1, 1)
            recurrent_indices = slot_idx.reshape(1, 1, 1, 1)
            expected_conv_state = mx.put_along_axis(
                expected_conv_state,
                conv_indices,
                new_state["conv_state"],
                axis=0,
            )
            expected_recurrent_state = mx.put_along_axis(
                expected_recurrent_state,
                recurrent_indices,
                new_state["recurrent_state"],
                axis=0,
            )
        expected_output = mx.stack(expected_outputs, axis=0)

        attn._conv_state = initial_conv_state
        attn._recurrent_state = initial_recurrent_state
        actual_output = attn(hidden_states, cache_metadata=metadata)
        mx.eval(actual_output, expected_output, attn._conv_state, attn._recurrent_state)

        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.asarray(attn._conv_state), np.asarray(expected_conv_state), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(
            np.asarray(attn._recurrent_state),
            np.asarray(expected_recurrent_state),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_model_decode_step_with_state_accepts_multi_token_blocks(self):
        """Model-level explicit decode state should support block verification."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention"],
            mlp_only_layers=[0],
        )
        model = Qwen3NextModel(config)
        input_ids = mx.array([[3, 4, 5]], dtype=mx.int32)
        initial_state = model.get_decode_state()

        loop_state = initial_state
        loop_outputs = []
        for idx in range(input_ids.shape[1]):
            output_t, loop_state = model.decode_step_with_state(
                input_ids[:, idx : idx + 1],
                decode_state=loop_state,
            )
            loop_outputs.append(output_t)
        expected = mx.concatenate(loop_outputs, axis=1)

        actual, block_state = model.decode_step_with_state(input_ids, decode_state=initial_state)
        mx.eval(
            expected,
            actual,
            loop_state["linear_layers"][0]["conv_state"],
            loop_state["linear_layers"][0]["recurrent_state"],
            block_state["linear_layers"][0]["conv_state"],
            block_state["linear_layers"][0]["recurrent_state"],
        )

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(
            np.asarray(block_state["linear_layers"][0]["conv_state"]),
            np.asarray(loop_state["linear_layers"][0]["conv_state"]),
            rtol=2e-3,
            atol=2e-3,
        )
        np.testing.assert_allclose(
            np.asarray(block_state["linear_layers"][0]["recurrent_state"]),
            np.asarray(loop_state["linear_layers"][0]["recurrent_state"]),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_linear_attention_decode_uses_step_decode(self):
        """Single-token decode should bypass the generic GDR wrapper path."""
        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        )
        attn = Qwen3NextLinearAttention(config)
        attn.reset_state(batch_size=1)

        class _DummyDecodeOp:
            def __init__(self):
                self.called = False

            def step_decode(self, *, query, key, value, beta, decay=None, recurrent_state=None, **_kwargs):
                self.called = True
                batch_size, seq_len, num_heads, head_dim = query.shape
                value_dim = value.shape[-1]
                next_state = mx.ones((batch_size, num_heads, head_dim, value_dim), dtype=mx.float32)
                return mx.zeros((batch_size, seq_len, num_heads, value_dim), dtype=mx.float32), next_state

            def __call__(self, *args, **kwargs):
                raise AssertionError("decode path should use step_decode")

        dummy_op = _DummyDecodeOp()
        attn.gdr_op = dummy_op

        hidden_states = mx.array(
            np.random.default_rng(42).standard_normal((1, 1, config.hidden_size)).astype(np.float16)
        )
        output = attn(hidden_states)
        mx.eval(output, attn._recurrent_state)

        assert dummy_op.called is True
        assert output.shape == hidden_states.shape
        np.testing.assert_allclose(
            np.asarray(attn._recurrent_state),
            np.ones((1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)),
        )

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

    def test_gdr_recurrent_fused_logdecay_metal_matches_math(self):
        """The fused recurrent GDR kernel should match the reference math."""
        rng = np.random.default_rng(7)
        batch = 2
        num_heads = 3
        head_dim = 8
        d_state = 8

        query = mx.array(rng.standard_normal((batch, num_heads, head_dim)).astype(np.float32))
        key = mx.array(rng.standard_normal((batch, num_heads, head_dim)).astype(np.float32))
        value = mx.array(rng.standard_normal((batch, num_heads, d_state)).astype(np.float32))
        beta = mx.array(rng.standard_normal((batch, num_heads)).astype(np.float32))
        log_decay = mx.array(rng.standard_normal((batch, num_heads)).astype(np.float32))
        recurrent_state = mx.array(rng.standard_normal((batch, num_heads, head_dim, d_state)).astype(np.float32))

        query_scale = head_dim**-0.5
        query_n = _l2_normalize(query.astype(mx.float32)) * query_scale
        key_n = _l2_normalize(key.astype(mx.float32))
        decay_scale = mx.exp(log_decay.astype(mx.float32))

        state_scaled = recurrent_state.astype(mx.float32) * decay_scale[:, :, None, None]
        kv_dot = mx.sum(state_scaled * key_n[:, :, :, None], axis=-2)
        delta = (value.astype(mx.float32) - kv_dot) * beta.astype(mx.float32)[:, :, None]
        expected_state = state_scaled + key_n[:, :, :, None] * delta[:, :, None, :]
        expected_output = mx.sum(expected_state * query_n[:, :, :, None], axis=-2)[:, None, :, :]

        try:
            actual_output, actual_state = _gdr_recurrent_step_metal_fused_logdecay(
                query=query.astype(mx.float32),
                key=key.astype(mx.float32),
                value=value.astype(mx.float32),
                beta=beta.astype(mx.float32),
                log_decay=log_decay.astype(mx.float32),
                query_scale=query_scale,
                recurrent_state=recurrent_state.astype(mx.float32),
            )
        except RuntimeError as exc:
            pytest.skip(f"Metal fused recurrent GDR kernel unavailable: {exc}")

        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(actual_state), np.asarray(expected_state), atol=1e-5, rtol=1e-5)

    def test_gdr_chunked_metal_matches_math(self):
        """The Metal chunked GDR step should match the reference math."""
        rng = np.random.default_rng(1)
        batch = 2
        seq_len = 4
        num_heads = 3
        head_dim = 8
        d_state = 8

        query = mx.array(rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(np.float32))
        key = mx.array(rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(np.float32))
        value = mx.array(rng.standard_normal((batch, seq_len, num_heads, d_state)).astype(np.float32))
        beta = mx.array(rng.standard_normal((batch, seq_len, num_heads)).astype(np.float32))
        decay = mx.array(rng.standard_normal((batch, seq_len, num_heads)).astype(np.float32))
        recurrent_state = mx.array(rng.standard_normal((batch, num_heads, head_dim, d_state)).astype(np.float32))

        query_n = _l2_normalize(query.astype(mx.float32), axis=-1) * (head_dim**-0.5)
        key_n = _l2_normalize(key.astype(mx.float32), axis=-1)
        decay_scale = mx.exp(decay.astype(mx.float32))

        expected_outputs = []
        state = recurrent_state.astype(mx.float32)
        for t in range(seq_len):
            decay_t = decay_scale[:, t, :, None, None]
            q_t = query_n[:, t]
            k_t = key_n[:, t]
            v_t = value[:, t].astype(mx.float32)
            b_t = beta[:, t].astype(mx.float32)
            scaled_state = state * decay_t
            kv_dot = mx.sum(scaled_state * k_t[:, :, :, None], axis=-2)
            delta = (v_t - kv_dot) * b_t[:, :, None]
            state = scaled_state + k_t[:, :, :, None] * delta[:, :, None, :]
            expected_outputs.append(mx.sum(state * q_t[:, :, :, None], axis=-2))
        expected_output = mx.stack(expected_outputs, axis=1)
        expected_state = state

        try:
            actual_output, actual_state = _gdr_chunked_forward_metal(
                query=query_n,
                key=key_n,
                value=value.astype(mx.float32),
                beta=beta.astype(mx.float32),
                decay_scale=decay_scale,
                recurrent_state=recurrent_state.astype(mx.float32),
            )
        except RuntimeError as exc:
            pytest.skip(f"Metal chunked GDR kernel unavailable: {exc}")

        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(actual_state), np.asarray(expected_state), atol=1e-5, rtol=1e-5)

    def test_gdr_chunked_grouped_metal_matches_math(self):
        """Grouped prefill should avoid repeated Q/K while matching the math."""
        rng = np.random.default_rng(9)
        batch = 2
        seq_len = 4
        query_heads = 2
        value_heads = 4
        head_dim = 8
        d_state = 8

        query = mx.array(rng.standard_normal((batch, seq_len, query_heads, head_dim)).astype(np.float32))
        key = mx.array(rng.standard_normal((batch, seq_len, query_heads, head_dim)).astype(np.float32))
        value = mx.array(rng.standard_normal((batch, seq_len, value_heads, d_state)).astype(np.float32))
        beta = mx.array(rng.standard_normal((batch, seq_len, value_heads)).astype(np.float32))
        decay = mx.array(rng.standard_normal((batch, seq_len, value_heads)).astype(np.float32))
        recurrent_state = mx.array(rng.standard_normal((batch, value_heads, head_dim, d_state)).astype(np.float32))

        query_n = _l2_normalize(query.astype(mx.float32), axis=-1) * (head_dim**-0.5)
        key_n = _l2_normalize(key.astype(mx.float32), axis=-1)
        decay_scale = mx.exp(decay.astype(mx.float32))

        expected_outputs = []
        state = recurrent_state.astype(mx.float32)
        head_states = [state[:, h] for h in range(value_heads)]
        expand_ratio = value_heads // query_heads
        for t in range(seq_len):
            token_outputs = []
            next_head_states = []
            for h in range(value_heads):
                qh = h // expand_ratio
                q_t = query_n[:, t, qh]
                k_t = key_n[:, t, qh]
                v_t = value[:, t, h].astype(mx.float32)
                b_t = beta[:, t, h].astype(mx.float32)
                decay_t = decay_scale[:, t, h, None, None]
                state_h = head_states[h]
                scaled_state = state_h * decay_t
                kv_dot = mx.sum(scaled_state * k_t[:, :, None], axis=-2)
                delta = (v_t - kv_dot) * b_t[:, None]
                state_h = scaled_state + k_t[:, :, None] * delta[:, None, :]
                next_head_states.append(state_h)
                token_outputs.append(mx.sum(state_h * q_t[:, :, None], axis=-2))
            head_states = next_head_states
            expected_outputs.append(mx.stack(token_outputs, axis=1))
        expected_output = mx.stack(expected_outputs, axis=1)
        expected_state = mx.stack(head_states, axis=1)

        try:
            actual_output, actual_state = _gdr_chunked_forward_metal_grouped(
                query=query_n,
                key=key_n,
                value=value.astype(mx.float32),
                beta=beta.astype(mx.float32),
                decay_scale=decay_scale,
                recurrent_state=recurrent_state.astype(mx.float32),
            )
        except RuntimeError as exc:
            pytest.skip(f"Metal grouped chunked GDR kernel unavailable: {exc}")

        np.testing.assert_allclose(np.asarray(actual_output), np.asarray(expected_output), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(actual_state), np.asarray(expected_state), atol=1e-5, rtol=1e-5)

    def test_decode_depthwise_conv_state_update_matches_math(self):
        """Decode convolution state update should match the reference ops."""
        rng = np.random.default_rng(11)
        batch = 2
        channels = 17
        kernel_size = 4
        conv_state = mx.array(rng.standard_normal((batch, channels, kernel_size)).astype(np.float16))
        new_value = mx.array(rng.standard_normal((batch, channels)).astype(np.float16))
        kernel = mx.array(rng.standard_normal((channels, kernel_size)).astype(np.float16))

        expected_state = mx.concatenate([conv_state[:, :, 1:], new_value[:, :, None]], axis=-1)
        expected_out = mx.array(np.asarray(mx.sum(expected_state * kernel[None, :, :], axis=-1))).astype(mx.float32)
        expected_out = expected_out / (1.0 + mx.exp(-expected_out))

        actual_out, actual_state = _decode_depthwise_conv_state_update(conv_state, new_value, kernel)
        mx.eval(actual_out, actual_state)

        np.testing.assert_allclose(np.asarray(actual_state), np.asarray(expected_state), atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(np.asarray(actual_out), np.asarray(expected_out), atol=2e-3, rtol=2e-3)
