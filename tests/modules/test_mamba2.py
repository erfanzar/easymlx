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

"""Tests for Mamba2 model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.mamba2 import Mamba2Config, Mamba2ForCausalLM, Mamba2Model
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestMamba2:
    """Test suite for Mamba2 SSM model."""

    @pytest.fixture
    def mamba2_config(self):
        return Mamba2Config(
            vocab_size=128,
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            state_size=16,
            num_hidden_layers=2,
            conv_kernel=4,
            n_groups=1,
            use_bias=False,
            use_conv_bias=True,
            time_step_rank=8,
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "mamba2")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "mamba2")

        assert base_registration.module is Mamba2Model
        assert base_registration.config is Mamba2Config
        assert lm_registration.module is Mamba2ForCausalLM
        assert lm_registration.config is Mamba2Config

    def test_causal_lm(self, mamba2_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="mamba2",
            model_cls=Mamba2ForCausalLM,
            config=mamba2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mamba2 CAUSAL_LM failed: {result.error_message}"

    def test_sanitize(self, mamba2_config):

        model = Mamba2ForCausalLM(mamba2_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        upstream_weights = dict(local_weights)
        for k, v in upstream_weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                upstream_weights[k] = v.moveaxis(1, 2)

        sanitized = model.sanitize(upstream_weights)
        for k, v in sanitized.items():
            if "conv1d.weight" in k and v.ndim == 3:
                assert v.shape[-1] == 1, f"conv1d weight {k} not transposed correctly"
