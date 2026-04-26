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

"""Tests for Mamba model."""

import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.mamba import MambaConfig, MambaForCausalLM, MambaModel
from mlx.utils import tree_flatten

from .test_utils import CausalLMTester


class TestMamba:
    """Test suite for Mamba SSM model."""

    @pytest.fixture
    def mamba_config(self):
        return MambaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            state_size=16,
            num_hidden_layers=2,
            conv_kernel=4,
            use_bias=False,
            use_conv_bias=True,
            time_step_rank=8,
            tie_word_embeddings=True,
        )

    def test_registry(self):
        base_registration = registry.get_module_registration(TaskType.BASE_MODULE, "mamba")
        lm_registration = registry.get_module_registration(TaskType.CAUSAL_LM, "mamba")

        assert base_registration.module is MambaModel
        assert base_registration.config is MambaConfig
        assert lm_registration.module is MambaForCausalLM
        assert lm_registration.config is MambaConfig

    def test_causal_lm(self, mamba_config, small_model_config):
        tester = CausalLMTester()
        result = tester.run(
            module_name="mamba",
            model_cls=MambaForCausalLM,
            config=mamba_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mamba CAUSAL_LM failed: {result.error_message}"

    def test_sanitize(self, mamba_config):

        model = MambaForCausalLM(mamba_config)
        local_weights = dict(tree_flatten(model.parameters(), destination={}))

        upstream_weights = dict(local_weights)
        for k, v in upstream_weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                upstream_weights[k] = v.moveaxis(1, 2)

        sanitized = model.sanitize(upstream_weights)
        for k, v in sanitized.items():
            if "conv1d.weight" in k and v.ndim == 3:
                assert v.shape[-1] == 1, f"conv1d weight {k} not transposed correctly"
