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

"""Solar Open model wrappers for EasyMLX."""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.glm4_moe.modeling_glm4_moe import Glm4MoeModel

from .solar_open_configuration import SolarOpenConfig


@register_module(task_type=TaskType.BASE_MODULE, config=SolarOpenConfig, model_type="solar_open")
class SolarOpenModel(Glm4MoeModel):
    """Solar Open base transformer model.

    Directly reuses the GLM-4 MoE transformer stack (``Glm4MoeModel``)
    without any structural modifications. Weight sanitization for expert
    stacking is handled at the causal LM level.

    Attributes:
        config_class: Associated configuration class (``SolarOpenConfig``).

    Example:
        >>> config = SolarOpenConfig(hidden_size=2048)
        >>> model = SolarOpenModel(config)
    """

    config_class = SolarOpenConfig


@register_module(task_type=TaskType.CAUSAL_LM, config=SolarOpenConfig, model_type="solar_open")
class SolarOpenForCausalLM(BaseCausalLMModule[SolarOpenModel, SolarOpenConfig]):
    """Solar Open causal language model with an LM head.

    Wraps ``SolarOpenModel`` and provides a custom ``sanitize`` method
    that stacks per-expert weight shards from the upstream checkpoint
    into the fused ``SwitchGLU`` format expected by the GLM-4 MoE layers.

    Attributes:
        config_class: Associated configuration class (``SolarOpenConfig``).

    Example:
        >>> config = SolarOpenConfig(hidden_size=2048, num_hidden_layers=4)
        >>> model = SolarOpenForCausalLM(config)
    """

    config_class = SolarOpenConfig

    def __init__(self, config: SolarOpenConfig):
        """Initialize the Solar Open causal LM.

        Args:
            config: Solar Open model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=SolarOpenModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Stack per-expert weights and normalize to the GLM-4 MoE format.

        The upstream Solar Open checkpoint stores each expert's gate_proj,
        down_proj, and up_proj as separate keys (e.g.,
        ``mlp.experts.0.gate_proj.weight``). This method stacks them along
        a new expert axis into a single ``switch_mlp.{proj}.{suffix}`` tensor,
        matching the fused ``SwitchGLU`` layout. It also strips any layers
        beyond ``num_hidden_layers`` (e.g., auxiliary MTP layers).

        Args:
            weights: Raw checkpoint weight dict with per-expert keys.

        Returns:
            Cleaned weight dict with stacked expert tensors and extra
            layers removed.
        """
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for _src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for suffix in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{dst}.{suffix}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{dst}.{suffix}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{dst}.{suffix}"] = mx.stack(to_join)

        mpt_layer = self.config.num_hidden_layers
        weights = {k: v for k, v in weights.items() if not k.startswith(f"model.layers.{mpt_layer}")}
        return super().sanitize(weights)


__all__ = ("SolarOpenForCausalLM", "SolarOpenModel")
