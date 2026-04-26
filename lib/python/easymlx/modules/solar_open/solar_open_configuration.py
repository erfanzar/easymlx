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

"""Solar Open configuration for EasyMLX."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..glm4_moe import Glm4MoeConfig


@register_config("solar_open")
class SolarOpenConfig(Glm4MoeConfig):
    """Configuration for the Solar Open Mixture-of-Experts model.

    Solar Open reuses the GLM-4 MoE transformer architecture with additional
    routing configuration parameters. It inherits all attributes from
    ``Glm4MoeConfig`` and adds ``scoring_func`` and ``topk_method`` for
    controlling expert routing behavior.

    Attributes:
        model_type: The model type identifier (``"solar_open"``).
        scoring_func: Scoring function for the MoE router (e.g., ``"sigmoid"``).
        topk_method: Top-k selection method for expert routing
            (e.g., ``"noaux_tc"``).

    Example:
        >>> config = SolarOpenConfig(scoring_func="sigmoid", topk_method="noaux_tc")
        >>> config.model_type
        'solar_open'
    """

    model_type = "solar_open"

    def __init__(
        self,
        *,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        """Initialize a Solar Open configuration.

        Args:
            scoring_func: Scoring function for expert routing. Defaults
                to ``"sigmoid"``.
            topk_method: Method for selecting top-k experts. Defaults to
                ``"noaux_tc"`` (no-auxiliary-loss top-choice).
            **kwargs: All remaining arguments are forwarded to ``Glm4MoeConfig``.
        """
        super().__init__(**kwargs)
        self.scoring_func = str(scoring_func)
        self.topk_method = str(topk_method)


__all__ = ("SolarOpenConfig",)
