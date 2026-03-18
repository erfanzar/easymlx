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

"""GLM-4.6V MLX implementation (serving/inference only).

This module provides the MLX model classes for GLM-4.6V, which extends the
GLM-4V architecture. ``Glm46VModel`` inherits from ``Glm4VModel`` and
``Glm46VForConditionalGeneration`` wraps it as a causal language model.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules.glm4v.modeling_glm4v import Glm4VModel

from .glm46v_configuration import Glm46VConfig


@register_module(task_type=TaskType.BASE_MODULE, config=Glm46VConfig, model_type="glm46v")
class Glm46VModel(Glm4VModel):
    """Base GLM-4.6V vision-language model.

    Inherits from ``Glm4VModel`` and is registered as a base module under the
    ``"glm46v"`` model type. This class reuses the full GLM-4V architecture
    with the GLM-4.6V configuration.

    Attributes:
        config_class: The configuration class for this model (``Glm46VConfig``).
    """

    config_class = Glm46VConfig


@register_module(task_type=TaskType.CAUSAL_LM, config=Glm46VConfig, model_type="glm46v")
class Glm46VForConditionalGeneration(EasyMLXBaseModule):
    """GLM-4.6V model with a causal language modeling head.

    Wraps ``Glm46VModel`` and produces logits suitable for autoregressive
    text generation conditioned on optional image/video inputs.

    Attributes:
        config_class: The configuration class for this model (``Glm46VConfig``).
        model: The underlying ``Glm46VModel`` instance.
    """

    config_class = Glm46VConfig

    def __init__(self, config: Glm46VConfig):
        """Initializes the GLM-4.6V conditional generation model.

        Args:
            config: Model configuration instance.
        """
        super().__init__(config)
        self.model = Glm46VModel(config)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        attention_mask: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> mx.array | CausalLMOutput:
        """Runs the forward pass to produce language model logits.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            pixel_values: Optional pixel values for image/video inputs.
            attention_mask: Optional attention mask of shape
                ``(batch_size, seq_len)``.
            image_grid_thw: Optional grid dimensions ``(T, H, W)`` per image.
            video_grid_thw: Optional grid dimensions ``(T, H, W)`` per video.
            return_dict: If True, returns a ``CausalLMOutput``; otherwise
                returns raw logits. Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            A ``CausalLMOutput`` containing logits if ``return_dict`` is True,
            otherwise the raw logits tensor.
        """
        logits = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        if return_dict:
            return CausalLMOutput(logits=logits)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes and remaps weight keys for loading pretrained checkpoints.

        Delegates to the underlying ``Glm46VModel.sanitize`` method.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary with corrected key names.
        """
        return self.model.sanitize(weights)
