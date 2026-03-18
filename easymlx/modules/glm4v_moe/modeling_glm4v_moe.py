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

"""GLM-4V MoE MLX implementation (serving/inference only).

This module provides the top-level GLM-4V MoE model classes that combine
a vision encoder with a Mixture-of-Experts language model for multimodal
inference.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules.glm4v.vision import VisionModel

from .glm4v_moe_configuration import Glm4VMoeConfig
from .language import LanguageModel


@register_module(task_type=TaskType.BASE_MODULE, config=Glm4VMoeConfig, model_type="glm4v_moe")
class Glm4VMoeModel(EasyMLXBaseModule):
    """Base GLM-4V MoE vision-language model.

    Combines a vision encoder with a MoE language backbone, merging vision
    features into the text embedding stream at image/video token positions.

    Attributes:
        config_class: The configuration class (``Glm4VMoeConfig``).
        model_config: Composite model configuration dataclass.
        vision_tower: Vision encoder module.
        language_model: MoE language model module.
    """

    config_class = Glm4VMoeConfig

    def __init__(self, config: Glm4VMoeConfig):
        """Initializes the GLM-4V MoE model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        model_config = config.to_model_config()
        self.model_config = model_config
        self.vision_tower = VisionModel(model_config.vision_config)
        self.language_model = LanguageModel(model_config.text_config, model_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ):
        """Computes input embeddings, optionally merging vision features.

        Args:
            input_ids: Input token IDs.
            pixel_values: Optional pixel values for vision inputs.
            image_grid_thw: Optional grid dimensions per image.

        Returns:
            Input embeddings with vision features merged at appropriate
            positions.
        """
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(pixel_values, image_grid_thw, output_hidden_states=False)

        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.model_config.image_token_id,
            self.model_config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merges vision features into text embeddings at token positions.

        Args:
            image_token_id: Token ID for image placeholders.
            video_token_id: Token ID for video placeholders.
            image_features: Vision features tensor.
            inputs_embeds: Text embeddings tensor.
            input_ids: Token IDs tensor.

        Returns:
            Merged embeddings tensor.

        Raises:
            ValueError: If position count mismatches feature count.
        """
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size, _ = input_ids.shape
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[feature_start_idx : feature_start_idx + num_positions]
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(image_mask_expanded, gathered_features, inputs_embeds[batch_idx])
                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

    @property
    def layers(self):
        """Returns the decoder layers from the language model."""
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        attention_mask: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        """Runs the full vision-language forward pass.

        Args:
            input_ids: Input token IDs.
            pixel_values: Optional pixel values.
            attention_mask: Optional attention mask.
            image_grid_thw: Optional image grid dimensions.
            video_grid_thw: Optional video grid dimensions.
            **kwargs: Additional keyword arguments for the language model.

        Returns:
            Logits tensor from the language model head.
        """
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        logits = self.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes and remaps weight keys for loading pretrained checkpoints.

        Args:
            weights: Weight dictionary.

        Returns:
            Sanitized weight dictionary with corrected key names.
        """

        def transform_key(key: str) -> str:
            """Remaps a single weight key to the internal naming convention.

            Args:
                key: Original weight key.

            Returns:
                Transformed key.
            """
            if "visual" in key:
                if "vision_tower" not in key:
                    key = key.replace("model.", "").replace("visual", "vision_tower")
            if "model.language_model" in key:
                key = key.replace("model.language_model", "language_model.model")
            if "lm_head" in key and not key.startswith("language_model"):
                key = key.replace("lm_head", "language_model.lm_head")
            return key

        return {transform_key(k): v for k, v in weights.items()}


@register_module(task_type=TaskType.CAUSAL_LM, config=Glm4VMoeConfig, model_type="glm4v_moe")
class Glm4VMoeForConditionalGeneration(EasyMLXBaseModule):
    """GLM-4V MoE model with a causal language modeling head.

    Wraps ``Glm4VMoeModel`` for autoregressive generation with optional
    vision inputs.

    Attributes:
        config_class: The configuration class (``Glm4VMoeConfig``).
        model: The underlying ``Glm4VMoeModel``.
    """

    config_class = Glm4VMoeConfig

    def __init__(self, config: Glm4VMoeConfig):
        """Initializes the conditional generation model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.model = Glm4VMoeModel(config)

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
        """Runs the forward pass to produce logits.

        Args:
            input_ids: Input token IDs.
            pixel_values: Optional pixel values.
            attention_mask: Optional attention mask.
            image_grid_thw: Optional image grid dimensions.
            video_grid_thw: Optional video grid dimensions.
            return_dict: Return ``CausalLMOutput`` or raw logits.
            **kwargs: Additional keyword arguments.

        Returns:
            ``CausalLMOutput`` or raw logits tensor.
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
        """Delegates weight sanitization to the underlying model.

        Args:
            weights: Weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """
        return self.model.sanitize(weights)
