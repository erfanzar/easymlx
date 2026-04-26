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

"""Base class for Vision-Language Models on MLX.

This module provides ``BaseVisionLanguageModule``, the foundation for multimodal
models that process both image and text inputs (e.g., LLaVA, Qwen2-VL).
It extends ``BaseConditionalGenerationModule`` with support for vision tower
management, feature extraction, multimodal embedding merging, and optional
video input processing.
"""

from __future__ import annotations

import typing as tp
from abc import abstractmethod

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.modeling_outputs import VLMCausalLMOutput

from .conditional_generation_module import BaseConditionalGenerationModule

ModelT = tp.TypeVar("ModelT", bound=nn.Module)
ConfigT = tp.TypeVar("ConfigT", bound=EasyMLXBaseConfig)


class BaseVisionLanguageModule(BaseConditionalGenerationModule[ModelT, ConfigT]):
    """Base class for all Vision-Language Models on MLX.

    Inherits from BaseConditionalGenerationModule and adds support for
    multimodal (vision + language) processing including vision tower
    management, feature extraction, and embedding merging.
    """

    _supports_video: bool = False
    _vision_tower_name: str = "vision_tower"
    _projector_name: str = "multi_modal_projector"
    _language_model_name: str = "language_model"

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
        create_lm_head: bool = True,
        image_token_index: int | None = None,
        video_token_index: int | None = None,
    ):
        """Initialize the vision-language module.

        Args:
            config: Model configuration containing both vision and language
                hyperparameters.
            base_model: Pre-instantiated base model instance.
            base_model_class: Base model class to instantiate if
                ``base_model`` is not provided.
            base_model_name: Attribute name for the base model.
                Defaults to ``"model"``.
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the LM head. Defaults to False.
            logit_cap: Maximum absolute value for output logits.
                Defaults to None.
            lm_head_name: Attribute name for the LM head. Defaults to
                ``"lm_head"``.
            lm_head_bias: Whether to include bias in the LM head.
                Defaults to False.
            create_lm_head: Whether to create an LM head layer.
                Defaults to True.
            image_token_index: Token ID used as placeholder for image
                embeddings. Falls back to ``config.image_token_id``.
            video_token_index: Token ID used as placeholder for video
                embeddings. Falls back to ``config.video_token_id``.
        """
        super().__init__(
            config=config,
            base_model=base_model,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            tie_word_embeddings=tie_word_embeddings,
            logit_cap=logit_cap,
            lm_head_name=lm_head_name,
            lm_head_bias=lm_head_bias,
            create_lm_head=create_lm_head,
        )
        self._image_token_index = image_token_index or getattr(config, "image_token_id", None)
        self._video_token_index = video_token_index or getattr(config, "video_token_id", None)

    @abstractmethod
    def get_image_features(self, pixel_values: mx.array, **kwargs) -> mx.array:
        """Extract and project image features from pixel values.

        Must be implemented by subclasses.

        Args:
            pixel_values: Input images (batch, channels, height, width).

        Returns:
            Projected image features (batch, num_patches, hidden).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_image_features()")

    def get_video_features(
        self, pixel_values_videos: mx.array, video_grid_thw: tuple | None = None, **kwargs
    ) -> mx.array:
        """Extract and project video features.

        Override this method for video-capable models.

        Args:
            pixel_values_videos: Input video frames tensor.
            video_grid_thw: Temporal-height-width grid dimensions for the
                video patches.
            **kwargs: Additional keyword arguments.

        Returns:
            Projected video features.

        Raises:
            NotImplementedError: If the model does not support video input.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support video input.")

    def get_vision_tower(self) -> nn.Module:
        """Return the vision encoder component.

        Searches for the vision tower using the configured attribute name
        and common fallback names.

        Returns:
            The vision encoder ``nn.Module``.

        Raises:
            AttributeError: If no vision tower can be found on the base model.
        """
        if hasattr(self.base_model, self._vision_tower_name):
            return getattr(self.base_model, self._vision_tower_name)
        for name in ["visual", "vision_model", "image_encoder"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(
            f"Cannot find vision tower. Tried: {self._vision_tower_name}, visual, vision_model, image_encoder"
        )

    def get_projector(self) -> nn.Module:
        """Return the multimodal projector component.

        The projector maps vision features to the language model's hidden
        dimension space.

        Returns:
            The projector ``nn.Module``.

        Raises:
            AttributeError: If no projector can be found on the base model.
        """
        if hasattr(self.base_model, self._projector_name):
            return getattr(self.base_model, self._projector_name)
        for name in ["projector", "mm_projector", "vision_projector"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(
            f"Cannot find projector. Tried: {self._projector_name}, projector, mm_projector, vision_projector"
        )

    def get_language_model(self) -> nn.Module:
        """Return the language model component.

        Returns:
            The language model ``nn.Module``.

        Raises:
            AttributeError: If no language model can be found on the base model.
        """
        if hasattr(self.base_model, self._language_model_name):
            return getattr(self.base_model, self._language_model_name)
        for name in ["text_model", "llm", "decoder", "language_model"]:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
        raise AttributeError(f"Cannot find language model. Tried: {self._language_model_name}, text_model, llm, decoder")

    def get_encoder(self) -> nn.Module:
        """Return the encoder, which for VLMs is the vision tower.

        Returns:
            The vision encoder ``nn.Module``.
        """
        return self.get_vision_tower()

    def get_decoder(self) -> nn.Module:
        """Return the decoder, which for VLMs is the language model.

        Returns:
            The language model ``nn.Module``.
        """
        return self.get_language_model()

    @staticmethod
    def merge_multimodal_embeddings(
        input_ids: mx.array,
        inputs_embeds: mx.array,
        multimodal_embeddings: mx.array,
        placeholder_token_id: int | list[int],
    ) -> mx.array:
        """Merge vision embeddings into text embeddings at placeholder positions.

        Args:
            input_ids: (batch, seq_len) token IDs with placeholder positions.
            inputs_embeds: (batch, seq_len, hidden) text embeddings.
            multimodal_embeddings: (num_tokens, hidden) flattened vision features.
            placeholder_token_id: Token ID(s) marking vision placeholder positions.

        Returns:
            Merged embeddings (batch, seq_len, hidden).
        """
        batch_size, seq_len, hidden = inputs_embeds.shape
        if isinstance(placeholder_token_id, list):
            is_multimodal = mx.zeros_like(input_ids, dtype=mx.bool_)
            for tid in placeholder_token_id:
                is_multimodal = is_multimodal | (input_ids == tid)
        else:
            is_multimodal = input_ids == placeholder_token_id

        flat_mask = is_multimodal.reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, hidden)

        dummy_row = mx.zeros_like(multimodal_embeddings[0:1])
        padded = mx.concatenate([dummy_row, multimodal_embeddings], axis=0)
        gather_indices = mx.cumsum(flat_mask.astype(mx.int32))
        update_values = padded[gather_indices]

        merged = mx.where(flat_mask[:, None], update_values, flat_embeds)
        return merged.reshape(batch_size, seq_len, hidden)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        pixel_values_videos: mx.array | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        cache: tp.Any | None = None,
        inputs_embeds: mx.array | None = None,
        apply_lm_head: bool = True,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass through the vision-language model.

        Passes all provided inputs to the base model and optionally applies
        the LM head to produce vocabulary logits.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
            position_ids: Position indices for positional embeddings.
            pixel_values: Input images tensor for the vision tower.
            pixel_values_videos: Input video frames tensor.
            image_grid_thw: Temporal-height-width grid for image patches.
            video_grid_thw: Temporal-height-width grid for video patches.
            cache: Cached key-value states for incremental decoding.
            inputs_embeds: Pre-computed input embeddings.
            apply_lm_head: Whether to project hidden states to logits.
                Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            ``VLMCausalLMOutput`` containing logits, cache, and optional
            image hidden states.
        """
        base_kwargs: dict[str, tp.Any] = {}
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds
        elif input_ids is not None:
            base_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            base_kwargs["position_ids"] = position_ids
        if pixel_values is not None:
            base_kwargs["pixel_values"] = pixel_values
        if pixel_values_videos is not None:
            base_kwargs["pixel_values_videos"] = pixel_values_videos
        if image_grid_thw is not None:
            base_kwargs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            base_kwargs["video_grid_thw"] = video_grid_thw
        if cache is not None:
            base_kwargs["cache"] = cache
        base_kwargs.update(kwargs)

        outputs = self.base_model(**base_kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            out_cache = outputs[1] if len(outputs) > 1 else None
        else:
            hidden_states = getattr(outputs, "last_hidden_state", outputs)
            out_cache = getattr(outputs, "cache", getattr(outputs, "past_key_values", None))

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(hidden_states)

        image_hidden_states = getattr(outputs, "image_hidden_states", None)

        return VLMCausalLMOutput(
            logits=logits,
            cache=out_cache,
            image_hidden_states=image_hidden_states,
        )
