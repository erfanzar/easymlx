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

"""GLM-4.6V configuration (serving/inference only).

This module defines the configuration class for the GLM-4.6V vision-language
model. GLM-4.6V extends the GLM-4V architecture and reuses its text and vision
configuration dataclasses.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.glm4v.glm4v_configuration import Glm4VModelConfig, Glm4VTextConfig, Glm4VVisionConfig


@register_config("glm46v")
class Glm46VConfig(EasyMLXBaseConfig):
    """Configuration for the GLM-4.6V vision-language model.

    This configuration extends ``EasyMLXBaseConfig`` and composes separate
    text and vision sub-configurations. It is registered under the model type
    ``"glm46v"`` and reuses the GLM-4V text/vision config dataclasses.

    Attributes:
        model_type: Identifier string for this model type (``"glm46v"``).
        text_config: Serialized dictionary of text sub-configuration parameters.
        vision_config: Serialized dictionary of vision sub-configuration parameters.
        image_token_id: Token ID representing an image placeholder in the input.
        video_token_id: Token ID representing a video placeholder in the input.
        image_start_token_id: Token ID marking the start of an image region.
        image_end_token_id: Token ID marking the end of an image region.
        video_start_token_id: Token ID marking the start of a video region.
        video_end_token_id: Token ID marking the end of a video region.
        vocab_size: Size of the vocabulary.
        ignore_index: Label index to ignore during loss computation.
        hidden_size: Dimensionality of the model hidden states.
    """

    model_type = "glm46v"

    def __init__(
        self,
        *,
        text_config: Glm4VTextConfig | dict[str, Any] | None = None,
        vision_config: Glm4VVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 151343,
        video_token_id: int = 151344,
        image_start_token_id: int = 151339,
        image_end_token_id: int = 151340,
        video_start_token_id: int = 151361,
        video_end_token_id: int = 151362,
        vocab_size: int = 257152,
        ignore_index: int = -100,
        hidden_size: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: list["int"] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initializes a GLM-4.6V configuration.

        Args:
            text_config: Text sub-configuration. Accepts a ``Glm4VTextConfig``
                instance, a dictionary of keyword arguments, or ``None`` for
                defaults.
            vision_config: Vision sub-configuration. Accepts a
                ``Glm4VVisionConfig`` instance, a dictionary of keyword
                arguments, or ``None`` for defaults.
            image_token_id: Token ID for image placeholders. Defaults to 151343.
            video_token_id: Token ID for video placeholders. Defaults to 151344.
            image_start_token_id: Token ID marking image region start.
                Defaults to 151339.
            image_end_token_id: Token ID marking image region end.
                Defaults to 151340.
            video_start_token_id: Token ID marking video region start.
                Defaults to 151361.
            video_end_token_id: Token ID marking video region end.
                Defaults to 151362.
            vocab_size: Vocabulary size. Defaults to 257152.
            ignore_index: Label index ignored by the loss function.
                Defaults to -100.
            hidden_size: Hidden state dimensionality. Defaults to 2048.
            pad_token_id: Padding token ID. Defaults to 0.
            eos_token_id: End-of-sequence token ID(s). Defaults to the text
                config's EOS token IDs.
            tie_word_embeddings: Whether to tie input and output embeddings.
                Defaults to False.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if text_config is None:
            text_cfg = Glm4VTextConfig()
        elif isinstance(text_config, Glm4VTextConfig):
            text_cfg = text_config
        else:
            text_cfg = Glm4VTextConfig(**text_config)

        if vision_config is None:
            vision_cfg = Glm4VVisionConfig(
                depth=24,
                hidden_size=1536,
                intermediate_size=13696,
                num_heads=12,
                patch_size=14,
            )
        elif isinstance(vision_config, Glm4VVisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Glm4VVisionConfig(**vision_config)

        if eos_token_id is None:
            eos_token_id = list(text_cfg.eos_token_id)

        self.text_config = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else vars(text_cfg)
        self.vision_config = vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.image_start_token_id = int(image_start_token_id)
        self.image_end_token_id = int(image_end_token_id)
        self.video_start_token_id = int(video_start_token_id)
        self.video_end_token_id = int(video_end_token_id)
        self.vocab_size = int(vocab_size)
        self.ignore_index = int(ignore_index)
        self.hidden_size = int(hidden_size)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_model_config(self) -> Glm4VModelConfig:
        """Converts this config into a ``Glm4VModelConfig`` dataclass.

        Returns:
            A ``Glm4VModelConfig`` populated from this configuration's text
            and vision sub-configs and token IDs.
        """
        text_cfg = Glm4VTextConfig(**self.text_config)
        vision_cfg = Glm4VVisionConfig(**self.vision_config)
        return Glm4VModelConfig(
            text_config=text_cfg,
            vision_config=vision_cfg,
            model_type=self.model_type,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_id,
            image_token_id=self.image_token_id,
            video_token_index=self.video_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.image_start_token_id,
            vision_end_token_id=self.image_end_token_id,
            hidden_size=self.hidden_size,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> Glm4VTextConfig:
        """Retrieves the text sub-configuration as a ``Glm4VTextConfig``.

        Args:
            decoder: Unused. Present for API compatibility.
            encoder: Unused. Present for API compatibility.

        Returns:
            The text configuration as a ``Glm4VTextConfig`` dataclass instance.
        """
        del decoder, encoder
        text_cfg = self.text_config
        if isinstance(text_cfg, dict):
            text_cfg = Glm4VTextConfig(**text_cfg)
        return text_cfg


__all__ = "Glm46VConfig"
