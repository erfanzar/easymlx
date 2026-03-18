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

"""GLM-4V configuration (serving/inference only).

This module defines configuration classes and the main config class for
the GLM-4V vision-language model, covering text, vision, and composite model
configurations.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


class Glm4VTextConfig(EasyMLXBaseConfig):
    """Configuration class for the GLM-4V text (language) sub-model.

    Attributes:
        model_type: Identifier for the text model type.
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the hidden states.
        eos_token_id: List of end-of-sequence token IDs.
        intermediate_size: Intermediate MLP dimensionality.
        max_position_embeddings: Maximum sequence length.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer decoder layers.
        num_key_value_heads: Number of key-value heads for GQA.
        rms_norm_eps: Epsilon for RMS normalization.
        rope_theta: Base frequency for rotary position embeddings.
        attention_bias: Whether attention projections include bias.
        attention_dropout: Dropout rate for attention weights.
        hidden_act: Activation function name.
        initializer_range: Weight initialization range.
        partial_rotary_factor: Fraction of head dim for rotary embeddings.
        rope_scaling: RoPE scaling configuration dictionary.
        pad_token_id: Padding token ID.
        use_cache: Whether to use KV caching during generation.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 151552,
        hidden_size: int = 4096,
        eos_token_id: list[int] | None = None,
        intermediate_size: int = 13696,
        max_position_embeddings: int = 65536,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 40,
        num_key_value_heads: int = 2,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        partial_rotary_factor: float = 0.5,
        rope_scaling: dict[str, Any] | None = None,
        pad_token_id: int = 151329,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eos_token_id = eos_token_id if eos_token_id is not None else [151329, 151336, 151338, 151348]
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_scaling = (
            rope_scaling if rope_scaling is not None else {"rope_type": "default", "mrope_section": [8, 12, 12]}
        )
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache


class Glm4VVisionConfig(EasyMLXBaseConfig):
    """Configuration class for the GLM-4V vision encoder sub-model.

    Attributes:
        model_type: Identifier for the vision model type.
        depth: Number of vision transformer blocks.
        hidden_size: Vision encoder hidden dimensionality.
        intermediate_size: Vision MLP intermediate dimensionality.
        num_heads: Number of attention heads in the vision encoder.
        patch_size: Spatial patch size for image tokenization.
        window_size: Window size for windowed attention (in pixels).
        image_size: Default input image size.
        in_channels: Number of input image channels.
        rms_norm_eps: Epsilon for RMS normalization.
        attention_bias: Whether attention includes bias terms.
        attention_dropout: Dropout rate for attention.
        hidden_act: Activation function name.
        initializer_range: Weight initialization range.
        out_hidden_size: Output hidden dimensionality after patch merging.
        spatial_merge_size: Spatial downsampling factor for patch merging.
        temporal_patch_size: Temporal patch size for video inputs.
    """

    def __init__(
        self,
        *,
        depth: int = 24,
        hidden_size: int = 1536,
        intermediate_size: int = 13696,
        num_heads: int = 12,
        patch_size: int = 14,
        window_size: int = 112,
        image_size: int = 336,
        in_channels: int = 3,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        out_hidden_size: int = 4096,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size


class Glm4VModelConfig(EasyMLXBaseConfig):
    """Composite configuration class for the full GLM-4V model.

    Combines text and vision configurations with model-level parameters
    such as special token IDs.

    Attributes:
        text_config: Text sub-model configuration.
        vision_config: Vision sub-model configuration.
        model_type: Model type identifier.
        vocab_size: Vocabulary size.
        ignore_index: Label index to ignore in loss computation.
        image_token_index: Token index for image placeholders.
        image_token_id: Token ID for image placeholders.
        video_token_index: Token index for video placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID for vision region start.
        vision_end_token_id: Token ID for vision region end.
        image_start_token_id: Token ID for image region start.
        image_end_token_id: Token ID for image region end.
        video_start_token_id: Token ID for video region start.
        video_end_token_id: Token ID for video region end.
        hidden_size: Model hidden state dimensionality.
        pad_token_id: Padding token ID.
        eos_token_id: End-of-sequence token ID(s).
    """

    def __init__(
        self,
        *,
        text_config: Glm4VTextConfig | dict[str, Any] | None = None,
        vision_config: Glm4VVisionConfig | dict[str, Any] | None = None,
        vocab_size: int = 257152,
        ignore_index: int = -100,
        image_token_index: int = 151363,
        image_token_id: int = 151363,
        video_token_index: int = 151364,
        video_token_id: int = 151364,
        vision_start_token_id: int = 151339,
        vision_end_token_id: int = 151340,
        image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        hidden_size: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Glm4VTextConfig(**text_config)
        if vision_config is not None and isinstance(vision_config, dict):
            vision_config = Glm4VVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.image_token_id = image_token_id
        self.video_token_index = video_token_index
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        if eos_token_id is None and self.text_config is not None:
            text_cfg_dict = (
                self.text_config.to_dict() if isinstance(self.text_config, Glm4VTextConfig) else self.text_config
            )
            eos_token_id = text_cfg_dict.get("eos_token_id")
        self.eos_token_id = eos_token_id

        if image_start_token_id is None:
            image_start_token_id = vision_start_token_id
        if image_end_token_id is None:
            image_end_token_id = vision_end_token_id
        if video_start_token_id is None:
            video_start_token_id = vision_start_token_id
        if video_end_token_id is None:
            video_end_token_id = vision_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id


@register_config("glm4v")
class Glm4VConfig(EasyMLXBaseConfig):
    """Configuration for the GLM-4V vision-language model.

    Extends ``EasyMLXBaseConfig`` and composes text and vision sub-configurations.
    Registered under the model type ``"glm4v"``. Handles normalization of
    image/video token IDs and their start/end markers.

    Attributes:
        model_type: Identifier string (``"glm4v"``).
        text_config: Serialized text sub-configuration dictionary.
        vision_config: Serialized vision sub-configuration dictionary.
        vocab_size: Vocabulary size.
        ignore_index: Label index to ignore during loss computation.
        image_token_index: Token index for image placeholders.
        image_token_id: Token ID for image placeholders.
        video_token_index: Token index for video placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking vision region start.
        vision_end_token_id: Token ID marking vision region end.
        image_start_token_id: Token ID marking image region start.
        image_end_token_id: Token ID marking image region end.
        video_start_token_id: Token ID marking video region start.
        video_end_token_id: Token ID marking video region end.
        hidden_size: Hidden state dimensionality.
    """

    model_type = "glm4v"

    def __init__(
        self,
        *,
        text_config: Glm4VTextConfig | dict[str, Any] | None = None,
        vision_config: Glm4VVisionConfig | dict[str, Any] | None = None,
        vocab_size: int = 257152,
        ignore_index: int = -100,
        image_token_index: int | None = None,
        image_token_id: int | None = None,
        video_token_index: int | None = None,
        video_token_id: int | None = None,
        vision_start_token_id: int = 151339,
        vision_end_token_id: int = 151340,
        image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        hidden_size: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initializes a GLM-4V configuration.

        Args:
            text_config: Text sub-configuration. Accepts a ``Glm4VTextConfig``
                instance, a dictionary, or None for defaults.
            vision_config: Vision sub-configuration. Accepts a
                ``Glm4VVisionConfig`` instance, a dictionary, or None for
                defaults.
            vocab_size: Vocabulary size. Defaults to 257152.
            ignore_index: Loss ignore index. Defaults to -100.
            image_token_index: Image token index. Defaults to None (auto).
            image_token_id: Image token ID. Defaults to None (auto).
            video_token_index: Video token index. Defaults to None (auto).
            video_token_id: Video token ID. Defaults to None (auto).
            vision_start_token_id: Vision start token. Defaults to 151339.
            vision_end_token_id: Vision end token. Defaults to 151340.
            image_start_token_id: Image start token. Defaults to None (auto).
            image_end_token_id: Image end token. Defaults to None (auto).
            video_start_token_id: Video start token. Defaults to None (auto).
            video_end_token_id: Video end token. Defaults to None (auto).
            hidden_size: Hidden state dim. Defaults to 2048.
            pad_token_id: Padding token ID. Defaults to 0.
            eos_token_id: EOS token ID(s). Defaults to text config values.
            tie_word_embeddings: Tie embeddings. Defaults to False.
            **kwargs: Additional keyword arguments for the base class.
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

        default_image_token_id = 151363
        default_video_token_id = 151364

        if image_token_id is None and image_token_index is None:
            image_token_id = default_image_token_id
        if image_token_index is None:
            image_token_index = image_token_id
        if image_token_id is None:
            image_token_id = image_token_index

        if video_token_id is None and video_token_index is None:
            video_token_id = default_video_token_id
        if video_token_index is None:
            video_token_index = video_token_id
        if video_token_id is None:
            video_token_id = video_token_index

        if image_start_token_id is None and video_start_token_id is not None:
            image_start_token_id = video_start_token_id
        if image_end_token_id is None and video_end_token_id is not None:
            image_end_token_id = video_end_token_id

        if image_start_token_id is not None:
            vision_start_token_id = int(image_start_token_id)
        if image_end_token_id is not None:
            vision_end_token_id = int(image_end_token_id)

        if image_start_token_id is None:
            image_start_token_id = vision_start_token_id
        if image_end_token_id is None:
            image_end_token_id = vision_end_token_id
        if video_start_token_id is None:
            video_start_token_id = vision_start_token_id
        if video_end_token_id is None:
            video_end_token_id = vision_end_token_id

        self.text_config = text_cfg.to_dict()
        self.vision_config = vision_cfg.to_dict()
        self.vocab_size = int(vocab_size)
        self.ignore_index = int(ignore_index)
        self.image_token_index = int(image_token_index)
        self.image_token_id = int(image_token_id)
        self.video_token_index = int(video_token_index)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        self.image_start_token_id = int(image_start_token_id)
        self.image_end_token_id = int(image_end_token_id)
        self.video_start_token_id = int(video_start_token_id)
        self.video_end_token_id = int(video_end_token_id)
        self.hidden_size = int(hidden_size)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_model_config(self) -> Glm4VModelConfig:
        """Converts this config into a ``Glm4VModelConfig`` instance.

        Returns:
            A ``Glm4VModelConfig`` populated from this configuration's
            text/vision sub-configs and token IDs.
        """
        text_cfg = Glm4VTextConfig(**self.text_config)
        vision_cfg = Glm4VVisionConfig(**self.vision_config)
        return Glm4VModelConfig(
            text_config=text_cfg,
            vision_config=vision_cfg,
            model_type=self.model_type,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            image_token_id=self.image_token_id,
            video_token_index=self.video_token_index,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
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
            The text configuration as a ``Glm4VTextConfig`` instance.
        """
        del decoder, encoder
        text_cfg = self.text_config
        if isinstance(text_cfg, dict):
            text_cfg = Glm4VTextConfig(**text_cfg)
        return text_cfg


Glm4vConfig = Glm4VConfig
Glm4vModelConfig = Glm4VModelConfig
Glm4vTextConfig = Glm4VTextConfig
Glm4vVisionConfig = Glm4VVisionConfig

__all__ = (
    "Glm4VConfig",
    "Glm4VModelConfig",
    "Glm4VTextConfig",
    "Glm4VVisionConfig",
    "Glm4vConfig",
    "Glm4vModelConfig",
    "Glm4vTextConfig",
    "Glm4vVisionConfig",
)
