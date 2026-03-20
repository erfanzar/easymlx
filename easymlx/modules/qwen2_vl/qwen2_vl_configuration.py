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

"""Qwen2-VL configuration classes for serving and inference.

This module defines the hierarchical configuration classes for the
Qwen2-VL multimodal model, covering vision encoder, text decoder, and the
top-level composite configuration registered with the EasyMLX factory.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


class Qwen2VLVisionConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen2-VL vision encoder.

    Attributes:
        model_type: Identifier string for the vision sub-model.
        depth: Number of vision transformer blocks.
        embed_dim: Dimensionality of the vision transformer embeddings.
        hidden_size: Dimensionality of the vision MLP intermediate layer.
        hidden_act: Activation function name for the vision MLP.
        mlp_ratio: MLP expansion ratio.
        num_heads: Number of attention heads in the vision transformer.
        in_channels: Number of input image channels.
        patch_size: Size of each image patch.
        spatial_merge_size: Spatial merge factor for patch tokens.
        temporal_patch_size: Temporal patch size for video inputs.
        initializer_range: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        *,
        depth: int = 32,
        embed_dim: int = 1280,
        hidden_size: int = 3584,
        hidden_act: str = "quick_gelu",
        mlp_ratio: int = 4,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        """Initialize the Qwen2-VL vision encoder configuration.

        Args:
            depth: Number of vision transformer blocks.
            embed_dim: Dimensionality of the vision transformer embeddings.
            hidden_size: Dimensionality of the vision MLP intermediate layer.
            hidden_act: Activation function name for the vision MLP.
            mlp_ratio: MLP expansion ratio.
            num_heads: Number of attention heads in the vision transformer.
            in_channels: Number of input image channels.
            patch_size: Size of each image patch.
            spatial_merge_size: Spatial merge factor for patch tokens.
            temporal_patch_size: Temporal patch size for video inputs.
            initializer_range: Standard deviation for weight initialization.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range


class Qwen2VLTextConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen2-VL text decoder.

    Attributes:
        model_type: Identifier string for the text sub-model.
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the text transformer hidden states.
        intermediate_size: Dimensionality of the MLP intermediate layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled.
        tie_word_embeddings: Whether input/output embeddings share weights.
        rope_theta: Base frequency for rotary positional embeddings.
        use_sliding_window: Whether sliding-window attention is enabled.
        sliding_window: Size of the sliding attention window.
        max_window_layers: Layer index threshold for sliding window.
        attention_dropout: Dropout rate for attention weights.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        rope_parameters: Alternative key for RoPE scaling parameters.
        layer_types: Per-layer attention type strings.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        intermediate_size: int = 29568,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 80,
        attention_dropout: float = 0.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the Qwen2-VL text decoder configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the text transformer hidden states.
            intermediate_size: Dimensionality of the MLP intermediate layer.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value heads for GQA. Defaults
                to ``num_attention_heads`` when ``None``.
            hidden_act: Activation function name for MLP layers.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMS normalization.
            use_cache: Whether KV caching is enabled.
            tie_word_embeddings: Whether input/output embeddings share weights.
            rope_theta: Base frequency for rotary positional embeddings.
            use_sliding_window: Whether sliding-window attention is enabled.
            sliding_window: Size of the sliding attention window.
            max_window_layers: Layer index threshold for sliding window.
            attention_dropout: Dropout rate for attention weights.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            rope_parameters: Alternative key for RoPE scaling parameters.
            layer_types: Per-layer attention type strings. Auto-generated
                from sliding window settings when ``None``.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.layer_types = layer_types

    def finalize(self) -> None:
        """Resolve derived attributes and fill in defaults.

        Computes ``num_key_value_heads``, normalizes ``rope_scaling``, and
        auto-generates ``layer_types`` based on sliding window settings.
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        rope_scaling = self.rope_scaling or self.rope_parameters
        if rope_scaling is not None:
            rtype = rope_scaling.get("type") or rope_scaling.get("rope_type", "default")
            if "mrope_section" in rope_scaling or rtype == "mrope":
                rope_scaling["type"] = "mrope"
                rope_scaling["rope_type"] = "mrope"
            elif "type" in rope_scaling and "rope_type" not in rope_scaling:
                rope_scaling["rope_type"] = rope_scaling["type"]
        self.rope_scaling = rope_scaling
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.use_sliding_window and i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class Qwen2VLModelConfig(EasyMLXBaseConfig):
    """Fully-resolved runtime configuration for the Qwen2-VL model.

    Bundles the finalized text and vision configs together with special
    token IDs, ready for direct consumption by model layers.

    Attributes:
        text_config: Finalized text decoder configuration.
        vision_config: Finalized vision encoder configuration.
        model_type: Identifier string for the composite model.
        vocab_size: Size of the token vocabulary.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking the start of vision content.
        vision_end_token_id: Token ID marking the end of vision content.
        pad_token_id: Padding token ID.
        eos_token_id: End-of-sequence token ID(s).
    """

    def __init__(
        self,
        *,
        text_config: Qwen2VLTextConfig | dict[str, Any] | None = None,
        vision_config: Qwen2VLVisionConfig | dict[str, Any] | None = None,
        vocab_size: int = 152064,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        pad_token_id: int = 0,
        eos_token_id: list[int] | None = None,
        **kwargs,
    ):
        """Initialize the fully-resolved Qwen2-VL runtime model configuration.

        Args:
            text_config: Text decoder configuration. Accepts a
                ``Qwen2VLTextConfig`` instance or a dict of keyword arguments.
            vision_config: Vision encoder configuration. Accepts a
                ``Qwen2VLVisionConfig`` instance or a dict of keyword arguments.
            vocab_size: Size of the token vocabulary.
            image_token_id: Token ID for image placeholders.
            video_token_id: Token ID for video placeholders.
            vision_start_token_id: Token ID marking vision content start.
            vision_end_token_id: Token ID marking vision content end.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(**kwargs)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Qwen2VLTextConfig(**text_config)
        if vision_config is not None and isinstance(vision_config, dict):
            vision_config = Qwen2VLVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.pad_token_id = pad_token_id
        if eos_token_id is None and self.text_config is not None:
            eos_token_id = getattr(self.text_config, "eos_token_id", None)
        self.eos_token_id = eos_token_id


@register_config("qwen2_vl")
class Qwen2VLConfig(EasyMLXBaseConfig):
    """Top-level configuration for the Qwen2-VL multimodal model.

    Registered with the EasyMLX factory under the ``"qwen2_vl"`` model type.
    Accepts raw dictionaries or pre-built sub-config classes for both
    the vision and text components.

    Attributes:
        model_type: The model type identifier (``"qwen2_vl"``).
        text_config: Finalized text decoder configuration.
        vision_config: Vision encoder configuration.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking the start of vision content.
        vision_end_token_id: Token ID marking the end of vision content.
    """

    model_type = "qwen2_vl"

    def __init__(
        self,
        *,
        text_config: Qwen2VLTextConfig | dict[str, Any] | None = None,
        vision_config: Qwen2VLVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        pad_token_id: int = 0,
        eos_token_id: list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the Qwen2-VL configuration.

        Args:
            text_config: Text decoder configuration. Accepts a
                ``Qwen2VLTextConfig`` instance or a dict of keyword arguments.
                Defaults to ``Qwen2VLTextConfig()`` when ``None``.
            vision_config: Vision encoder configuration. Accepts a
                ``Qwen2VLVisionConfig`` instance or a dict of keyword arguments.
                Defaults to ``Qwen2VLVisionConfig()`` when ``None``.
            image_token_id: Token ID for image placeholders.
            video_token_id: Token ID for video placeholders.
            vision_start_token_id: Token ID marking vision content start.
            vision_end_token_id: Token ID marking vision content end.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if vision_config is None:
            vision_cfg = Qwen2VLVisionConfig()
        elif isinstance(vision_config, Qwen2VLVisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Qwen2VLVisionConfig(**vision_config)

        if text_config is None:
            text_cfg = Qwen2VLTextConfig()
        elif isinstance(text_config, Qwen2VLTextConfig):
            text_cfg = text_config
        else:
            text_cfg = Qwen2VLTextConfig(**text_config)

        text_cfg.finalize()

        self.text_config = text_cfg
        self.vision_config = vision_cfg
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_model_config(self) -> Qwen2VLModelConfig:
        """Build a fully-resolved ``Qwen2VLModelConfig`` for runtime use.

        Returns:
            A ``Qwen2VLModelConfig`` with finalized text and vision sub-configs.
        """
        return Qwen2VLModelConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            model_type=self.model_type,
            vocab_size=self.text_config.vocab_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )


__all__ = ("Qwen2VLConfig", "Qwen2VLModelConfig", "Qwen2VLTextConfig", "Qwen2VLVisionConfig")
