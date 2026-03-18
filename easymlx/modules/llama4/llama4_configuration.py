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

"""Llama4 configuration classes for serving and inference.

This module defines the hierarchical configuration classes for the Llama4
multimodal model, covering vision encoder, text decoder, and the top-level
composite configuration registered with the EasyMLX factory.
"""

from __future__ import annotations

from typing import Any, ClassVar

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


class Llama4VisionConfig(EasyMLXBaseConfig):
    """Configuration for the Llama4 vision encoder.

    Attributes:
        model_type: Identifier string for the vision sub-model.
        hidden_size: Dimensionality of the vision transformer hidden states.
        hidden_act: Activation function name used in the vision MLP layers.
        num_hidden_layers: Number of vision transformer blocks.
        num_attention_heads: Number of attention heads in each vision block.
        num_channels: Number of input image channels (e.g. 3 for RGB).
        intermediate_size: Dimensionality of the vision MLP intermediate layer.
        vision_output_dim: Dimensionality of the vision encoder output projection.
        image_size: Expected input image resolution (height and width in pixels).
        patch_size: Size of each image patch for the patch embedding convolution.
        norm_eps: Epsilon for layer normalization in vision blocks.
        vision_feature_layer: Index of the vision layer whose output is used as features.
        vision_feature_select_strategy: Strategy for selecting vision features.
        initializer_range: Standard deviation for weight initialization.
        pixel_shuffle_ratio: Ratio for pixel shuffle operations.
        projector_input_dim: Input dimensionality of the multi-modal projector.
        projector_output_dim: Output dimensionality of the multi-modal projector.
        multi_modal_projector_bias: Whether to use bias in the multi-modal projector.
        projector_dropout: Dropout rate in the multi-modal projector.
        attention_dropout: Dropout rate for attention weights.
        rope_theta: Base frequency for rotary positional embeddings.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 34,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5632,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "default",
        initializer_range: float = 0.02,
        pixel_shuffle_ratio: float = 0.5,
        projector_input_dim: int = 4096,
        projector_output_dim: int = 4096,
        multi_modal_projector_bias: bool = False,
        projector_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.vision_output_dim = vision_output_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.initializer_range = initializer_range
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.projector_input_dim = projector_input_dim
        self.projector_output_dim = projector_output_dim
        self.multi_modal_projector_bias = multi_modal_projector_bias
        self.projector_dropout = projector_dropout
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta


class Llama4TextConfig(EasyMLXBaseConfig):
    """Configuration for the Llama4 text decoder.

    Holds all hyperparameters governing the text-side transformer, including
    MoE routing, chunked attention, and RoPE settings.

    Attributes:
        model_type: Identifier string for the text sub-model.
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the text transformer hidden states.
        intermediate_size: MoE expert intermediate dimensionality.
        intermediate_size_mlp: Dense MLP intermediate dimensionality.
        num_hidden_layers: Total number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for grouped-query attention.
        head_dim: Dimensionality of each attention head.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length supported by RoPE.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled during generation.
        pad_token_id: Token ID used for padding.
        bos_token_id: Token ID for beginning-of-sequence.
        eos_token_id: Token ID for end-of-sequence.
        tie_word_embeddings: Whether input/output embeddings share weights.
        rope_theta: Base frequency for rotary positional embeddings.
        attention_dropout: Dropout rate for attention weights.
        num_experts_per_tok: Number of experts activated per token in MoE layers.
        num_local_experts: Total number of experts in MoE layers.
        moe_layers: List of layer indices that use MoE instead of dense MLP.
        interleave_moe_layer_step: Step interval for interleaving MoE layers.
        use_qk_norm: Whether to apply RMS normalization to Q and K projections.
        output_router_logits: Whether to return router logits in the output.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        router_jitter_noise: Jitter noise added to router inputs during training.
        rope_scaling: Optional dictionary specifying RoPE scaling parameters.
        no_rope_layers: Per-layer flags indicating which layers skip RoPE.
        no_rope_layer_interval: Interval for determining no-RoPE layers.
        attention_chunk_size: Window size for chunked attention layers.
        attn_temperature_tuning: Temperature tuning factor for attention.
        floor_scale: Floor scale parameter for attention temperature.
        attn_scale: Attention scaling factor.
        layer_types: Per-layer attention type strings (e.g. ``"chunked_attention"``).
        attention_bias: Whether to use bias in attention projections.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 202048,
        hidden_size: int = 5120,
        intermediate_size: int = 8192,
        intermediate_size_mlp: int = 16384,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 40,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 1,
        num_local_experts: int = 16,
        moe_layers: list[int] | None = None,
        interleave_moe_layer_step: int = 1,
        use_qk_norm: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        rope_scaling: dict[str, Any] | None = None,
        no_rope_layers: list[int] | None = None,
        no_rope_layer_interval: int = 4,
        attention_chunk_size: int = 8192,
        attn_temperature_tuning: int = 4,
        floor_scale: int = 8192,
        attn_scale: float = 0.1,
        layer_types: list[str] | None = None,
        attention_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.moe_layers = moe_layers
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.use_qk_norm = use_qk_norm
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.rope_scaling = rope_scaling
        self.no_rope_layers = no_rope_layers
        self.no_rope_layer_interval = no_rope_layer_interval
        self.attention_chunk_size = attention_chunk_size
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.layer_types = layer_types
        self.attention_bias = attention_bias

    def finalize(self) -> None:
        """Resolve derived attributes and fill in defaults.

        Computes ``num_key_value_heads``, ``head_dim``, ``no_rope_layers``,
        ``layer_types``, ``moe_layers``, and normalizes ``rope_scaling`` if
        they have not been explicitly provided.
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((layer_idx + 1) % self.no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
            ]

        if self.layer_types is None:
            self.layer_types = ["chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers]

        if self.moe_layers is None:
            step = max(1, int(self.interleave_moe_layer_step))
            self.moe_layers = list(range(step - 1, self.num_hidden_layers, step))

        if self.rope_scaling is not None and "type" in self.rope_scaling and "rope_type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]


class Llama4ModelConfig(EasyMLXBaseConfig):
    """Fully-resolved runtime configuration for the Llama4 model.

    This class bundles the finalized text and vision configs together
    with special token indices, ready for direct consumption by model layers.

    Attributes:
        text_config: Finalized text decoder configuration.
        vision_config: Finalized vision encoder configuration.
        model_type: Identifier string for the composite model.
        boi_token_index: Token index for beginning-of-image.
        eoi_token_index: Token index for end-of-image.
        image_token_index: Token index representing an image placeholder.
        boi_token_id: Token ID for beginning-of-image.
        eoi_token_id: Token ID for end-of-image.
        image_token_id: Token ID for image placeholder.
        pad_token_id: Optional padding token ID.
        eos_token_id: Optional end-of-sequence token ID.
    """

    def __init__(
        self,
        *,
        text_config: Llama4TextConfig | dict[str, Any] | None = None,
        vision_config: Llama4VisionConfig | dict[str, Any] | None = None,
        boi_token_index: int = 0,
        eoi_token_index: int = 0,
        image_token_index: int = 0,
        boi_token_id: int = 0,
        eoi_token_id: int = 0,
        image_token_id: int = 0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Llama4TextConfig(**text_config)
        if vision_config is not None and isinstance(vision_config, dict):
            vision_config = Llama4VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


@register_config("llama4")
class Llama4Config(EasyMLXBaseConfig):
    """Top-level configuration for the Llama4 multimodal model.

    Registered with the EasyMLX factory under the ``"llama4"`` model type.
    Accepts raw dictionaries or pre-built sub-config dataclasses for both
    the vision and text components, and handles token ID aliasing.

    Attributes:
        model_type: The model type identifier (``"llama4"``).
        attribute_map: Maps alternative token ID attribute names to canonical ones.
    """

    model_type = "llama4"
    attribute_map: ClassVar[dict[str, str]] = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    def __init__(
        self,
        *,
        vision_config: Llama4VisionConfig | dict[str, Any] | None = None,
        text_config: Llama4TextConfig | dict[str, Any] | None = None,
        boi_token_index: int = 200080,
        eoi_token_index: int = 200081,
        image_token_index: int = 200092,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
        image_token_id: int | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the Llama4 configuration.

        Args:
            vision_config: Vision encoder configuration. Accepts a
                ``Llama4VisionConfig`` instance or a dict of keyword arguments.
                Defaults to ``Llama4VisionConfig()`` when ``None``.
            text_config: Text decoder configuration. Accepts a
                ``Llama4TextConfig`` instance or a dict of keyword arguments.
                Defaults to ``Llama4TextConfig()`` when ``None``.
            boi_token_index: Token index for beginning-of-image.
            eoi_token_index: Token index for end-of-image.
            image_token_index: Token index for the image placeholder.
            boi_token_id: Alias for ``boi_token_index``; takes precedence if set.
            eoi_token_id: Alias for ``eoi_token_index``; takes precedence if set.
            image_token_id: Alias for ``image_token_index``; takes precedence if set.
            pad_token_id: Padding token ID. Falls back to text config default.
            bos_token_id: Beginning-of-sequence token ID.
            eos_token_id: End-of-sequence token ID.
            tie_word_embeddings: Whether to tie input and output embeddings.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if vision_config is None:
            vision_cfg = Llama4VisionConfig()
        elif isinstance(vision_config, Llama4VisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Llama4VisionConfig(**vision_config)

        if text_config is None:
            text_cfg = Llama4TextConfig()
        elif isinstance(text_config, Llama4TextConfig):
            text_cfg = text_config
        else:
            text_cfg = Llama4TextConfig(**text_config)

        text_cfg.finalize()

        if boi_token_id is None:
            boi_token_id = boi_token_index
        else:
            boi_token_index = boi_token_id

        if eoi_token_id is None:
            eoi_token_id = eoi_token_index
        else:
            eoi_token_index = eoi_token_id

        if image_token_id is None:
            image_token_id = image_token_index
        else:
            image_token_index = image_token_id

        self.text_config = text_cfg.to_dict()
        self.vision_config = vision_cfg.to_dict()
        self.boi_token_index = int(boi_token_index)
        self.eoi_token_index = int(eoi_token_index)
        self.image_token_index = int(image_token_index)
        self.boi_token_id = int(boi_token_id)
        self.eoi_token_id = int(eoi_token_id)
        self.image_token_id = int(image_token_id)

        if pad_token_id is None:
            pad_token_id = text_cfg.pad_token_id
        if bos_token_id is None:
            bos_token_id = text_cfg.bos_token_id
        if eos_token_id is None:
            eos_token_id = text_cfg.eos_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_model_config(self) -> Llama4ModelConfig:
        """Build a fully-resolved ``Llama4ModelConfig`` for runtime use.

        Returns:
            A ``Llama4ModelConfig`` with finalized text and vision sub-configs.
        """
        text_cfg = Llama4TextConfig(**self.text_config)
        text_cfg.finalize()
        vision_cfg = Llama4VisionConfig(**self.vision_config)
        return Llama4ModelConfig(
            text_config=text_cfg,
            vision_config=vision_cfg,
            model_type=self.model_type,
            boi_token_index=self.boi_token_index,
            eoi_token_index=self.eoi_token_index,
            image_token_index=self.image_token_index,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            image_token_id=self.image_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> Llama4TextConfig:
        """Retrieve the finalized text decoder configuration.

        Args:
            decoder: Unused; kept for API compatibility.
            encoder: Unused; kept for API compatibility.

        Returns:
            A finalized ``Llama4TextConfig`` instance.
        """
        del decoder, encoder
        text_cfg = self.text_config
        if isinstance(text_cfg, dict):
            text_cfg = Llama4TextConfig(**text_cfg)
        text_cfg.finalize()
        return text_cfg


__all__ = ("Llama4Config", "Llama4ModelConfig", "Llama4TextConfig", "Llama4VisionConfig")
