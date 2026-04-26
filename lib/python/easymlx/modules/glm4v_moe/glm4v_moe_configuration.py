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

"""GLM-4V MoE configuration (serving/inference only).

This module defines configuration classes and the main config class
for the GLM-4V MoE vision-language model, which combines a vision encoder
with a Mixture-of-Experts language backbone.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.glm4v.glm4v_configuration import Glm4VVisionConfig


class Glm4VMoeVisionConfig(Glm4VVisionConfig):
    """Vision configuration for the GLM-4V MoE model.

    Extends ``Glm4VVisionConfig`` with default values specific to the MoE
    variant.

    Attributes:
        model_type: Vision model type identifier.
        depth: Number of vision transformer blocks.
        hidden_size: Vision encoder hidden dimensionality.
        intermediate_size: Vision MLP intermediate dimensionality.
        num_heads: Number of vision attention heads.
        patch_size: Spatial patch size.
    """

    def __init__(
        self,
        *,
        depth: int = 24,
        hidden_size: int = 1536,
        intermediate_size: int = 13696,
        num_heads: int = 12,
        patch_size: int = 14,
        **kwargs,
    ):
        """Initializes the GLM-4V MoE vision config with MoE-specific defaults.

        Args:
            depth: Number of vision transformer blocks. Defaults to 24.
            hidden_size: Vision encoder hidden dimensionality. Defaults to 1536.
            intermediate_size: Vision MLP intermediate dimensionality. Defaults to 13696.
            num_heads: Number of vision attention heads. Defaults to 12.
            patch_size: Spatial patch size. Defaults to 14.
            **kwargs: Additional keyword arguments forwarded to ``Glm4VVisionConfig``.
        """
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            patch_size=patch_size,
            **kwargs,
        )


class Glm4VMoeTextConfig(EasyMLXBaseConfig):
    """Text (language) configuration for the GLM-4V MoE model.

    Defines hyperparameters for the MoE language backbone including
    attention, feed-forward, and expert routing parameters.

    Attributes:
        model_type: Text model type identifier.
        vocab_size: Vocabulary size.
        hidden_size: Hidden state dimensionality.
        intermediate_size: Dense MLP intermediate size.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        use_cache: Whether to use KV caching.
        attention_bias: Whether attention includes bias.
        attention_dropout: Attention dropout rate.
        partial_rotary_factor: Fraction of head dim for RoPE.
        rope_theta: RoPE base frequency.
        rope_scaling: RoPE scaling configuration.
        moe_intermediate_size: Expert intermediate size.
        num_experts_per_tok: Experts activated per token.
        n_shared_experts: Shared (always-active) expert count.
        n_routed_experts: Total routed expert count.
        routed_scaling_factor: Expert score scaling.
        n_group: Expert group count for routing.
        topk_group: Top groups to keep during routing.
        first_k_dense_replace: Dense layers before MoE.
        norm_topk_prob: Normalize routing probabilities.
        use_qk_norm: Apply QK normalization.
        tie_word_embeddings: Tie input/output embeddings.
        scoring_func: Scoring function for routing.
        topk_method: Top-k routing method.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 151424,
        hidden_size: int = 4096,
        intermediate_size: int = 10944,
        num_hidden_layers: int = 46,
        num_attention_heads: int = 96,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 65536,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        moe_intermediate_size: int = 1408,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 128,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = False,
        tie_word_embeddings: bool | None = None,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        """Initializes a GLM-4V MoE text configuration.

        Args:
            vocab_size: Vocabulary size. Defaults to 151424.
            hidden_size: Hidden state dimensionality. Defaults to 4096.
            intermediate_size: Dense MLP intermediate size. Defaults to 10944.
            num_hidden_layers: Number of decoder layers. Defaults to 46.
            num_attention_heads: Number of attention heads. Defaults to 96.
            num_key_value_heads: KV heads for GQA. Defaults to 8.
            head_dim: Per-head dim. Defaults to 128.
            hidden_act: Activation function. Defaults to ``"silu"``.
            max_position_embeddings: Max sequence length. Defaults to 65536.
            initializer_range: Init range. Defaults to 0.02.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            use_cache: Enable KV caching. Defaults to True.
            attention_bias: Include attention bias. Defaults to True.
            attention_dropout: Attention dropout rate. Defaults to 0.0.
            partial_rotary_factor: Fraction of head dim for RoPE. Defaults to 0.5.
            rope_theta: RoPE base frequency. Defaults to 10000.0.
            rope_scaling: RoPE scaling config. Defaults to M-RoPE with
                sections ``[64, 32, 32]``.
            moe_intermediate_size: Expert intermediate size. Defaults to 1408.
            num_experts_per_tok: Experts per token. Defaults to 8.
            n_shared_experts: Shared expert count. Defaults to 1.
            n_routed_experts: Routed expert count. Defaults to 128.
            routed_scaling_factor: Expert score scaling. Defaults to 1.0.
            n_group: Expert group count. Defaults to 1.
            topk_group: Top groups to keep. Defaults to 1.
            first_k_dense_replace: Dense layers before MoE. Defaults to 1.
            norm_topk_prob: Normalize routing probs. Defaults to True.
            use_qk_norm: Apply QK normalization. Defaults to False.
            tie_word_embeddings: Tie embeddings. Defaults to None.
            scoring_func: Scoring function for routing. Defaults to ``"sigmoid"``.
            topk_method: Top-k routing method. Defaults to ``"noaux_tc"``.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.rope_scaling = (
            rope_scaling if rope_scaling is not None else {"type": "default", "mrope_section": [64, 32, 32]}
        )
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.scoring_func = scoring_func
        self.topk_method = topk_method


class Glm4VMoeModelConfig(EasyMLXBaseConfig):
    """Composite configuration for the full GLM-4V MoE model.

    Combines text and vision configurations with model-level parameters
    including special token IDs.

    Attributes:
        text_config: MoE text sub-model configuration.
        vision_config: Vision sub-model configuration.
        model_type: Model type identifier.
        vocab_size: Vocabulary size.
        ignore_index: Loss ignore index.
        image_token_index: Image token index.
        image_token_id: Image token ID.
        video_token_index: Video token index.
        video_token_id: Video token ID.
        vision_start_token_id: Vision start token ID.
        vision_end_token_id: Vision end token ID.
        image_start_token_id: Image start token ID.
        image_end_token_id: Image end token ID.
        video_start_token_id: Video start token ID.
        video_end_token_id: Video end token ID.
        hidden_size: Model hidden state dimensionality.
        pad_token_id: Padding token ID.
        eos_token_id: End-of-sequence token ID(s).
    """

    def __init__(
        self,
        *,
        text_config: Glm4VMoeTextConfig | dict[str, Any] | None = None,
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
        """Initializes the composite GLM-4V MoE model config.

        Args:
            text_config: MoE text sub-configuration. Accepts a
                ``Glm4VMoeTextConfig``, dictionary, or None.
            vision_config: Vision sub-configuration. Accepts a
                ``Glm4VVisionConfig``, dictionary, or None.
            vocab_size: Vocabulary size. Defaults to 257152.
            ignore_index: Loss ignore index. Defaults to -100.
            image_token_index: Image token index. Defaults to 151363.
            image_token_id: Image token ID. Defaults to 151363.
            video_token_index: Video token index. Defaults to 151364.
            video_token_id: Video token ID. Defaults to 151364.
            vision_start_token_id: Vision start token. Defaults to 151339.
            vision_end_token_id: Vision end token. Defaults to 151340.
            image_start_token_id: Image start token. Defaults to None (auto).
            image_end_token_id: Image end token. Defaults to None (auto).
            video_start_token_id: Video start token. Defaults to None (auto).
            video_end_token_id: Video end token. Defaults to None (auto).
            hidden_size: Hidden state dim. Defaults to 2048.
            pad_token_id: Padding token ID. Defaults to 0.
            eos_token_id: EOS token ID(s). Defaults to
                ``[151329, 151336, 151338]``.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Glm4VMoeTextConfig(**text_config)
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

        if eos_token_id is None:
            eos_token_id = [151329, 151336, 151338]
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


@register_config("glm4v_moe")
class Glm4VMoeConfig(EasyMLXBaseConfig):
    """Configuration for the GLM-4V MoE vision-language model.

    Extends ``EasyMLXBaseConfig`` with text (MoE) and vision sub-configs.
    Registered under model type ``"glm4v_moe"``.

    Attributes:
        model_type: Identifier string (``"glm4v_moe"``).
        text_config: Serialized text sub-configuration dictionary.
        vision_config: Serialized vision sub-configuration dictionary.
        vocab_size: Vocabulary size.
        ignore_index: Loss ignore index.
        image_token_index: Image token index.
        image_token_id: Image token ID.
        video_token_index: Video token index.
        video_token_id: Video token ID.
        vision_start_token_id: Vision start token ID.
        vision_end_token_id: Vision end token ID.
        hidden_size: Hidden state dimensionality.
    """

    model_type = "glm4v_moe"

    def __init__(
        self,
        *,
        text_config: Glm4VMoeTextConfig | dict[str, Any] | None = None,
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
        """Initializes a GLM-4V MoE configuration.

        Args:
            text_config: Text sub-configuration. Accepts a
                ``Glm4VMoeTextConfig``, dictionary, or None for defaults.
            vision_config: Vision sub-configuration. Accepts a
                ``Glm4VVisionConfig``, ``Glm4VMoeVisionConfig``,
                dictionary, or None for defaults.
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
            eos_token_id: EOS token ID(s). Defaults to None.
            tie_word_embeddings: Tie embeddings. Defaults to False.
            **kwargs: Additional keyword arguments for the base class.
        """
        if text_config is None:
            text_cfg = Glm4VMoeTextConfig()
        elif isinstance(text_config, Glm4VMoeTextConfig):
            text_cfg = text_config
        else:
            text_cfg = Glm4VMoeTextConfig(**text_config)

        if vision_config is None:
            vision_cfg = Glm4VMoeVisionConfig(
                depth=24,
                hidden_size=1536,
                intermediate_size=13696,
                num_heads=12,
                patch_size=14,
            )
        elif isinstance(vision_config, Glm4VMoeVisionConfig):
            vision_cfg = vision_config
        elif isinstance(vision_config, Glm4VVisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Glm4VMoeVisionConfig(**vision_config)

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

    def to_model_config(self) -> Glm4VMoeModelConfig:
        """Converts this config into a ``Glm4VMoeModelConfig`` instance.

        Returns:
            A ``Glm4VMoeModelConfig`` populated from this configuration.
        """
        text_cfg = Glm4VMoeTextConfig(**self.text_config)
        vision_cfg = Glm4VMoeVisionConfig(**self.vision_config)
        return Glm4VMoeModelConfig(
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

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> Glm4VMoeTextConfig:
        """Retrieves the text sub-configuration as a ``Glm4VMoeTextConfig``.

        Args:
            decoder: Unused. Present for API compatibility.
            encoder: Unused. Present for API compatibility.

        Returns:
            The text configuration instance.
        """
        del decoder, encoder
        text_cfg = self.text_config
        if isinstance(text_cfg, dict):
            text_cfg = Glm4VMoeTextConfig(**text_cfg)
        return text_cfg


Glm4vMoeVisionConfig = Glm4VMoeVisionConfig
Glm4vMoeTextConfig = Glm4VMoeTextConfig
Glm4vMoeModelConfig = Glm4VMoeModelConfig
Glm4vMoeConfig = Glm4VMoeConfig

__all__ = (
    "Glm4VMoeConfig",
    "Glm4VMoeModelConfig",
    "Glm4VMoeTextConfig",
    "Glm4VMoeVisionConfig",
    "Glm4vMoeConfig",
    "Glm4vMoeModelConfig",
    "Glm4vMoeTextConfig",
    "Glm4vMoeVisionConfig",
)
