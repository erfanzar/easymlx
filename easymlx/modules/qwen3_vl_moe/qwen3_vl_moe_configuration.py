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

"""Qwen3-VL-MoE configuration (serving/inference only)."""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


class Qwen3VLMoeVisionConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3-VL-MoE vision encoder.

    Attributes:
        depth: Number of transformer blocks in the vision encoder.
        hidden_size: Hidden dimensionality of vision transformer layers.
        hidden_act: Activation function name for the vision MLP layers.
        intermediate_size: Dimensionality of the feed-forward hidden layer.
        num_heads: Number of attention heads in each vision block.
        in_channels: Number of input image channels (e.g., 3 for RGB).
        patch_size: Size of each image patch for the patch embedding.
        spatial_merge_size: Factor for spatial merging of vision tokens.
        temporal_patch_size: Patch size along the temporal axis for video.
        out_hidden_size: Output projection dimensionality after the vision encoder.
        num_position_embeddings: Maximum number of position embeddings.
        deepstack_visual_indexes: Layer indices for deep-stack visual features.
        tokens_per_second: Number of tokens generated per second for video.
        initializer_range: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        *,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        tokens_per_second: float = 2.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.tokens_per_second = tokens_per_second
        self.initializer_range = initializer_range


class Qwen3VLMoeTextConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3-VL-MoE text decoder.

    Extends the standard Qwen3-VL text config with Mixture-of-Experts
    parameters including expert count, routing, and sparse layer scheduling.

    Attributes:
        vocab_size: Size of the token vocabulary.
        hidden_size: Hidden dimensionality of transformer layers.
        intermediate_size: Dimensionality of the dense MLP hidden layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for grouped-query attention.
        head_dim: Dimensionality of each attention head (computed if None).
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length for position embeddings.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm layers.
        use_cache: Whether to use KV caching during generation.
        tie_word_embeddings: Whether input and output embeddings share weights.
        rope_theta: Base frequency for rotary position embeddings.
        attention_bias: Whether attention projections include bias terms.
        attention_dropout: Dropout probability for attention weights.
        rope_scaling: Optional dictionary configuring RoPE scaling.
        use_sliding_window: Whether to enable sliding window attention.
        sliding_window: Size of the sliding window (in tokens).
        max_window_layers: Layer index threshold above which sliding window is used.
        decoder_sparse_step: Interval at which MoE layers replace dense MLPs.
        moe_intermediate_size: Intermediate size for each MoE expert.
        num_experts_per_tok: Number of experts activated per token (top-k).
        num_experts: Total number of experts in the MoE layers.
        norm_topk_prob: Whether to normalize the top-k routing probabilities.
        output_router_logits: Whether to output router logits for auxiliary losses.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        mlp_only_layers: Layer indices that always use a dense MLP (no MoE).
        layer_types: Per-layer attention type list (``"full_attention"`` or
            ``"sliding_attention"``).
    """

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_scaling: dict | None = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 80,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 1408,
        num_experts_per_tok: int = 4,
        num_experts: int = 60,
        norm_topk_prob: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
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
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers
        self.layer_types = layer_types

    def finalize(self) -> None:
        """Finalize configuration by computing derived values.

        Computes ``head_dim`` from ``hidden_size`` and ``num_attention_heads``
        if not explicitly set. Initializes ``mlp_only_layers`` to an empty list
        and builds ``layer_types`` based on sliding window settings.
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.use_sliding_window and i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


@register_config("qwen3_vl_moe")
class Qwen3VLMoeConfig(EasyMLXBaseConfig):
    """Top-level configuration for the Qwen3-VL-MoE vision-language model.

    Composes a ``Qwen3VLMoeVisionConfig`` for the vision encoder and a
    ``Qwen3VLMoeTextConfig`` for the MoE text decoder. Sub-configs can be
    passed as class instances or plain dictionaries.

    Attributes:
        model_type: Identifier string (``"qwen3_vl_moe"``).
        vision_config: Vision encoder sub-configuration.
        text_config: MoE text decoder sub-configuration.
        image_token_id: Token ID used for image placeholders in the input.
        video_token_id: Token ID used for video placeholders in the input.
        vision_start_token_id: Token ID marking the start of vision tokens.
        vision_end_token_id: Token ID marking the end of vision tokens.
    """

    model_type = "qwen3_vl_moe"

    def __init__(
        self,
        *,
        vision_config: Qwen3VLMoeVisionConfig | dict[str, Any] | None = None,
        text_config: Qwen3VLMoeTextConfig | dict[str, Any] | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if vision_config is None:
            vision_cfg = Qwen3VLMoeVisionConfig()
        elif isinstance(vision_config, Qwen3VLMoeVisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Qwen3VLMoeVisionConfig(**vision_config)

        if text_config is None:
            text_cfg = Qwen3VLMoeTextConfig()
        elif isinstance(text_config, Qwen3VLMoeTextConfig):
            text_cfg = text_config
        else:
            text_cfg = Qwen3VLMoeTextConfig(**text_config)

        text_cfg.finalize()

        self.vision_config = vision_cfg
        self.text_config = text_cfg
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ("Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig", "Qwen3VLMoeVisionConfig")
