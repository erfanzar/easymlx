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

"""Qwen3.5 configuration for serving and inference.

This module defines configuration classes for the Qwen3.5 vision-language
model. The composite ``Qwen3_5Config`` combines a ``Qwen3_5TextConfig``
(extending Qwen3NextConfig) for the hybrid attention text backbone and a
``Qwen3_5VisionConfig`` for the vision encoder.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
from easymlx.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLVisionConfig


class Qwen3_5TextConfig(Qwen3NextConfig):
    """Configuration for the Qwen3.5 text backbone.

    Extends ``Qwen3NextConfig`` with Qwen3.5-specific defaults. The text
    model is dense (no MoE FFN layers) and uses a hybrid mix of full
    softmax attention and linear attention with gated delta rule.

    Attributes:
        model_type: The model type identifier (``"qwen3_5_text"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dimensionality of the dense MLP intermediate layer.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads (GQA).
        head_dim: Dimensionality of each attention head.
        hidden_act: Activation function in the MLP layers.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMS normalisation layers.
        rope_theta: Base period for rotary position embeddings.
        partial_rotary_factor: Fraction of head dimension using rotary embeddings.
        full_attention_interval: Interval between full-attention layers.
        linear_conv_kernel_dim: Kernel size for the convolutional component in linear attention.
        linear_key_head_dim: Key head dimension for linear attention layers.
        linear_value_head_dim: Value head dimension for linear attention layers.
        linear_num_key_heads: Number of key heads in linear attention layers.
        linear_num_value_heads: Number of value heads in linear attention layers.
    """

    model_type = "qwen3_5_text"

    def __init__(
        self,
        *,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float | None = None,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        partial_rotary_factor: float = 0.25,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Qwen3.5 text config with hybrid attention parameters.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of the transformer hidden states.
            intermediate_size: Dense MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query heads for full attention.
            num_key_value_heads: Number of key/value heads for GQA.
            head_dim: Dimensionality per attention head.
            hidden_act: Activation function name.
            max_position_embeddings: Maximum sequence length.
            initializer_range: Standard deviation for weight initialization.
            rms_norm_eps: Epsilon for RMS normalization.
            use_cache: Whether KV caching is enabled.
            tie_word_embeddings: Whether to tie input/output embeddings.
            rope_theta: Base frequency for RoPE. Resolved from
                ``rope_scaling``/``rope_parameters`` if not set directly;
                falls back to 10000.0.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            rope_parameters: Alias for ``rope_scaling`` (HF interop).
            attention_bias: Whether to use bias in attention projections.
            attention_dropout: Dropout rate for attention weights.
            partial_rotary_factor: Fraction of head_dim for partial RoPE.
            layer_types: Per-layer attention type strings. Auto-generated
                based on ``full_attention_interval`` when ``None``.
            full_attention_interval: Interval for full attention layers.
            linear_conv_kernel_dim: Conv1d kernel size in linear attention.
            linear_key_head_dim: Key head dim for linear attention.
            linear_value_head_dim: Value head dim for linear attention.
            linear_num_key_heads: Number of key heads in linear attention.
            linear_num_value_heads: Number of value heads in linear attention.
            decoder_sparse_step: Layer interval for MoE routing.
            moe_intermediate_size: Per-expert intermediate dimensionality.
            shared_expert_intermediate_size: Shared expert intermediate size.
            num_experts_per_tok: Number of experts activated per token.
            num_experts: Total number of routing experts.
            norm_topk_prob: Whether to normalize top-k routing probabilities.
            output_router_logits: Whether to return router logits.
            router_aux_loss_coef: Router auxiliary loss coefficient.
            mlp_only_layers: Indices of layers that skip MoE. Defaults to
                all layers (Qwen3.5 text is dense).
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        effective_rope_scaling = rope_scaling or rope_parameters
        if rope_theta is None and isinstance(effective_rope_scaling, dict):
            rope_theta = effective_rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0

        if mlp_only_layers is None or len(mlp_only_layers) == 0:
            mlp_only_layers = list(range(num_hidden_layers))

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=effective_rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            partial_rotary_factor=partial_rotary_factor,
            layer_types=layer_types,
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            decoder_sparse_step=decoder_sparse_step,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            norm_topk_prob=norm_topk_prob,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            mlp_only_layers=mlp_only_layers,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Qwen3.5 text checkpoints are dense (no MoE FFN layers).

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            Always ``False`` for Qwen3.5 text.
        """
        return False


class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    """Configuration for the Qwen3.5 vision encoder.

    Inherits from ``Qwen3VLVisionConfig``. Qwen3.5 does not use
    deepstack mergers, so ``deepstack_visual_indexes`` defaults to
    an empty list.

    Attributes:
        model_type: The model type identifier (``"qwen3_5"``).
        deepstack_visual_indexes: Indices of vision transformer layers
            whose outputs are used as deepstack embeddings.
    """

    model_type = "qwen3_5"

    def __init__(
        self,
        *,
        deepstack_visual_indexes: list[int] | None = None,
        vision_attn_bias: bool = True,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2048,
        spatial_merge_size: int = 2,
        **kwargs,
    ):
        """Initialize Qwen3.5 vision config.

        Args:
            deepstack_visual_indexes: Layer indices for deep-stack visual
                features. Defaults to an empty list for Qwen3.5.
            vision_attn_bias: Whether to use bias in vision attention QKV and
                projection layers. Defaults to True for Qwen3.5.
            num_position_embeddings: Number of learned position embeddings.
            out_hidden_size: Output hidden size after the patch merger.
            spatial_merge_size: Spatial merge factor for the patch merger.
            **kwargs: Forwarded to ``Qwen3VLVisionConfig``.
        """
        requested_indexes = [] if deepstack_visual_indexes is None else list(deepstack_visual_indexes)
        super().__init__(deepstack_visual_indexes=requested_indexes or [0], **kwargs)
        self.vision_attn_bias = vision_attn_bias
        self.deepstack_visual_indexes = requested_indexes
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size


@register_config("qwen3_5")
class Qwen3_5Config(EasyMLXBaseConfig):
    """Configuration for the Qwen3.5 multimodal (vision-language) model.

    Composes a ``Qwen3_5TextConfig`` for the language backbone and a
    ``Qwen3_5VisionConfig`` for the vision encoder. Registered with the
    EasyMLX factory under the ``"qwen3_5"`` model type.

    Attributes:
        model_type: The model type identifier (``"qwen3_5"``).
        text_config: Text model sub-configuration.
        vision_config: Vision encoder sub-configuration.
        image_token_id: Token ID used to represent image placeholders.
        video_token_id: Token ID used to represent video placeholders.
        vision_start_token_id: Token ID marking the start of a vision span.
        vision_end_token_id: Token ID marking the end of a vision span.
    """

    model_type = "qwen3_5"

    def __init__(
        self,
        *,
        text_config: Qwen3_5TextConfig | dict[str, Any] | None = None,
        vision_config: Qwen3_5VisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize Qwen3.5 composite configuration.

        Args:
            text_config: Text model sub-configuration. Instantiated from
                a dict if needed.
            vision_config: Vision encoder sub-configuration. Instantiated
                from a dict if needed.
            image_token_id: Token ID for image placeholders in the input.
            video_token_id: Token ID for video placeholders in the input.
            vision_start_token_id: Token ID marking vision span start.
            vision_end_token_id: Token ID marking vision span end.
            tie_word_embeddings: Whether to tie input and output embedding
                weights.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if text_config is None:
            self.text_config = Qwen3_5TextConfig()
        elif isinstance(text_config, Qwen3_5TextConfig):
            self.text_config = text_config
        else:
            self.text_config = Qwen3_5TextConfig(**text_config)

        if vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        elif isinstance(vision_config, Qwen3_5VisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = Qwen3_5VisionConfig(**vision_config)

        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_hidden_layers(self) -> int:
        """Proxy to ``text_config.num_hidden_layers``."""
        return self.text_config.num_hidden_layers

    @property
    def layer_types(self) -> list[str] | None:
        """Proxy to ``text_config.layer_types``."""
        return getattr(self.text_config, "layer_types", None)

    @property
    def num_attention_heads(self) -> int:
        """Proxy to ``text_config.num_attention_heads``."""
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """Proxy to ``text_config.num_key_value_heads``."""
        return self.text_config.num_key_value_heads

    @property
    def hidden_size(self) -> int:
        """Proxy to ``text_config.hidden_size``."""
        return self.text_config.hidden_size

    @property
    def head_dim(self) -> int:
        """Proxy to ``text_config.head_dim``."""
        return getattr(self.text_config, "head_dim", self.hidden_size // self.num_attention_heads)

    @property
    def vocab_size(self) -> int:
        """Proxy to ``text_config.vocab_size``."""
        return self.text_config.vocab_size

    @property
    def attn_mechanism(self):
        """Proxy to ``text_config.attn_mechanism``."""
        return getattr(self, "_attn_mechanism_override", None) or getattr(self.text_config, "attn_mechanism", None)

    @attn_mechanism.setter
    def attn_mechanism(self, value):
        """Allow base class to set attn_mechanism."""
        self._attn_mechanism_override = value

    def get_text_config(self):
        """Return the text sub-configuration."""
        return self.text_config


__all__ = (
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
)
