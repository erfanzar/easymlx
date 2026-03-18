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

"""Configuration for Qwen3.5-MoE text and multimodal models on MLX."""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeVisionConfig


class Qwen3_5MoeTextConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3.5-MoE text backbone.

    Extends the Qwen3-Next hybrid attention architecture with MoE parameters.
    Supports a mix of full softmax attention and linear attention layers,
    partial rotary embeddings, and MoE routing with shared experts.

    Attributes:
        model_type: The model type identifier (``"qwen3_5_moe_text"``).
        vocab_size: Size of the token vocabulary.
        hidden_size: Dimensionality of the transformer hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads (full attention).
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMS normalization.
        use_cache: Whether KV caching is enabled.
        rope_theta: Base frequency for rotary positional embeddings.
        rope_scaling: Optional RoPE scaling configuration dictionary.
        attention_bias: Whether to use bias in attention projections.
        attention_dropout: Dropout rate for attention weights.
        partial_rotary_factor: Fraction of head_dim to apply RoPE to.
        layer_types: Per-layer attention type list.
        full_attention_interval: Interval for inserting full attention layers.
        linear_conv_kernel_dim: Kernel size for the 1-D convolution in linear attention.
        linear_key_head_dim: Key head dimensionality for linear attention.
        linear_value_head_dim: Value head dimensionality for linear attention.
        linear_num_key_heads: Number of key heads in linear attention.
        linear_num_value_heads: Number of value heads in linear attention.
        decoder_sparse_step: Layer interval step for MoE layers.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        shared_expert_intermediate_size: Shared expert intermediate dimensionality.
        num_experts_per_tok: Number of experts activated per token.
        num_experts: Total number of routing experts.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        output_router_logits: Whether to return router logits in the output.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        mlp_only_layers: Layer indices that use dense MLP instead of MoE.
    """

    model_type = "qwen3_5_moe_text"

    def __init__(
        self,
        *,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
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
        linear_attention_separate_proj: bool | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Qwen3.5-MoE text config with hybrid attention and MoE parameters.

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
            rope_theta: Base frequency for RoPE.
            rope_scaling: Optional RoPE scaling configuration dictionary.
            rope_parameters: Alias for ``rope_scaling`` for HF interop.
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
            mlp_only_layers: Indices of layers that skip MoE.
            linear_attention_separate_proj: Whether to use separate linear
                attention projections. Defaults to ``True`` for HF compat.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        # Resolve rope_scaling from rope_parameters alias.
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.partial_rotary_factor = float(partial_rotary_factor)

        self.full_attention_interval = int(full_attention_interval)
        if layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_num_value_heads = int(linear_num_value_heads)

        self.decoder_sparse_step = int(decoder_sparse_step)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_experts = int(num_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.output_router_logits = bool(output_router_logits)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.mlp_only_layers = [] if mlp_only_layers is None else list(mlp_only_layers)

        # HF Qwen3.5 checkpoints expose split linear-attention projections.
        if linear_attention_separate_proj is None:
            self.linear_attention_separate_proj = True
        else:
            self.linear_attention_separate_proj = bool(linear_attention_separate_proj)

        # Mirror HF naming for rope config interop.
        self.rope_parameters = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rotary_dim(self) -> int:
        """Compute the number of dimensions that receive rotary embeddings.

        Returns:
            The rotary embedding dimensionality, derived from ``head_dim``
            and ``partial_rotary_factor``.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_d_inner(self) -> int:
        """Compute the inner dimensionality for the linear attention conv.

        Returns:
            The sum of ``2 * key_dim + value_dim``.
        """
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        return key_dim * 2 + value_dim

    @property
    def linear_d_state(self) -> int:
        """Compute the state dimensionality for linear attention.

        Returns:
            The linear value head dimensionality.
        """
        return int(self.linear_value_head_dim)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Determine whether a given layer uses full softmax attention.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            ``True`` if the layer uses full attention, ``False`` otherwise.
        """
        return self.layer_types[layer_idx] == "full_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Determine whether a given layer uses MoE routing.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            ``True`` if the layer uses MoE routing, ``False`` otherwise.
        """
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % max(self.decoder_sparse_step, 1) == 0


class Qwen3_5MoeVisionConfig(Qwen3VLMoeVisionConfig):
    """Configuration for the Qwen3.5-MoE vision encoder.

    Inherits from ``Qwen3VLMoeVisionConfig``. Qwen3.5-MoE does not use
    deepstack mergers, so ``deepstack_visual_indexes`` defaults to an
    empty list.

    Attributes:
        model_type: The model type identifier (``"qwen3_5_moe"``).
    """

    model_type = "qwen3_5_moe"

    def __init__(
        self,
        deepstack_visual_indexes: list[int] | None = None,
        vision_attn_bias: bool = True,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2048,
        spatial_merge_size: int = 2,
        **kwargs,
    ):
        """Initialize Qwen3.5-MoE vision config.

        Args:
            deepstack_visual_indexes: Indices of vision transformer layers
                whose outputs are used as deepstack embeddings. Defaults to
                an empty list for Qwen3.5-MoE.
            vision_attn_bias: Whether to use bias in vision attention.
            num_position_embeddings: Number of learned position embeddings.
            out_hidden_size: Output hidden size after the patch merger.
            spatial_merge_size: Spatial merge factor for the patch merger.
            **kwargs: Forwarded to ``Qwen3VLMoeVisionConfig``.
        """
        requested_indexes = [] if deepstack_visual_indexes is None else list(deepstack_visual_indexes)
        bootstrap_indexes = requested_indexes if requested_indexes else [0]
        super().__init__(deepstack_visual_indexes=bootstrap_indexes, **kwargs)
        self.deepstack_visual_indexes = requested_indexes
        self.vision_attn_bias = vision_attn_bias
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size


@register_config("qwen3_5_moe")
class Qwen3_5MoeConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3.5-MoE multimodal (vision-language) model.

    Composes a ``Qwen3_5MoeTextConfig`` for the language backbone and a
    ``Qwen3_5MoeVisionConfig`` for the vision encoder.

    Attributes:
        model_type: The model type identifier (``"qwen3_5_moe"``).
        text_config: Text model sub-configuration.
        vision_config: Vision encoder sub-configuration.
        image_token_id: Token id for image placeholders.
        video_token_id: Token id for video placeholders.
        vision_start_token_id: Token id marking the start of a vision span.
        vision_end_token_id: Token id marking the end of a vision span.
    """

    model_type = "qwen3_5_moe"

    def __init__(
        self,
        *,
        text_config: Qwen3_5MoeTextConfig | dict[str, Any] | None = None,
        vision_config: Qwen3_5MoeVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize Qwen3.5-MoE composite configuration.

        Args:
            text_config: Text model sub-configuration. Instantiated from a
                dict if needed.
            vision_config: Vision encoder sub-configuration. Instantiated
                from a dict if needed.
            image_token_id: Token id for image placeholders.
            video_token_id: Token id for video placeholders.
            vision_start_token_id: Token id marking vision span start.
            vision_end_token_id: Token id marking vision span end.
            tie_word_embeddings: Whether to tie input and output embedding
                weights.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        if text_config is None:
            text_cfg = Qwen3_5MoeTextConfig()
        elif isinstance(text_config, Qwen3_5MoeTextConfig):
            text_cfg = text_config
        else:
            text_cfg = Qwen3_5MoeTextConfig(**text_config)

        if vision_config is None:
            vision_cfg = Qwen3_5MoeVisionConfig()
        elif isinstance(vision_config, Qwen3_5MoeVisionConfig):
            vision_cfg = vision_config
        else:
            vision_cfg = Qwen3_5MoeVisionConfig(**vision_config)

        self.text_config = text_cfg
        self.vision_config = vision_cfg
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
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeVisionConfig",
)
