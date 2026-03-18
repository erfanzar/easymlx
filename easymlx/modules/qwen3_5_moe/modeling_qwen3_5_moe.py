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

"""Qwen3.5-MoE MLX model implementation for serving and inference.

This module provides the Qwen3.5-MoE architecture on MLX, combining a
vision encoder (reused from Qwen3-VL) with a hybrid full/linear attention
MoE text backbone (mirroring Qwen3-Next). Supports both text-only causal
language modeling and multimodal vision-language conditional generation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCache, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from easymlx.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
)

from .qwen3_5_moe_configuration import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

CacheView = TransformerCacheView | PageCache


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3_5MoeTextConfig, model_type="qwen3_5_moe_text")
class Qwen3_5MoeTextModel(EasyMLXBaseModule):
    """Qwen3.5-MoE text-only base model (no LM head).

    Uses the Qwen3-Next hybrid attention architecture (full softmax + linear
    attention layers) with MoE routing and shared experts.

    Attributes:
        config_class: The associated configuration class.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers (mixed full/linear attention with MoE).
        norm: Final RMSNorm with ``(1+w)`` scaling.
    """

    config_class = Qwen3_5MoeTextConfig

    def __init__(self, config: Qwen3_5MoeTextConfig):
        """Initialize the base Qwen3.5-MoE text model.

        Args:
            config: Qwen3.5-MoE text configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Qwen3.5-MoE text transformer stack.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states from the transformer stack.

        Raises:
            ValueError: If ``cache_views`` length does not match layer count.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3_5MoeTextConfig, model_type="qwen3_5_moe_text")
class Qwen3_5MoeForCausalLM(BaseCausalLMModule[Qwen3_5MoeTextModel, Qwen3_5MoeTextConfig]):
    """Qwen3.5-MoE text causal language model.

    Wraps ``Qwen3_5MoeTextModel`` with a linear LM head for next-token
    prediction. Supports MoE routing with auxiliary load-balancing loss.

    Attributes:
        config_class: The associated configuration class.
    """

    config_class = Qwen3_5MoeTextConfig

    def __init__(self, config: Qwen3_5MoeTextConfig):
        """Initialize the Qwen3.5-MoE text causal LM.

        Args:
            config: Qwen3.5-MoE text configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3_5MoeTextModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3_5MoeConfig, model_type="qwen3_5_moe")
class Qwen3_5MoeModel(EasyMLXBaseModule):
    """Qwen3.5-MoE multimodal (vision-language) base model.

    Combines a vision encoder (reused from Qwen3-VL), a vision-to-text
    projection, and a Qwen3.5-MoE text language model with hybrid
    attention and MoE routing.

    Attributes:
        config_class: The associated configuration class.
        vision_tower: Qwen3-VL vision encoder.
        vision_proj: Linear projection from vision to text hidden size.
        language_model: Qwen3.5-MoE text backbone.
    """

    config_class = Qwen3_5MoeConfig

    def __init__(self, config: Qwen3_5MoeConfig):
        """Initialize the Qwen3.5-MoE multimodal model.

        Args:
            config: Qwen3.5-MoE multimodal configuration.
        """
        super().__init__(config)
        self.model_config = config
        self.vision_tower = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
    ) -> mx.array:
        """Compute input embeddings, optionally merging vision features.

        Args:
            input_ids: Token IDs of shape ``[batch, seq_len]``.
            pixel_values: Optional image tensor for vision encoding.

        Returns:
            Combined embeddings of shape ``[batch, seq_len, hidden_size]``.
        """
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(pixel_values)
        return self.merge_input_ids_with_image_features(
            self.model_config.image_token_id,
            self.model_config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image/video features into text embeddings at placeholder positions.

        Args:
            image_token_id: Token ID used for image placeholders.
            video_token_id: Token ID used for video placeholders.
            image_features: Vision features of shape
                ``[total_image_tokens, hidden_size]``.
            inputs_embeds: Text embeddings of shape
                ``[batch, seq_len, hidden_size]``.
            input_ids: Token IDs of shape ``[batch, seq_len]``.

        Returns:
            Merged embeddings of shape ``[batch, seq_len, hidden_size]``.

        Raises:
            ValueError: If the number of image token positions does not
                match the available image features for a batch element.
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
        """Return the list of transformer decoder layers."""
        return self.language_model.layers

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        **kwargs,
    ) -> mx.array:
        """Run the full vision-language MoE model forward pass.

        Args:
            input_ids: Token IDs.
            pixel_values: Optional image tensor for vision input.
            attention_mask: Optional attention mask.
            cache_views: Optional per-layer KV cache views.
            cache_metadata: Optional page metadata for paged attention.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Hidden states from the language model.
        """
        del kwargs
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)
        return self.language_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=inputs_embeds,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3_5MoeConfig, model_type="qwen3_5_moe")
class Qwen3_5MoeForConditionalGeneration(EasyMLXBaseModule):
    """Qwen3.5-MoE multimodal conditional generation model.

    End-to-end vision-language model that wraps ``Qwen3_5MoeModel`` and
    adds a causal LM head for image/video-conditioned text generation.
    Supports both image and video inputs via the underlying vision encoder
    and uses MoE routing in the language backbone.

    Attributes:
        config_class: The associated configuration class.
        model: The base Qwen3.5-MoE multimodal model.
    """

    config_class = Qwen3_5MoeConfig

    def __init__(self, config: Qwen3_5MoeConfig):
        """Initialize Qwen3.5-MoE vision-language model for conditional generation.

        Args:
            config: Vision-language model configuration with text and vision
                configs.
        """
        super().__init__(config)
        self.model = Qwen3_5MoeModel(config)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            text_config = config.text_config
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Rename HF weight keys to match the easymlx model structure.

        Args:
            weights: Dictionary mapping HF parameter names to arrays.

        Returns:
            Sanitized weight dictionary with corrected key names.
        """
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("mtp."):
                continue
            new_key = key.replace("model.visual.", "model.vision_tower.")
            new_key = new_key.replace(".mlp.linear_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.linear_fc2.", ".mlp.fc2.")
            new_key = new_key.replace("model.vision_tower.merger.linear_fc1.", "model.vision_tower.merger.fc1.")
            new_key = new_key.replace("model.vision_tower.merger.linear_fc2.", "model.vision_tower.merger.fc2.")
            if new_key.endswith("patch_embed.proj.weight") and value.ndim == 5:
                value = value.sum(axis=2).transpose(0, 2, 3, 1)
            if new_key.endswith("conv1d.weight") and value.ndim == 3:
                value = value.transpose(0, 2, 1)
            sanitized[new_key] = value
        return sanitized

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> mx.array | CausalLMOutput:
        """Compute next-token logits for conditional generation.

        Args:
            input_ids: Token IDs.
            pixel_values: Optional image tensor.
            attention_mask: Optional attention mask.
            cache_views: Optional per-layer KV cache views.
            cache_metadata: Optional page metadata.
            return_dict: If True, return a ``CausalLMOutput``; otherwise
                return raw logits.
            **kwargs: Additional keyword arguments (forwarded to the model).

        Returns:
            A ``CausalLMOutput`` containing logits if ``return_dict`` is True,
            or a raw logits tensor otherwise.
        """
        hidden_states = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
            **kwargs,
        )

        if hidden_states.ndim == 2 and cache_metadata is not None:
            qsl = cache_metadata.query_start_loc
            if not isinstance(qsl, mx.array):
                qsl = mx.array(list(qsl), dtype=mx.int32)
            last_indices = (qsl[1:].astype(mx.int32) - 1).astype(mx.int32)
            hidden_states = mx.take(hidden_states, last_indices, axis=0)

        if self._tie_word_embeddings:
            logits = self.model.language_model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if return_dict:
            return CausalLMOutput(logits=logits)
        return logits


__all__ = (
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeTextModel",
)
