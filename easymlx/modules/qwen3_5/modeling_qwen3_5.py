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

"""Qwen3.5 text and multimodal model wrappers for MLX inference.

This module provides the Qwen3.5 architecture on MLX, including a text-only
model (extending Qwen3-Next) and a vision-language model that combines a
Qwen3-VL vision encoder with the Qwen3.5 hybrid text backbone.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCache, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.qwen3_next.modeling_qwen3_next import Qwen3NextModel
from easymlx.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
)

from .qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

CacheView = TransformerCacheView | PageCache


class Qwen3_5VisionMerger(nn.Module):
    """Patch merger for Qwen3.5 vision encoder.

    Reduces spatial resolution by merging adjacent patches and projecting
    the concatenated features through a two-layer MLP with GELU activation.

    Attributes:
        norm: Layer normalization before projection.
        fc1: First projection layer.
        fc2: Second projection layer.
    """

    def __init__(self, hidden_size: int, output_size: int, *, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merge_dim = hidden_size * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(merge_dim, merge_dim, bias=True)
        self.fc2 = nn.Linear(merge_dim, output_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Merge patches and project to output dimensionality.

        Args:
            hidden_states: Vision features of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Merged features of shape ``(batch, seq_len // merge_factor, output_size)``.
        """
        hidden_states = self.norm(hidden_states)
        batch, seq_len, dim = hidden_states.shape
        s = self.spatial_merge_size
        h = w = int(seq_len**0.5)
        hidden_states = hidden_states.reshape(batch, h // s, s, w // s, s, dim)
        hidden_states = hidden_states.transpose(0, 1, 3, 2, 4, 5)
        hidden_states = hidden_states.reshape(batch, (h // s) * (w // s), s * s * dim)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return self.fc2(hidden_states)


class Qwen3_5VisionModel(nn.Module):
    """Qwen3.5 vision encoder with learned positional embeddings and merger.

    Extends the base Qwen3VL vision architecture with a learned ``pos_embed``
    (instead of sinusoidal) and a patch merger that reduces spatial resolution.

    Attributes:
        patch_embed: Patch embedding layer.
        pos_embed: Learned positional embeddings.
        blocks: Stack of vision transformer blocks.
        merger: Patch merger module.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.blocks = [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        num_positions = getattr(config, "num_position_embeddings", 2304)
        self.pos_embed = nn.Embedding(num_positions, config.hidden_size)
        self.merger = Qwen3_5VisionMerger(
            config.hidden_size,
            getattr(config, "out_hidden_size", config.hidden_size),
            spatial_merge_size=spatial_merge_size,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Encode pixel values into merged vision features.

        Args:
            pixel_values: Input image tensor.

        Returns:
            Merged vision features.
        """
        hidden_states = self.patch_embed(pixel_values)
        batch_size, embed_dim, height, width = hidden_states.shape
        num_patches = height * width
        hidden_states = hidden_states.reshape(batch_size, embed_dim, num_patches).transpose(0, 2, 1)
        pos_ids = mx.arange(num_patches)
        hidden_states = hidden_states + self.pos_embed(pos_ids)[None, :, :]
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.merger(hidden_states)


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3_5TextConfig, model_type="qwen3_5_text")
class Qwen3_5TextModel(Qwen3NextModel):
    """Qwen3.5 text-only base model (no LM head).

    Thin wrapper around ``Qwen3NextModel`` registered with the
    ``qwen3_5_text`` model type. The underlying architecture is a hybrid
    of full softmax attention and linear attention layers from Qwen3-Next.

    Attributes:
        config_class: The associated configuration class (``Qwen3_5TextConfig``).
    """

    config_class = Qwen3_5TextConfig


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3_5TextConfig, model_type="qwen3_5_text")
class Qwen3_5ForCausalLM(BaseCausalLMModule[Qwen3_5TextModel, Qwen3_5TextConfig]):
    """Qwen3.5 text causal language model.

    Wraps ``Qwen3_5TextModel`` with a linear LM head for next-token
    prediction. The text backbone uses Qwen3-Next hybrid attention
    (full softmax + linear with gated delta rule).

    Attributes:
        config_class: The associated configuration class (``Qwen3_5TextConfig``).
    """

    config_class = Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        """Initialize the Qwen3.5 text causal LM.

        Args:
            config: Qwen3.5 text configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3_5TextModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3_5Config, model_type="qwen3_5")
class Qwen3_5Model(EasyMLXBaseModule):
    """Qwen3.5 multimodal (vision-language) base model.

    Combines a vision encoder (from Qwen3-VL) with a ``Qwen3_5TextModel``
    language backbone. Image and video pixels are encoded into continuous
    embeddings, fused into the token embedding stream, and processed by
    the language model.

    Attributes:
        config_class: The associated configuration class (``Qwen3_5Config``).
        vision_tower: Vision encoder module.
        vision_proj: Linear projection from vision hidden size to text hidden size.
        language_model: Qwen3.5 text decoder.
    """

    config_class = Qwen3_5Config

    def __init__(self, config: Qwen3_5Config):
        """Initialize the Qwen3.5 multimodal base model.

        Args:
            config: Qwen3.5 multimodal configuration.
        """
        super().__init__(config)
        self.model_config = config
        self.vision_tower = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
    ) -> mx.array:
        """Compute input embeddings, optionally merging vision features.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            pixel_values: Optional image tensor for vision encoding.

        Returns:
            Combined embeddings of shape ``(batch, seq_len, hidden_size)``.
        """
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(pixel_values)
        return Qwen3VLModel.merge_input_ids_with_image_features(
            self.model_config.image_token_id,
            self.model_config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

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
        """Run the full vision-language model forward pass.

        Encodes image/video inputs via the vision tower, merges them into
        the text embedding stream, and runs the language model decoder.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3_5Config, model_type="qwen3_5")
class Qwen3_5ForConditionalGeneration(EasyMLXBaseModule):
    """Qwen3.5 multimodal conditional generation model.

    End-to-end vision-language model that wraps ``Qwen3_5Model`` and adds
    a causal LM head for image/video-conditioned text generation. Supports
    both image and video inputs via the underlying vision encoder.

    Attributes:
        config_class: The associated configuration class (``Qwen3_5Config``).
        model: The base Qwen3.5 multimodal model.
        lm_head: Optional vocabulary projection head (when embeddings are not tied).
    """

    config_class = Qwen3_5Config

    def __init__(self, config: Qwen3_5Config):
        """Initialize Qwen3.5 for conditional generation.

        Args:
            config: Qwen3.5 multimodal configuration.
        """
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            text_config = config.text_config
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Rename HF weight keys to match the easymlx model structure.

        Handles:
            - ``model.visual.*`` -> ``model.vision_tower.*``
            - Drops ``mtp.*`` (multi-token prediction head, not used for inference).

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

    @property
    def visual(self):
        """Access the vision transformer for backward compatibility."""
        return self.model.vision_tower

    @property
    def language_model(self):
        """Access the language model for backward compatibility."""
        return self.model.language_model

    def get_input_embeddings(self):
        """Get the input embedding layer."""
        return self.model.language_model.embed_tokens

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
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
)
