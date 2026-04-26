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

"""Qwen2-VL MLX model implementation for serving and inference.

This module provides the full Qwen2-VL multimodal architecture on MLX,
including a vision encoder with sinusoidal positional embeddings, a text
decoder with optional sliding-window attention, and a conditional generation
wrapper for image/video understanding tasks.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask, scaled_dot_product_attention
from easymlx.layers.rotary import get_rope

from .qwen2_vl_configuration import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig

CacheView = TransformerCacheView | PageCacheView


def _sinusoidal_positions(seq_len: int, dim: int) -> mx.array:
    """Generate sinusoidal positional embeddings.

    Args:
        seq_len: Length of the sequence.
        dim: Embedding dimensionality (must be even).

    Returns:
        Positional embeddings of shape ``(seq_len, dim)`` with alternating
        sine and cosine values.
    """
    inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    positions = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(positions, inv_freq)
    emb = mx.concatenate([mx.sin(freqs), mx.cos(freqs)], axis=-1)
    return emb.astype(mx.float32)


class Qwen2VLVisionPatchEmbed(nn.Module):
    """Patch embedding layer for the Qwen2-VL vision encoder.

    Converts raw pixel values into patch embeddings using a 2-D convolution.

    Attributes:
        patch_size: Size of each image patch.
        in_channels: Number of input image channels.
        out_channels: Dimensionality of the output patch embeddings.
        proj: Convolutional projection layer.
    """

    def __init__(self, config: Qwen2VLVisionConfig):
        """Initialize the patch embedding layer.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = int(config.in_channels)
        self.out_channels = int(config.embed_dim)
        self.proj = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Project pixel values into patch embeddings.

        Args:
            pixel_values: Input image tensor of shape
                ``(batch, channels, height, width)`` or
                ``(batch, height, width, channels)``.

        Returns:
            Patch embeddings of shape
            ``(batch, embed_dim, num_patches_h, num_patches_w)``.

        Raises:
            ValueError: If ``pixel_values`` is not 4-D or channel dimension
                does not match ``in_channels``.
        """
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a 4D tensor.")
        if pixel_values.shape[-1] == self.in_channels:
            conv_input = pixel_values
        elif pixel_values.shape[1] == self.in_channels:
            conv_input = pixel_values.transpose(0, 2, 3, 1)
        else:
            raise ValueError("pixel_values channel dimension does not match expected in_channels.")

        hidden_states = self.proj(conv_input)
        if hidden_states.shape[-1] == self.out_channels:
            hidden_states = hidden_states.transpose(0, 3, 1, 2)
        return hidden_states


class Qwen2VLVisionAttention(nn.Module):
    """Multi-head self-attention for the Qwen2-VL vision encoder.

    Uses a fused QKV projection for efficiency.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor for attention logits.
        qkv: Fused query/key/value linear projection.
        proj: Output linear projection.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """Initialize the vision attention module.

        Args:
            embed_dim: Total embedding dimensionality.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute multi-head self-attention over vision features.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Attention output of shape ``(batch, seq_len, embed_dim)``.
        """
        batch_size, seq_len, dim = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=2)
        q = q.squeeze(2).transpose(0, 2, 1, 3)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        return self.proj(attn)


class Qwen2VLVisionMLP(nn.Module):
    """Feed-forward MLP for the Qwen2-VL vision encoder.

    Attributes:
        fc1: First linear projection.
        fc2: Second linear projection.
        activation: Name of the activation function.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, activation: str = "quick_gelu"):
        """Initialize the vision MLP.

        Args:
            embed_dim: Input and output dimensionality.
            hidden_dim: Intermediate dimensionality.
            activation: Name of the activation function.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = activation

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the MLP transformation with GELU activation.

        Args:
            hidden_states: Input tensor of shape ``(..., embed_dim)``.

        Returns:
            Output tensor of the same shape.
        """
        if self.activation == "quick_gelu":
            hidden_states = nn.gelu(hidden_states)
        else:
            hidden_states = nn.gelu(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return self.fc2(hidden_states)


class Qwen2VLVisionBlock(nn.Module):
    """Single transformer block for the Qwen2-VL vision encoder.

    Attributes:
        norm1: Layer normalization before attention.
        attn: Multi-head self-attention module.
        norm2: Layer normalization before the MLP.
        mlp: Feed-forward MLP module.
    """

    def __init__(self, config: Qwen2VLVisionConfig):
        """Initialize the vision transformer block.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = Qwen2VLVisionAttention(config.embed_dim, config.num_heads)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = Qwen2VLVisionMLP(config.embed_dim, config.hidden_size, activation=config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run one vision transformer block.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Output tensor of the same shape.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen2VLVisionModel(nn.Module):
    """Qwen2-VL vision encoder stack.

    Converts raw images into a flat sequence of vision features with added
    sinusoidal positional embeddings.

    Attributes:
        config: Vision configuration.
        patch_embed: Patch embedding layer.
        blocks: List of vision transformer blocks.
        norm: Final layer normalization.
    """

    def __init__(self, config: Qwen2VLVisionConfig):
        """Initialize the vision encoder.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.config = config
        self.patch_embed = Qwen2VLVisionPatchEmbed(config)
        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.embed_dim)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Encode images into vision feature vectors.

        Args:
            pixel_values: Input images of shape
                ``(batch, channels, height, width)`` or
                ``(batch, height, width, channels)``.

        Returns:
            Flattened vision features of shape ``(total_patches, embed_dim)``.
        """
        hidden_states = self.patch_embed(pixel_values)
        batch_size, embed_dim, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, embed_dim, height * width).transpose(0, 2, 1)
        pos = _sinusoidal_positions(hidden_states.shape[1], embed_dim)
        hidden_states = hidden_states + pos[None, :, :]
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states.reshape(-1, embed_dim)


class Qwen2VLTextAttention(nn.Module):
    """Multi-head attention with GQA for the Qwen2-VL text decoder.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.
    """

    def __init__(self, config: Qwen2VLTextConfig):
        """Initialize the text attention module.

        Args:
            config: Text decoder configuration.
        """
        super().__init__()
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(config.hidden_size // config.num_attention_heads)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False)

        self.rope = get_rope(
            dims=head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute grouped-query attention for the text decoder.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class Qwen2VLTextMLP(nn.Module):
    """SiLU-gated feed-forward MLP for the Qwen2-VL text decoder.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
    """

    def __init__(self, config: Qwen2VLTextConfig):
        """Initialize the text MLP.

        Args:
            config: Text decoder configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen2VLTextLayer(nn.Module):
    """Single decoder layer for the Qwen2-VL text model.

    Attributes:
        use_sliding: Whether this layer uses sliding-window attention.
        self_attn: Multi-head attention module.
        mlp: Feed-forward MLP module.
        input_layernorm: Pre-attention RMS normalization.
        post_attention_layernorm: Post-attention RMS normalization.
    """

    def __init__(self, config: Qwen2VLTextConfig, *, use_sliding: bool = False):
        """Initialize the text decoder layer.

        Args:
            config: Text decoder configuration.
            use_sliding: Whether this layer uses sliding-window attention.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Qwen2VLTextAttention(config)
        self.mlp = Qwen2VLTextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one text decoder layer.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same shape as the input.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen2VLTextModel(nn.Module):
    """Qwen2-VL text decoder transformer stack.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: List of text decoder layers.
        norm: Final RMS normalization.
        sliding_window: Sliding window size for applicable layers.
    """

    def __init__(self, config: Qwen2VLTextConfig):
        """Initialize the text decoder stack.

        Args:
            config: Finalized text decoder configuration.
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2VLTextLayer(config, use_sliding=layer_type == "sliding_attention") for layer_type in config.layer_types
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the text decoder on input tokens or embeddings.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; overrides token lookup.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states from the text decoder.

        Raises:
            ValueError: If ``cache_views`` length does not match layer count.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(input_ids)

        mask: mx.array | str | None = None
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.use_sliding for layer in self.layers):
                    sliding_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.sliding_window,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.use_sliding:
                layer_metadata = cache_metadata.with_sliding_window(self.sliding_window)
            layer_mask = sliding_mask if layer.use_sliding else mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLModel(EasyMLXBaseModule):
    """Base Qwen2-VL multimodal model combining vision and language towers.

    Attributes:
        config_class: The associated configuration class (``Qwen2VLConfig``).
        model_config: Fully-resolved runtime model configuration.
        vision_tower: Vision encoder module.
        vision_proj: Linear projection from vision to text embedding space.
        language_model: Text decoder transformer stack.
    """

    config_class = Qwen2VLConfig

    def __init__(self, config: Qwen2VLConfig):
        """Initialize the base Qwen2-VL model.

        Args:
            config: Top-level Qwen2-VL configuration.
        """
        super().__init__(config)
        model_config = config.to_model_config()
        self.model_config = model_config
        self.vision_tower = Qwen2VLVisionModel(model_config.vision_config)
        self.vision_proj = nn.Linear(model_config.vision_config.embed_dim, model_config.text_config.hidden_size)
        self.language_model = Qwen2VLTextModel(model_config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
    ) -> mx.array:
        """Compute input embeddings, optionally merging vision features.

        Args:
            input_ids: Token IDs.
            pixel_values: Optional image tensor for the vision encoder.

        Returns:
            Combined embeddings.
        """
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(pixel_values)
        hidden_states = self.vision_proj(hidden_states)
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
        """Replace image/video placeholder tokens with projected vision features.

        Looks for positions matching ``image_token_id`` first; if none are
        found, falls back to ``video_token_id``.

        Args:
            image_token_id: Token ID for image placeholders.
            video_token_id: Token ID for video placeholders.
            image_features: Vision features of shape ``(total_patches, hidden_size)``.
            inputs_embeds: Text embeddings of shape ``(batch, seq_len, hidden_size)``.
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Merged embeddings of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If position count does not match feature count.
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
        """Return the list of decoder layers from the language model."""
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
        """Run the full multimodal forward pass.

        Args:
            input_ids: Token IDs.
            pixel_values: Optional image tensor.
            attention_mask: Optional attention mask.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Hidden states from the text decoder.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen2VLConfig, model_type="qwen2_vl")
class Qwen2VLForConditionalGeneration(EasyMLXBaseModule):
    """Qwen2-VL model with a language modeling head for conditional generation.

    Wraps ``Qwen2VLModel`` and adds vocabulary projection to produce
    next-token logits. Supports tied embeddings.

    Attributes:
        config_class: The associated configuration class (``Qwen2VLConfig``).
        model: The base Qwen2-VL multimodal model.
        lm_head: Vocabulary projection (absent when embeddings are tied).
    """

    config_class = Qwen2VLConfig

    def __init__(self, config: Qwen2VLConfig):
        """Initialize the conditional generation model.

        Args:
            config: Top-level Qwen2-VL configuration.
        """
        super().__init__(config)
        self.model = Qwen2VLModel(config)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            text_config = self.model.model_config.text_config
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

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
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.
            return_dict: If ``True``, return ``CausalLMOutput``; otherwise
                return raw logits.
            **kwargs: Additional keyword arguments (forwarded to the model).

        Returns:
            A ``CausalLMOutput`` or raw logits tensor.
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


__all__ = ("Qwen2VLForConditionalGeneration", "Qwen2VLModel")
