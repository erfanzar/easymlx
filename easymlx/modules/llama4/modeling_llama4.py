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

"""Llama4 MLX model implementation for serving and inference.

This module provides the full Llama4 multimodal architecture on MLX, including
a vision encoder, text decoder with MoE and chunked attention, and a
conditional generation wrapper. The unified ``__call__`` API at every layer
accepts both ``TransformerCacheView`` and ``PageCacheView`` for flexible serving.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask, scaled_dot_product_attention
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope

from .llama4_configuration import Llama4Config, Llama4TextConfig, Llama4VisionConfig

CacheView = TransformerCacheView | PageCacheView


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Look up an activation function by name.

    Args:
        name: Case-insensitive activation name. Supported values are
            ``"silu"``, ``"swish"``, ``"gelu"``, and ``"gelu_pytorch_tanh"``.

    Returns:
        The corresponding MLX activation callable.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name in {"gelu", "gelu_pytorch_tanh"}:
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


class Llama4VisionPatchEmbed(nn.Module):
    """Patch embedding layer for the Llama4 vision encoder.

    Converts raw pixel values into a sequence of patch embeddings using a
    2-D convolution with kernel and stride equal to ``patch_size``.

    Attributes:
        in_channels: Number of input image channels.
        out_channels: Dimensionality of the output patch embeddings.
        proj: Convolutional projection layer.
    """

    def __init__(self, config: Llama4VisionConfig):
        """Initialize the patch embedding layer.

        Args:
            config: Vision configuration specifying channel counts, hidden
                size, and patch size.
        """
        super().__init__()
        self.in_channels = int(config.num_channels)
        self.out_channels = int(config.hidden_size)
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
            Patch embeddings of shape ``(batch, embed_dim, num_patches_h, num_patches_w)``.

        Raises:
            ValueError: If ``pixel_values`` is not 4-D or the channel
                dimension does not match ``in_channels``.
        """
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a 4D tensor.")
        if pixel_values.shape[-1] == self.in_channels:
            conv_input = pixel_values
        elif pixel_values.shape[1] == self.in_channels:
            conv_input = pixel_values.transpose(0, 2, 3, 1)
        else:
            raise ValueError("pixel_values channel dimension does not match expected num_channels.")

        hidden_states = self.proj(conv_input)
        if hidden_states.shape[-1] == self.out_channels:
            hidden_states = hidden_states.transpose(0, 3, 1, 2)
        return hidden_states


class Llama4VisionAttention(nn.Module):
    """Multi-head self-attention for the Llama4 vision encoder.

    Uses a fused QKV projection for efficiency. No positional encoding is
    applied (vision patches rely on learned embeddings).

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor applied to attention logits.
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


class Llama4VisionMLP(nn.Module):
    """Feed-forward MLP for the Llama4 vision encoder.

    A standard two-layer MLP with a configurable activation function.

    Attributes:
        fc1: First linear projection (embed_dim -> hidden_dim).
        fc2: Second linear projection (hidden_dim -> embed_dim).
        act_fn: Activation function applied after ``fc1``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, activation: str = "gelu"):
        """Initialize the vision MLP.

        Args:
            embed_dim: Input and output dimensionality.
            hidden_dim: Intermediate dimensionality.
            activation: Name of the activation function (default: ``"gelu"``).
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act_fn = _get_activation(activation)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., embed_dim)``.

        Returns:
            Output tensor of the same shape as the input.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.fc2(hidden_states)


class Llama4VisionBlock(nn.Module):
    """Single transformer block for the Llama4 vision encoder.

    Consists of layer-normalized self-attention followed by a layer-normalized
    MLP, each with residual connections.

    Attributes:
        norm1: Layer normalization before the attention sub-layer.
        attn: Multi-head self-attention module.
        norm2: Layer normalization before the MLP sub-layer.
        mlp: Feed-forward MLP module.
    """

    def __init__(self, config: Llama4VisionConfig):
        """Initialize the vision transformer block.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Llama4VisionAttention(config.hidden_size, config.num_attention_heads)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = Llama4VisionMLP(config.hidden_size, config.intermediate_size, activation=config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run one vision transformer block.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.

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


class Llama4VisionModel(nn.Module):
    """Llama4 vision encoder stack.

    Converts raw images into a flat sequence of projected vision features
    suitable for merging with text token embeddings.

    Attributes:
        config: Vision configuration.
        patch_embed: Patch embedding layer.
        blocks: List of vision transformer blocks.
        norm: Final layer normalization.
        proj: Linear projection to ``vision_output_dim``.
    """

    def __init__(self, config: Llama4VisionConfig):
        """Initialize the vision encoder.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.config = config
        self.patch_embed = Llama4VisionPatchEmbed(config)
        self.blocks = [Llama4VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.vision_output_dim, bias=False)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Encode images into vision feature vectors.

        Args:
            pixel_values: Input images of shape
                ``(batch, channels, height, width)`` or
                ``(batch, height, width, channels)``.

        Returns:
            Flattened vision features of shape
            ``(total_patches, vision_output_dim)``.
        """
        hidden_states = self.patch_embed(pixel_values)
        batch_size, embed_dim, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, embed_dim, height * width).transpose(0, 2, 1)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, hidden_states.shape[-1])


class Llama4Attention(nn.Module):
    """Multi-head attention for the Llama4 text decoder.

    Supports grouped-query attention (GQA), optional QK normalization, and
    optional rotary positional embeddings. Works with both standard and
    paged KV caches.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
        use_qk_norm: Whether QK normalization is applied.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: RMS normalization for queries.
        k_norm: RMS normalization for keys.
        rope: Rotary positional embedding module, or ``None``.
        attention_performer: Attention computation backend.
    """

    def __init__(self, config: Llama4TextConfig, *, use_rope: bool = True):
        """Initialize the text attention module.

        Args:
            config: Text decoder configuration.
            use_rope: Whether to apply rotary positional embeddings.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = bool(config.use_qk_norm)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = None
        if use_rope:
            self.rope = get_rope(
                dims=self.head_dim,
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
        """Compute multi-head attention for a text decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``
                or ``(total_tokens, hidden_size)`` for paged mode.
            mask: Attention mask (broadcastable 4-D array, string sentinel,
                or ``None`` for no masking).
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Optional paged-cache metadata.

        Returns:
            Output tensor of the same leading shape as ``hidden_states``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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


class Llama4MLP(nn.Module):
    """Dense gated MLP for the Llama4 text decoder.

    Uses a SiLU-gated architecture: ``down(silu(gate(x)) * up(x))``.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
        act_fn: Activation function (SiLU by default).
    """

    def __init__(self, config: Llama4TextConfig):
        """Initialize the dense MLP.

        Args:
            config: Text decoder configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size_mlp, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size_mlp, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size_mlp, config.hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Llama4SparseBlock(nn.Module):
    """Mixture-of-Experts block for the Llama4 text decoder.

    Routes each token to the top-k experts via a softmax router, then
    aggregates expert outputs weighted by router scores.

    Attributes:
        router: Top-k expert routing module.
        experts: SwitchGLU expert bank.
    """

    def __init__(self, config: Llama4TextConfig):
        """Initialize the MoE block.

        Args:
            config: Text decoder configuration specifying expert count and
                routing parameters.
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
            norm_topk_prob=True,
        )
        self.experts = SwitchGLU(config.hidden_size, config.intermediate_size, config.num_local_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and aggregate outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Weighted sum of expert outputs, same shape as input.
        """
        inds, scores = self.router(hidden_states)
        out = self.experts(hidden_states, inds)
        return (out * scores[..., None]).sum(axis=-2).astype(out.dtype)


class Llama4DecoderLayer(nn.Module):
    """Single decoder layer for the Llama4 text model.

    Consists of RMS-normalized self-attention followed by either a dense MLP
    or an MoE block, each with residual connections.

    Attributes:
        use_chunked: Whether this layer uses chunked (windowed) attention.
        self_attn: Attention module.
        mlp: Dense MLP or MoE sparse block.
        input_layernorm: Pre-attention RMS normalization.
        post_attention_layernorm: Post-attention RMS normalization.
    """

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_idx: int,
        *,
        use_chunked: bool = False,
        use_rope: bool = True,
    ):
        """Initialize the decoder layer.

        Args:
            config: Text decoder configuration.
            layer_idx: Zero-based index of this layer in the stack.
            use_chunked: Whether this layer uses chunked attention.
            use_rope: Whether to apply rotary positional embeddings.
        """
        super().__init__()
        self.use_chunked = use_chunked
        self.self_attn = Llama4Attention(config, use_rope=use_rope)
        use_moe = layer_idx in config.moe_layers
        self.mlp = Llama4SparseBlock(config) if use_moe else Llama4MLP(config)
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
        """Run one decoder layer.

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


class Llama4TextModel(nn.Module):
    """Llama4 text decoder transformer stack.

    Embeds input tokens, runs them through a stack of decoder layers (with
    mixed chunked/full attention and MoE routing), and applies final RMS
    normalization.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
        attention_chunk_size: Window size used for chunked attention layers.
    """

    def __init__(self, config: Llama4TextConfig):
        """Initialize the text decoder stack.

        Args:
            config: Finalized text decoder configuration.
        """
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Llama4DecoderLayer(
                config,
                idx,
                use_chunked=layer_type == "chunked_attention",
                use_rope=not bool(config.no_rope_layers[idx]),
            )
            for idx, layer_type in enumerate(config.layer_types)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_chunk_size = config.attention_chunk_size

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
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; if provided, ``input_ids``
                is used only for shape inference.
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` or
            ``(total_tokens, hidden_size)`` in paged mode.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of layers.
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
        chunk_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.use_chunked for layer in self.layers):
                    chunk_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.attention_chunk_size,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.use_chunked:
                layer_metadata = cache_metadata.with_sliding_window(self.attention_chunk_size)
            layer_mask = chunk_mask if layer.use_chunked else mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.BASE_MODULE, config=Llama4Config, model_type="llama4")
class Llama4Model(EasyMLXBaseModule):
    """Llama4 base multimodal model combining vision and language towers.

    Encodes images through the vision tower, projects them into the text
    embedding space, merges visual features with text embeddings, and runs
    the combined sequence through the text decoder.

    Attributes:
        config_class: The associated configuration class (``Llama4Config``).
        model_config: Fully-resolved runtime model configuration.
        vision_tower: Vision encoder module.
        vision_proj: Linear projection from vision output to text hidden size.
        language_model: Text decoder transformer stack.
    """

    config_class = Llama4Config

    def __init__(self, config: Llama4Config):
        """Initialize the Llama4 base model.

        Args:
            config: Top-level Llama4 configuration.
        """
        super().__init__(config)
        model_config = config.to_model_config()
        self.model_config = model_config
        self.vision_tower = Llama4VisionModel(model_config.vision_config)
        self.vision_proj = nn.Linear(
            model_config.vision_config.vision_output_dim,
            model_config.text_config.hidden_size,
            bias=model_config.vision_config.multi_modal_projector_bias,
        )
        self.language_model = Llama4TextModel(model_config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
    ) -> mx.array:
        """Compute input embeddings, optionally merging vision features.

        When ``pixel_values`` is provided, encodes the images and merges the
        resulting features into the token embedding sequence at positions
        marked by the image token ID.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            pixel_values: Optional image tensor for the vision encoder.

        Returns:
            Combined embeddings of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``input_ids`` is ``None``.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided to compute embeddings.")
        input_ids = mx.array(input_ids, dtype=mx.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(pixel_values)
        hidden_states = self.vision_proj(hidden_states)
        return self.merge_input_ids_with_image_features(
            self.model_config.image_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Replace image placeholder tokens with projected vision features.

        For each batch element, finds positions where ``input_ids`` equals
        ``image_token_id`` and substitutes the corresponding vision feature
        vectors.

        Args:
            image_token_id: The token ID that marks image placeholder positions.
            image_features: Vision features of shape ``(total_patches, hidden_size)``.
            inputs_embeds: Text token embeddings of shape ``(batch, seq_len, hidden_size)``.
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Merged embeddings of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If the number of image token positions does not match
                the number of available image features for a batch element.
        """
        image_positions = input_ids == image_token_id
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
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            pixel_values: Optional image tensor for the vision encoder.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=Llama4Config, model_type="llama4")
class Llama4ForConditionalGeneration(EasyMLXBaseModule):
    """Llama4 model with a language modeling head for conditional generation.

    Wraps ``Llama4Model`` and adds vocabulary projection (``lm_head``) to
    produce next-token logits. Supports tied embeddings.

    Attributes:
        config_class: The associated configuration class (``Llama4Config``).
        model: The base Llama4 multimodal model.
        lm_head: Vocabulary projection layer (absent when embeddings are tied).
    """

    config_class = Llama4Config

    def __init__(self, config: Llama4Config):
        """Initialize the conditional generation model.

        Args:
            config: Top-level Llama4 configuration.
        """
        super().__init__(config)
        self.model = Llama4Model(config)
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
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            pixel_values: Optional image tensor for the vision encoder.
            attention_mask: Optional attention mask.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-cache metadata.
            return_dict: If ``True``, return a ``CausalLMOutput``; otherwise
                return raw logits.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A ``CausalLMOutput`` containing logits when ``return_dict`` is
            ``True``, or raw logits tensor otherwise.
        """
        del kwargs
        hidden_states = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
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


__all__ = ("Llama4ForConditionalGeneration", "Llama4Model")
