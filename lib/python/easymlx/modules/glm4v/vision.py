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

"""GLM-4V vision components (serving/inference only).

This module implements the vision encoder for GLM-4V, including patch
embedding, vision rotary embeddings, attention blocks, spatial downsampling,
and patch merging.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.layers.attention import scaled_dot_product_attention
from easymlx.layers.embeddings import grid_sample

from .glm4v_configuration import Glm4VVisionConfig


def check_array_shape(arr: mx.array) -> bool:
    """Checks whether a weight array has the expected MLX convolution layout.

    Validates 4D ``(out_channels, kH, kW, in_channels)`` or 5D
    ``(groups, out_channels, kH, kW, in_channels)`` shapes to determine
    if transposition is needed.

    Args:
        arr: Weight array to check.

    Returns:
        True if the array is already in MLX-native layout, False otherwise.
    """
    shape = arr.shape
    if len(shape) == 4:
        out_channels, k_h, k_w, _ = shape
        return (out_channels >= k_h) and (out_channels >= k_w) and (k_h == k_w)
    if len(shape) == 5:
        _, out_channels, k_h, k_w, t = shape
        if t == 3:
            return True
        return (out_channels >= k_h) and (out_channels >= k_w) and (k_h == k_w)
    return False


def rotate_half(x: mx.array) -> mx.array:
    """Rotates the second half of the last dimension and negates it.

    Used for standard rotary position embedding application.

    Args:
        x: Input tensor with last dimension divisible by 2.

    Returns:
        Tensor with the two halves swapped and the first half negated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    """Applies rotary position embeddings to a vision tensor.

    Args:
        tensor: Input tensor of shape ``(batch, seq, heads, dim)``.
        freqs: Frequency tensor of shape ``(seq, dim//2)``.

    Returns:
        Tensor with rotary embeddings applied, same shape as input,
        cast back to the original dtype.
    """
    orig_dtype = tensor.dtype
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class Glm4vVisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for the GLM-4V vision encoder.

    Generates frequency-based position embeddings for spatial positions
    in the vision encoder.

    Attributes:
        dim: Dimensionality of the rotary embedding (half of head_dim).
        theta: Base frequency for the rotary embedding.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Initializes the vision rotary embedding.

        Args:
            dim: Embedding dimension (typically ``head_dim // 2``).
            theta: Base frequency. Defaults to 10000.0.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        """Computes rotary frequencies for a given sequence length.

        Args:
            seqlen: Sequence length (as a scalar mx.array or int).

        Returns:
            Frequency matrix of shape ``(seqlen, dim)``.
        """
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        seq = mx.arange(seqlen.item(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class Glm4vVisionEmbeddings(nn.Module):
    """Position-aware embeddings for the GLM-4V vision encoder.

    Adds interpolated 2D position embeddings to patch embeddings based
    on the spatial coordinates of each patch within its image.

    Attributes:
        config: Vision configuration.
        embed_dim: Embedding dimensionality.
        image_size: Default input image size.
        patch_size: Spatial patch size.
        num_patches: Number of patches for the default image size.
        position_embedding: Learnable position embedding table.
    """

    def __init__(self, config: Glm4VVisionConfig):
        """Initializes the vision embeddings.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, embeddings, lengths, image_shapes, h_coords, w_coords):
        """Adds interpolated position embeddings to patch embeddings.

        Uses bilinear grid sampling to interpolate the learned 2D position
        embedding grid to the actual spatial coordinates of each patch.

        Args:
            embeddings: Patch embeddings of shape
                ``(total_patches, embed_dim)``.
            lengths: Number of patches per image segment.
            image_shapes: Grid shapes ``(T, H, W)`` for each image segment.
            h_coords: Height coordinates for each patch.
            w_coords: Width coordinates for each patch.

        Returns:
            Embeddings with position information added, same shape as input.
        """
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]

        if total_seq == 0:
            adapted_pos_embed = mx.empty(0, hidden_size, dtype=pos_embed_weight.dtype)
        else:
            if isinstance(lengths, list):
                lengths = mx.array(lengths, dtype=mx.int32)
            if not isinstance(image_shapes, mx.array):
                image_shapes = mx.array(image_shapes, dtype=mx.int32)

            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.reshape(orig_size, orig_size, hidden_size)
                .transpose(2, 0, 1)[None, ...]
                .astype(mx.float32)
            )

            target_h = mx.concatenate([mx.repeat(image_shapes[i, 1], lengths[i]) for i in range(len(lengths))]).astype(
                mx.float32
            )
            target_w = mx.concatenate([mx.repeat(image_shapes[i, 2], lengths[i]) for i in range(len(lengths))]).astype(
                mx.float32
            )

            h_coords = h_coords.astype(mx.float32)
            w_coords = w_coords.astype(mx.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            grid = mx.stack((norm_w, norm_h), axis=-1)[None, :, None, ...]
            interpolated_embed_fp32 = grid_sample(pos_embed_2d.transpose(0, 2, 3, 1), grid)
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(1)
            adapted_pos_embed = adapted_pos_embed_fp32.astype(pos_embed_weight.dtype)

        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vVisionPatchEmbed(nn.Module):
    """3D patch embedding for the GLM-4V vision encoder.

    Converts raw pixel inputs into patch embeddings using a 3D convolution
    that handles both spatial and temporal dimensions.

    Attributes:
        config: Vision configuration.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        in_channels: Number of input channels.
        embed_dim: Output embedding dimensionality.
        proj: 3D convolution layer for patch extraction.
    """

    def __init__(self, config: Glm4VVisionConfig) -> None:
        """Initializes the patch embedding layer.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Extracts patch embeddings from pixel values.

        Args:
            hidden_states: Flattened pixel values to be reshaped into
                patches of shape
                ``(N, in_channels, temporal_patch_size, patch_size, patch_size)``.

        Returns:
            Patch embeddings of shape ``(num_patches, embed_dim)``.
        """
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class Glm4vVisionPatchMerger(nn.Module):
    """Patch merger for the GLM-4V vision encoder.

    Applies a projection, layer normalization, and a gated MLP to merge
    spatially downsampled patches into the final vision representation.

    Attributes:
        proj: Initial linear projection.
        post_projection_norm: Layer normalization after projection.
        gate_proj: Gate projection for the gated MLP.
        up_proj: Up projection for the gated MLP.
        down_proj: Down projection for the gated MLP.
    """

    def __init__(self, dim: int, context_dim: int, bias: bool = False) -> None:
        """Initializes the patch merger.

        Args:
            dim: Input and output dimensionality.
            context_dim: Intermediate dimensionality for the gated MLP.
            bias: Whether to include bias terms. Defaults to False.
        """
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies projection, normalization, and gated MLP.

        Args:
            hidden_states: Input tensor of shape ``(N, dim)``.

        Returns:
            Merged features of shape ``(N, dim)``.
        """
        hidden_states = self.proj(hidden_states)
        hidden_states = nn.gelu(self.post_projection_norm(hidden_states))
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm4vVisionAttention(nn.Module):
    """Multi-head self-attention for the GLM-4V vision encoder.

    Applies rotary position embeddings and computes attention within
    variable-length sequences defined by cumulative sequence lengths.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        qkv: Fused QKV projection.
        proj: Output projection.
    """

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        """Initializes the vision attention layer.

        Args:
            dim: Input/output dimensionality.
            num_heads: Number of attention heads. Defaults to 16.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(
        self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array | None = None
    ) -> mx.array:
        """Computes vision self-attention with rotary embeddings.

        Processes variable-length sequences by splitting along cumulative
        sequence length boundaries.

        Args:
            hidden_states: Input tensor of shape ``(total_seq, dim)``.
            cu_seqlens: Cumulative sequence lengths array for splitting
                the flattened sequence into individual images/videos.
            rotary_pos_emb: Rotary position embeddings or None.

        Returns:
            Output tensor of shape ``(total_seq, dim)``.
        """
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3)

        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [mx.split(tensor, [lengths[0], sum(lengths[:2])], axis=2) for tensor in (q, k, v)]

        outputs = []
        for q_split, k_split, v_split in zip(*splits, strict=False):
            output = scaled_dot_product_attention(q_split, k_split, v_split, scale=self.scale, mask=None)
            outputs.append(output)

        output = mx.concatenate(outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class Glm4vVisionMLP(nn.Module):
    """Feed-forward network for the GLM-4V vision encoder.

    Uses a SwiGLU-style gated activation.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.
    """

    def __init__(self, dim: int, hidden_dim: int):
        """Initializes the vision MLP.

        Args:
            dim: Input/output dimensionality.
            hidden_dim: Intermediate dimensionality.
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm4vVisionBlock(nn.Module):
    """Single vision transformer block for GLM-4V.

    Applies pre-norm attention and MLP with residual connections.

    Attributes:
        norm1: Pre-attention normalization.
        norm2: Pre-MLP normalization.
        attn: Vision attention sub-layer.
        mlp: Vision MLP sub-layer.
    """

    def __init__(self, config: Glm4VVisionConfig) -> None:
        """Initializes a vision block.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Glm4vVisionAttention(dim=config.hidden_size, num_heads=config.num_heads)
        self.mlp = Glm4vVisionMLP(dim=config.hidden_size, hidden_dim=config.out_hidden_size)

    def __call__(self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array) -> mx.array:
        """Runs the vision block forward pass.

        Args:
            hidden_states: Input tensor of shape ``(total_seq, hidden_size)``.
            cu_seqlens: Cumulative sequence lengths.
            rotary_pos_emb: Rotary position embeddings.

        Returns:
            Output tensor of shape ``(total_seq, hidden_size)``.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    """Complete vision encoder for GLM-4V.

    Processes pixel values through patch embedding, position embedding,
    vision transformer blocks, and spatial downsampling to produce
    vision features compatible with the language model.

    Attributes:
        config: Vision configuration.
        model_type: Vision model type identifier.
        spatial_merge_size: Spatial downsampling factor.
        embeddings: Position-aware embedding module.
        patch_embed: 3D patch embedding module.
        window_size: Window size for windowed attention.
        patch_size: Spatial patch size.
        spatial_merge_unit: Total spatial merge factor (squared).
        rotary_pos_emb: Vision rotary embedding module.
        blocks: List of vision transformer blocks.
        merger: Patch merger module.
        post_conv_layernorm: Normalization after patch embedding.
        downsample: 2D convolution for spatial downsampling.
        post_layernorm: Normalization before downsampling.
    """

    def __init__(self, config: Glm4VVisionConfig) -> None:
        """Initializes the vision model.

        Args:
            config: Vision configuration.
        """
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = Glm4vVisionPatchEmbed(config)

        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Glm4vVisionBlock(config) for _ in range(config.depth)]
        self.merger = Glm4vVisionPatchMerger(dim=config.out_hidden_size, context_dim=config.intermediate_size)

        self.post_conv_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def rot_pos_emb(self, grid_thw: mx.array):
        """Computes rotary position embeddings for the vision encoder.

        Generates 2D (height, width) position IDs grouped by spatial merge
        size, then computes rotary embeddings for those positions.

        Args:
            grid_thw: Grid dimensions of shape ``(num_segments, 3)`` with
                ``(T, H, W)`` per image/video segment.

        Returns:
            A tuple of ``(rotary_pos_emb, pos_ids)`` where
            ``rotary_pos_emb`` has shape ``(total_patches, rotary_dim)``
            and ``pos_ids`` has shape ``(total_patches, 2)``.
        """
        pos_ids = []

        for t, h, w in grid_thw.tolist():
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids]
        return rotary_pos_emb.reshape(pos_ids.shape[0], -1), pos_ids

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: bool | None = None,
    ) -> mx.array:
        """Runs the vision encoder forward pass.

        Args:
            hidden_states: Raw pixel values to be processed.
            grid_thw: Grid dimensions of shape ``(num_segments, 3)``.
            output_hidden_states: Unused. Present for API compatibility.

        Returns:
            Vision features of shape ``(total_merged_patches, out_hidden_size)``.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)

        seq_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeats = grid_thw[:, 0]
        repeated_values = []
        for seq_len, repeat_count in zip(seq_lens.tolist(), repeats.tolist(), strict=False):
            repeated_values.extend([seq_len] * repeat_count)

        cu_seqlens = mx.array(repeated_values).cumsum(axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.reshape(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = self.downsample(hidden_states).reshape(-1, self.config.out_hidden_size)
        hidden_states = self.merger(hidden_states)
        return hidden_states

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes vision encoder weights for MLX compatibility.

        Transposes convolution weights from PyTorch layout to MLX layout
        when necessary and removes position ID keys.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary with properly transposed convolutions.
        """
        sanitized_weights: dict[str, mx.array] = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if "patch_embed.proj.weight" in key or "downsample.weight" in key:
                if check_array_shape(value):
                    sanitized_weights[key] = value
                else:
                    if value.ndim == 5:
                        sanitized_weights[key] = value.transpose(0, 2, 3, 4, 1)
                    elif value.ndim == 4:
                        sanitized_weights[key] = value.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[key] = value
        return sanitized_weights
