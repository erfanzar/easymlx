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

"""GLM-4V language components (serving/inference only).

This module implements the language (text) backbone for GLM-4V, including
multimodal rotary position embeddings (M-RoPE), the decoder stack, and
the language model wrapper with cache management.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.layers.attention import AttentionPerformer, build_attention_mask

from .glm4v_configuration import Glm4VModelConfig, Glm4VTextConfig

CacheView = TransformerCacheView | PageCacheView


def _compute_default_rope_parameters(
    config: Glm4VTextConfig | None = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:
    """Computes default rotary position embedding parameters.

    Derives inverse frequency values and attention scaling from either a
    text config or explicit keyword arguments.

    Args:
        config: Text configuration to derive parameters from. Mutually
            exclusive with ``**rope_kwargs``.
        **rope_kwargs: Explicit RoPE parameters. Must include ``"base"``
            and ``"dim"`` when used.

    Returns:
        A tuple of ``(inv_freq, attention_factor)`` where ``inv_freq`` is
        an array of inverse frequency values and ``attention_factor`` is
        a float scaling factor (always 1.0 for default RoPE).

    Raises:
        ValueError: If both ``config`` and ``rope_kwargs`` are provided,
            or neither is provided.
    """
    if config is not None and rope_kwargs:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in _compute_default_rope_parameters"
        )
    if rope_kwargs:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
    else:
        raise ValueError("config or rope_kwargs must be provided")

    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim))
    return inv_freq, attention_factor


class GLM4VRotaryEmbedding(nn.Module):
    """Multimodal rotary position embedding for GLM-4V.

    Computes 3D rotary embeddings (temporal, height, width) suitable for
    vision-language inputs. The inverse frequencies are cached and reused.

    Attributes:
        rope_type: Type of RoPE (e.g., ``"default"``).
        max_seq_len_cached: Maximum cached sequence length.
        original_max_seq_len: Original maximum sequence length from config.
        attention_scaling: Attention scaling factor.
    """

    def __init__(self, config: Glm4VTextConfig):
        """Initializes the rotary embedding module.

        Args:
            config: Text configuration providing RoPE parameters.
        """
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        inv_freq, self.attention_scaling = _compute_default_rope_parameters(config)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)

    def __call__(self, x: mx.array, position_ids: mx.array):
        """Computes cosine and sine embeddings for multimodal positions.

        Args:
            x: Input tensor used only for dtype inference.
            position_ids: Position IDs of shape ``(3, batch_size, seq_len)``
                representing temporal, height, and width positions.

        Returns:
            A tuple of ``(cos, sin)`` embeddings, each of shape
            ``(3, batch_size, seq_len, rotary_dim)``.
        """
        inv_freq_expanded = self._inv_freq[None, None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(inv_freq_expanded, (3, position_ids.shape[1], self._inv_freq.shape[0], 1))
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 1, 3, 2)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half_llm(x: mx.array) -> mx.array:
    """Rotates interleaved pairs of elements for RoPE application.

    Splits the input into even and odd indexed elements, negates the odd
    elements, and interleaves them back.

    Args:
        x: Input tensor with last dimension divisible by 2.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.flatten(mx.stack([-x2, x1], axis=-1), start_axis=-2, end_axis=-1)


def apply_multimodal_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, mrope_section: list["int"]):
    """Applies multimodal rotary position embeddings to query and key tensors.

    Splits the rotary dimensions according to ``mrope_section`` boundaries,
    assigning each section to a different modality axis (temporal, height,
    width).

    Args:
        q: Query tensor of shape ``(batch, heads, seq_len, head_dim)``.
        k: Key tensor of shape ``(batch, heads, seq_len, head_dim)``.
        cos: Cosine embeddings from the rotary embedding module.
        sin: Sine embeddings from the rotary embedding module.
        mrope_section: List of integers defining the section sizes for
            each modality axis.

    Returns:
        A tuple of ``(q_embed, k_embed)`` with rotary embeddings applied
        to the rotary portion and passthrough for the remainder.
    """
    mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()

    cos = mx.concatenate([m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))], axis=-1)[:, None, :, :]
    sin = mx.concatenate([m[i % 3] for i, m in enumerate(mx.split(sin, mrope_section, axis=-1))], axis=-1)[:, None, :, :]

    cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
    sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


class Glm4vAttention(nn.Module):
    """Multi-head attention for the GLM-4V language model.

    Supports grouped-query attention (GQA) with multimodal rotary position
    embeddings.

    Attributes:
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key-value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        rope_scaling: RoPE scaling configuration dictionary.
    """

    def __init__(self, config: Glm4VTextConfig):
        """Initializes the attention layer.

        Args:
            config: Text configuration.
        """
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads

        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope_scaling = config.rope_scaling
        self.attention_performer = AttentionPerformer(
            scale=self.scale, attn_mechanism=getattr(config, "attn_mechanism", None)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Computes multi-head attention with multimodal RoPE.

        Args:
            hidden_states: Input tensor of shape
                ``(batch_size, seq_len, hidden_size)`` or
                ``(seq_len, hidden_size)`` for paged mode.
            mask: Attention mask, a mask string, or None.
            position_embeddings: Pre-computed ``(cos, sin)`` rotary
                embeddings, or None.
            cache_view: Optional KV cache view for incremental decoding.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output tensor with the same leading dimensions as input.
        """
        lead = hidden_states.shape[:-1]

        q = self.q_proj(hidden_states).reshape(*lead, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.n_kv_heads, self.head_dim)

        if position_embeddings is not None:
            if q.ndim == 4:
                q_rope = q.transpose(0, 2, 1, 3)
                k_rope = k.transpose(0, 2, 1, 3)
            else:
                q_rope = q[None].transpose(0, 2, 1, 3)
                k_rope = k[None].transpose(0, 2, 1, 3)
            cos, sin = position_embeddings
            q_rope, k_rope = apply_multimodal_rotary_pos_emb(
                q_rope, k_rope, cos, sin, self.rope_scaling["mrope_section"]
            )
            if q.ndim == 4:
                q = q_rope.transpose(0, 2, 1, 3)
                k = k_rope.transpose(0, 2, 1, 3)
            else:
                q = q_rope.transpose(0, 2, 1, 3).squeeze(0)
                k = k_rope.transpose(0, 2, 1, 3).squeeze(0)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=None,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class Glm4vMLP(nn.Module):
    """Feed-forward network for the GLM-4V language model.

    Uses a fused gate-up projection followed by SiLU gating and a
    down projection.

    Attributes:
        gate_up_proj: Fused linear projection for gate and up paths.
        down_proj: Down projection back to hidden size.
    """

    def __init__(self, config: Glm4VTextConfig):
        """Initializes the MLP layer.

        Args:
            config: Text configuration.
        """
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        hidden_states = self.gate_up_proj(hidden_states)
        gate, up = mx.split(hidden_states, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class Glm4vDecoderLayer(nn.Module):
    """Single transformer decoder layer for the GLM-4V language model.

    Applies pre-norm attention and MLP with additional post-attention and
    post-MLP layer normalizations, plus residual connections.

    Attributes:
        self_attn: Attention sub-layer.
        mlp: Feed-forward sub-layer.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Pre-MLP normalization.
        post_self_attn_layernorm: Post-attention normalization.
        post_mlp_layernorm: Post-MLP normalization.
    """

    def __init__(self, config: Glm4VTextConfig):
        """Initializes a decoder layer.

        Args:
            config: Text configuration.
        """
        super().__init__()
        self.self_attn = Glm4vAttention(config)
        self.mlp = Glm4vMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the decoder layer forward pass.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            position_embeddings: Pre-computed rotary embeddings or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output hidden states tensor.
        """
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            position_embeddings=position_embeddings,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GLM4VModel(nn.Module):
    """Core transformer model for GLM-4V language processing.

    Contains the embedding layer, rotary embeddings, decoder stack, and
    final normalization.

    Attributes:
        vocab_size: Vocabulary size.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
        rotary_emb: Multimodal rotary embedding module.
    """

    def __init__(self, config: Glm4VTextConfig):
        """Initializes the core transformer model.

        Args:
            config: Text configuration.
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Glm4vDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GLM4VRotaryEmbedding(config)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the transformer forward pass.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            inputs_embeds: Optional pre-computed embeddings. If provided,
                ``input_ids`` is ignored for embedding lookup.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs of shape
                ``(3, batch_size, seq_len)``.
            cache_views: Optional list of KV cache views, one per layer.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Final hidden states after all layers and normalization.

        Raises:
            ValueError: If ``cache_views`` length does not match layers.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds.astype(self.norm.weight.dtype)

        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = mx.arange(seq_len)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, batch_size, 1))

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            if not (cache_views is not None and seq_len == 1):
                mask = build_attention_mask(attention_mask, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                position_embeddings=position_embeddings,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    """Language model wrapper for GLM-4V.

    Wraps ``GLM4VModel`` and adds a language modeling head, multimodal
    rope index computation, and cache initialization utilities.

    Attributes:
        args: Text configuration.
        config: Composite model configuration.
        model_type: Model type identifier string.
        model: The underlying ``GLM4VModel``.
    """

    def __init__(self, args: Glm4VTextConfig, config: Glm4VModelConfig):
        """Initializes the language model.

        Args:
            args: Text sub-configuration.
            config: Composite model configuration with vision parameters.
        """
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = GLM4VModel(args)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ):
        """Computes multimodal RoPE position indices.

        Calculates 3D (temporal, height, width) position IDs for
        interleaved text and vision tokens. Text tokens receive
        sequential positions; vision tokens receive spatial grid positions.

        Args:
            input_ids: Token IDs of shape ``(batch_size, seq_len)``.
            image_grid_thw: Optional image grid dimensions of shape
                ``(num_images, 3)`` with ``(T, H, W)`` per image.
            video_grid_thw: Optional video grid dimensions of shape
                ``(num_videos, 3)`` with ``(T, H, W)`` per video.
            attention_mask: Optional attention mask of shape
                ``(batch_size, seq_len)``.

        Returns:
            A tuple of ``(position_ids, mrope_position_deltas)`` where
            ``position_ids`` has shape ``(3, batch_size, seq_len)`` and
            ``mrope_position_deltas`` contains the position offset deltas.
        """
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_row in enumerate(total_input_ids):
                input_row = mx.where(attention_mask[i] == 1, input_row, mx.zeros_like(input_row))
                vision_start_indices = mx.sum(
                    mx.where(
                        input_row == vision_start_token_id,
                        mx.arange(input_row.shape[0]),
                        mx.zeros_like(input_row),
                    )
                )
                vision_tokens = input_row[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_row.tolist()
                llm_pos_ids_list: list[mx.array] = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)

                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w))
                    t_index = t_index.flatten()

                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w))
                    h_index = h_index.flatten()

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w))
                    w_index = w_index.flatten()

                    llm_pos_ids_list.append(mx.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))
                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(expanded_mask, expanded_positions, position_ids[:, i : i + 1, :])
                position_ids = mx.concatenate(
                    [position_ids[:, :i, :], new_positions, position_ids[:, i + 1 :, :]], axis=1
                )
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas

        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
            position_ids = mx.expand_dims(position_ids[0], axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
            max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, input_ids.shape[0], input_ids.shape[1]))
            mrope_position_deltas = mx.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        **kwargs,
    ) -> mx.array:
        """Runs the language model forward pass to produce logits.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            inputs_embeds: Optional pre-computed embeddings.
            attention_mask: Optional attention mask.
            cache_views: Optional KV cache views.
            cache_metadata: Optional paged cache metadata.
            **kwargs: Additional keyword arguments including
                ``position_ids``, ``image_grid_thw``, ``video_grid_thw``.

        Returns:
            Logits tensor of shape ``(batch_size, seq_len, vocab_size)``
            or ``(batch_size, vocab_size)`` during cached generation.
        """
        position_ids = kwargs.pop("position_ids", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if attention_mask is not None and not isinstance(attention_mask, mx.array):
            attention_mask = mx.array(attention_mask)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

        hidden_states = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes weights for loading pretrained checkpoints.

        Removes rotary embedding inverse frequency keys and optionally
        removes the LM head when using tied embeddings.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self._tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        """Returns the list of decoder layers from the underlying model."""
        return self.model.layers

    @property
    def n_kv_heads(self):
        """Returns the number of key-value heads from the text configuration."""
        return self.args.num_key_value_heads
