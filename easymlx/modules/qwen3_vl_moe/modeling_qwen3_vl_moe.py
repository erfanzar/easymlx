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

"""Qwen3-VL-MoE MLX implementation (serving/inference only).

This module provides the MLX neural network modules for the Qwen3
Vision-Language Mixture-of-Experts model. It includes a text decoder with
sparse MoE feed-forward layers, reusing the vision encoder from the
``qwen3_vl`` package, and wrapper classes for inference and generation.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCache, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from .qwen3_vl_moe_configuration import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig

CacheView = TransformerCacheView | PageCache


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Look up an activation function by name.

    Args:
        name: Name of the activation function. Supported values are
            ``"silu"``, ``"swish"``, and ``"gelu"``.

    Returns:
        The corresponding MLX activation function callable.

    Raises:
        ValueError: If the activation name is not recognized.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


class Qwen3VLMoeTextAttention(nn.Module):
    """Multi-head attention for the Qwen3-VL-MoE text decoder.

    Supports grouped-query attention (GQA), RMSNorm on Q/K projections,
    rotary position embeddings, and optional KV caching.

    Args:
        config: MoE text decoder configuration.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute multi-head attention with optional caching and masking.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask, a string sentinel, or None.
            cache_view: Optional KV cache view for incremental decoding.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Output tensor with the same leading dimensions as the input.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
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


class Qwen3VLMoeTextMLP(nn.Module):
    """Gated MLP (SwiGLU-style) for the Qwen3-VL-MoE text decoder.

    Args:
        config: MoE text decoder configuration.
        intermediate_size: Override for the intermediate size. Uses
            ``config.intermediate_size`` if None.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP.

        Args:
            hidden_states: Input tensor of shape ``[..., hidden_size]``.

        Returns:
            Output tensor of shape ``[..., hidden_size]``.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3VLMoeSparseBlock(nn.Module):
    """Sparse Mixture-of-Experts feed-forward block for the text decoder.

    Routes each token to a subset of experts via a top-k router and
    combines expert outputs weighted by routing scores.

    Args:
        config: MoE text decoder configuration specifying MoE parameters.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
            norm_topk_prob=config.norm_topk_prob,
        )
        self.experts = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and combine outputs.

        Args:
            hidden_states: Input tensor of shape ``[..., hidden_size]``.

        Returns:
            Combined expert output of shape ``[..., hidden_size]``.
        """
        inds, scores = self.router(hidden_states)
        out = self.experts(hidden_states, inds)
        return (out * scores[..., None]).sum(axis=-2).astype(out.dtype)


class Qwen3VLMoeTextLayer(nn.Module):
    """Single transformer decoder layer for the Qwen3-VL-MoE text model.

    Applies pre-norm attention followed by pre-norm feed-forward (either dense
    MLP or sparse MoE), both with residual connections.

    Args:
        config: MoE text decoder configuration.
        layer_idx: Zero-based index of this layer.
        use_sliding: Whether this layer uses sliding window attention.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int, *, use_sliding: bool = False):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Qwen3VLMoeTextAttention(config)
        use_moe = (layer_idx % max(config.decoder_sparse_step, 1) == 0) and layer_idx not in config.mlp_only_layers
        self.mlp = Qwen3VLMoeSparseBlock(config) if use_moe else Qwen3VLMoeTextMLP(config)
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
        """Apply the decoder layer.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask, a string sentinel, or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional page metadata.

        Returns:
            Output tensor with the same shape as the input.
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


class Qwen3VLMoeTextModel(nn.Module):
    """Text decoder model for Qwen3-VL-MoE with sparse MoE layers.

    Embeds input tokens, applies a stack of transformer decoder layers
    (some with MoE feed-forward), and produces normalized hidden states.

    Args:
        config: MoE text decoder configuration.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3VLMoeTextLayer(config, idx, use_sliding=layer_type == "sliding_attention")
            for idx, layer_type in enumerate(config.layer_types)
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
        """Run the MoE text decoder forward pass.

        Args:
            input_ids: Token IDs of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Optional pre-computed embeddings.
            cache_views: Optional per-layer KV cache views.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Normalized hidden states.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of layers.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeModel(EasyMLXBaseModule):
    """Base vision-language model for Qwen3-VL-MoE.

    Combines a vision tower (reused from Qwen3-VL), a vision-to-text
    projection, and an MoE text language model.

    Args:
        config: Top-level model configuration.
    """

    config_class = Qwen3VLMoeConfig

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__(config)
        self.model_config = config
        self.vision_tower = Qwen3VLVisionModel(config.vision_config)
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.language_model = Qwen3VLMoeTextModel(config.text_config)

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


@register_module(task_type=TaskType.CAUSAL_LM, config=Qwen3VLMoeConfig, model_type="qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(EasyMLXBaseModule):
    """Causal language model with a language modeling head for Qwen3-VL-MoE.

    Wraps ``Qwen3VLMoeModel`` and adds a linear head (or uses tied
    embeddings) to produce next-token logits.

    Args:
        config: Top-level model configuration.
    """

    config_class = Qwen3VLMoeConfig

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__(config)
        self.model = Qwen3VLMoeModel(config)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            text_config = config.text_config
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


__all__ = ("Qwen3VLMoeForConditionalGeneration", "Qwen3VLMoeModel")
