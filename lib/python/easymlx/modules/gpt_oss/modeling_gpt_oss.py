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

"""GPT-OSS MLX implementation (serving/inference only).

This module implements the GPT-OSS Mixture-of-Experts transformer on MLX,
featuring SwiGLU-based expert routing with activation clamping, sliding
window attention, and alternating attention patterns.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .gpt_oss_configuration import GptOssConfig

CacheView = TransformerCacheView | PageCacheView


def mlx_topk(values: mx.array, k: int, axis: int = -1) -> tuple[mx.array, mx.array]:
    """Selects top-k values and their indices along an axis.

    Uses ``argpartition`` for efficient partial sorting.

    Args:
        values: Input tensor.
        k: Number of top values to select.
        axis: Axis along which to select. Defaults to -1.

    Returns:
        A tuple of ``(top_k_values, top_k_indices)``.
    """
    partitioned_indices = mx.argpartition(values, kth=-k, axis=axis)
    top_k_indices = partitioned_indices[..., -k:]
    top_k_values = mx.take_along_axis(values, top_k_indices, axis=axis)
    return top_k_values, top_k_indices


def swiglu(x_linear: mx.array, x_glu: mx.array, *, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    """Applies SwiGLU activation with clamping.

    Computes ``x_glu * sigmoid(alpha * x_glu) * (x_linear + 1)`` with
    clamped inputs for numerical stability.

    Args:
        x_linear: Linear path input tensor.
        x_glu: Gated path input tensor.
        alpha: Sigmoid scaling factor. Defaults to 1.702.
        limit: Clamping limit for activations. Defaults to 7.0.

    Returns:
        Activated output tensor.
    """
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    sig = mx.sigmoid(alpha * x_glu)
    return x_glu * sig * (x_linear + 1)


class SwiGLU(nn.Module):
    """SwiGLU activation module with configurable clamping limit.

    Attributes:
        limit: Clamping limit for activations.
    """

    def __init__(self, *, limit: float = 7.0):
        """Initializes the SwiGLU module.

        Args:
            limit: Activation clamping limit. Defaults to 7.0.
        """
        super().__init__()
        self.limit = float(limit)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        """Applies SwiGLU activation.

        Args:
            x: Linear path input.
            gate: Gated path input.

        Returns:
            Activated output.
        """
        return swiglu(x, gate, limit=self.limit)


class AttentionBlock(nn.Module):
    """Multi-head attention block for GPT-OSS.

    Supports GQA with RoPE and optional YaRN scaling.

    Attributes:
        head_dim: Per-head dimensionality.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        num_key_value_groups: Number of query heads per KV head.
        scale: Attention scaling factor.
    """

    def __init__(self, config: GptOssConfig):
        """Initializes the attention block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

        self.scale = self.head_dim**-0.5
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
        """Computes multi-head attention.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output tensor with the same leading dimensions as input.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, -1, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, -1, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, -1, self.head_dim)
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


class MLPBlock(nn.Module):
    """Mixture-of-Experts MLP block for GPT-OSS.

    Routes tokens to top-k experts using softmax routing with a
    SwiGLU activation function.

    Attributes:
        num_local_experts: Total number of local experts.
        num_experts_per_tok: Experts activated per token.
        experts: SwitchGLU layer containing all experts.
        router: Linear routing projection.
    """

    def __init__(self, config: GptOssConfig):
        """Initializes the MoE MLP block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            num_experts=config.num_local_experts,
            activation=SwiGLU(limit=config.mlp_activations_limit),
            bias=True,
        )
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Applies MoE routing and expert computation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Weighted sum of expert outputs.
        """
        gate = self.router(hidden_states)
        experts, indices = mlx_topk(gate, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(experts, axis=-1, precise=True)

        hidden_states = self.experts(hidden_states, indices)
        hidden_states = hidden_states * mx.expand_dims(expert_weights, axis=-1)
        return hidden_states.sum(axis=-2)


class TransformerBlock(nn.Module):
    """Single transformer block for GPT-OSS.

    Applies pre-norm attention and MoE MLP with residual connections.

    Attributes:
        self_attn: Attention sub-layer.
        mlp: MoE MLP sub-layer.
        input_layernorm: Pre-attention normalization.
        post_attention_layernorm: Pre-MLP normalization.
    """

    def __init__(self, config: GptOssConfig):
        """Initializes a transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = AttentionBlock(config)
        self.mlp = MLPBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the transformer block forward pass.

        Args:
            hidden_states: Input tensor.
            mask: Attention mask or None.
            cache_view: Optional KV cache view.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Output hidden states tensor.
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


@register_module(task_type=TaskType.BASE_MODULE, config=GptOssConfig, model_type="gpt_oss")
class GptOssModel(EasyMLXBaseModule):
    """Base GPT-OSS transformer model.

    Features alternating sliding window and full attention layers with
    Mixture-of-Experts MLP blocks.

    Attributes:
        config_class: The configuration class (``GptOssConfig``).
        embed_tokens: Token embedding layer.
        norm: Final RMS normalization.
        layer_types: Per-layer attention type.
        layers: List of transformer blocks.
        window_size: Sliding window size.
    """

    config_class = GptOssConfig

    def __init__(self, config: GptOssConfig):
        """Initializes the base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types = config.layer_types
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.window_size = config.sliding_window

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Runs the transformer forward pass with sliding window support.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            input_embeddings: Optional pre-computed embeddings.
            cache_views: Optional KV cache views.
            cache_metadata: Optional paged cache metadata.

        Returns:
            Final hidden states.

        Raises:
            ValueError: If ``cache_views`` length mismatches layer count.
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
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                sliding_mask = build_attention_mask(
                    attention_mask_arr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    window_size=self.window_size,
                )

        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types, strict=False)):
            layer_mask = sliding_mask if layer_type != "full_attention" else mask
            layer_cache = None if cache_views is None else cache_views[layer_idx]

            layer_metadata = cache_metadata
            if cache_metadata is not None and layer_type != "full_attention":
                layer_metadata = cache_metadata.with_sliding_window(self.window_size)

            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=GptOssConfig, model_type="gpt_oss")
class GptOssForCausalLM(BaseCausalLMModule[GptOssModel, GptOssConfig]):
    """GPT-OSS model with a causal language modeling head.

    Extends ``BaseCausalLMModule`` with GPT-OSS-specific weight sanitization
    to handle fused gate-up projections and quantized weight formats.

    Attributes:
        config_class: The configuration class (``GptOssConfig``).
    """

    config_class = GptOssConfig

    def __init__(self, config: GptOssConfig):
        """Initializes the causal LM model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=GptOssModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitizes weights by splitting fused gate-up projections.

        Handles conversion from fused ``gate_up_proj`` format (including
        quantized ``_blocks``/``_scales`` variants) into separate
        ``gate_proj`` and ``up_proj`` parameters.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weight dictionary with split projections.
        """
        if any(gate_proj.weight in k for k in weights.keys()):
            return super().sanitize(weights)

        new_weights: dict[str, mx.array] = {}
        for key, value in weights.items():
            if "gate_up_proj" in key and "bias" not in key:
                if "_blocks" in key:
                    value = value.view(mx.uint32).flatten(-2)
                    key = key.replace("_blocks", ".weight")
                if "_scales" in key:
                    key = key.replace("_scales", ".scales")
                new_weights[key.replace("gate_up_proj", "gate_proj")] = mx.contiguous(value[..., ::2, :])
                new_weights[key.replace("gate_up_proj", "up_proj")] = mx.contiguous(value[..., 1::2, :])
            elif "down_proj" in key and "bias" not in key:
                if "_blocks" in key:
                    value = value.view(mx.uint32).flatten(-2)
                    key = key.replace("_blocks", ".weight")
                if "_scales" in key:
                    key = key.replace("_scales", ".scales")
                new_weights[key] = value
            elif "gate_up_proj_bias" in key:
                new_weights[key.replace("gate_up_proj_bias", "gate_proj.bias")] = mx.contiguous(value[..., ::2])
                new_weights[key.replace("gate_up_proj_bias", "up_proj.bias")] = mx.contiguous(value[..., 1::2])
            elif "down_proj_bias" in key:
                new_weights[key.replace("down_proj_bias", "down_proj.bias")] = value
            else:
                new_weights[key] = value

        return super().sanitize(new_weights)


__all__ = ("GptOssForCausalLM", "GptOssModel")
