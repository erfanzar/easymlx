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

"""GLM MLX implementation for serving and inference.

This module provides the MLX-native implementation of the GLM (General Language
Model) architecture, including the attention layer, MLP, decoder layer, base
model, and causal LM wrapper.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCache, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .glm_configuration import GlmConfig

CacheView = TransformerCacheView | PageCache


def _get_activation(name: str) -> tp.Callable[[mx.array], mx.array]:
    """Resolve an activation function by name.

    Args:
        name: Activation function name. Supported values: ``"silu"``,
            ``"swish"``, ``"gelu"``.

    Returns:
        The corresponding MLX activation function.

    Raises:
        ValueError: If the activation name is not supported.
    """
    name = name.lower()
    if name in {"silu", "swish"}:
        return nn.silu
    if name == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation: {name!r}")


class GlmMLP(nn.Module):
    """GLM feed-forward MLP layer with gated activation.

    Uses a fused gate-up projection followed by a down projection with
    the configured activation function (default: SiLU).

    Attributes:
        gate_up_proj: Fused linear layer projecting to 2x intermediate size.
        down_proj: Linear layer projecting back to hidden size.
        act_fn: Activation function applied to the gate.
    """

    def __init__(self, config: GlmConfig):
        """Initialize the GLM MLP.

        Args:
            config: GLM configuration containing ``hidden_size``,
                ``intermediate_size``, and ``hidden_act``.
        """
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the gated MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(up * self.act_fn(gate))


class GlmAttention(nn.Module):
    """GLM multi-head attention layer with Grouped Query Attention (GQA).

    Implements multi-head attention with separate query, key, and value
    projections, rotary positional embeddings, and an output projection.
    Supports GQA when ``num_key_value_heads < num_attention_heads``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimensionality of each attention head.
        rotary_dim: Number of dimensions to apply RoPE to.
        hidden_size: Total hidden dimensionality.
        q_proj: Query projection linear layer.
        k_proj: Key projection linear layer.
        v_proj: Value projection linear layer.
        o_proj: Output projection linear layer.
        rope: Rotary positional embedding module, or None if rotary_dim is 0.
        attention_performer: Attention computation module.
    """

    def __init__(self, config: GlmConfig):
        """Initialize GLM attention.

        Args:
            config: GLM configuration.

        Raises:
            ValueError: If ``hidden_size != num_attention_heads * head_dim``.
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.rotary_dim = int(config.rotary_dim)
        self.hidden_size = int(config.hidden_size)

        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim for GLM models")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

        self.rope = (
            get_rope(
                dims=self.rotary_dim,
                base=config.rope_theta,
                traditional=False,
                scaling_config=getattr(config, "rope_scaling", None),
                max_position_embeddings=config.max_position_embeddings,
            )
            if self.rotary_dim > 0
            else None
        )
        self.attention_performer = AttentionPerformer(scale=self.head_dim**-0.5)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute multi-head attention.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask. Can be a tensor, a string identifier
                (e.g., ``"causal"``), or None for no masking.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged cache metadata for paged attention.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
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


class GlmDecoderLayer(nn.Module):
    """Single GLM transformer decoder layer.

    Consists of pre-norm self-attention followed by pre-norm MLP, both
    with residual connections.

    Attributes:
        input_layernorm: RMS normalization before self-attention.
        post_attention_layernorm: RMS normalization before MLP.
        self_attn: Multi-head attention sub-layer.
        mlp: Feed-forward MLP sub-layer.
    """

    def __init__(self, config: GlmConfig):
        """Initialize a GLM decoder layer.

        Args:
            config: GLM configuration.
        """
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GlmAttention(config)
        self.mlp = GlmMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask for the self-attention sub-layer.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged cache metadata.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=GlmConfig, model_type="glm")
class GlmModel(EasyMLXBaseModule):
    """GLM base transformer model (decoder-only).

    Processes input token IDs through an embedding layer, a stack of
    transformer decoder layers, and a final RMS normalization.

    Attributes:
        config_class: The configuration class for this model (``GlmConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``GlmDecoderLayer`` instances.
        norm: Final RMS normalization layer.
    """

    config_class = GlmConfig

    def __init__(self, config: GlmConfig):
        """Initialize the GLM base model.

        Args:
            config: GLM configuration specifying model architecture.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GlmDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the GLM base model.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``
                or ``(num_tokens,)`` for paged serving.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed input embeddings. Overrides
                ``input_ids`` if provided.
            cache_views: Per-layer KV cache views for incremental decoding.
            cache_metadata: Paged cache metadata for paged attention.

        Returns:
            Normalized hidden states of shape
            ``(batch_size, seq_len, hidden_size)`` or
            ``(num_tokens, hidden_size)``.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=GlmConfig, model_type="glm")
class GlmForCausalLM(BaseCausalLMModule[GlmModel, GlmConfig]):
    """GLM model with a causal language modeling head.

    Wraps ``GlmModel`` with an LM head that projects hidden states to
    vocabulary logits. Supports weight tying based on configuration.

    Attributes:
        config_class: The configuration class for this model (``GlmConfig``).
    """

    config_class = GlmConfig

    def __init__(self, config: GlmConfig):
        """Initialize GLM for causal language modeling.

        Args:
            config: GLM configuration. The ``tie_word_embeddings`` attribute
                determines whether weight tying is enabled.
        """
        super().__init__(
            config=config,
            base_model_class=GlmModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("GlmForCausalLM", "GlmModel")
