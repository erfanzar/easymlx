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

"""Nemotron MLX implementation (serving/inference only).

Structure mirrors upstream Nemotron:
  NemotronConfig -> NemotronAttention -> NemotronMLP
  -> NemotronDecoderLayer -> NemotronModel -> NemotronForCausalLM

Nemotron uses ReLU-squared activation, NemotronLayerNorm1P (weight + 1),
and partial RoPE (applied to first portion of head dimensions).
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .nemotron_configuration import NemotronConfig

CacheView = TransformerCacheView | PageCacheView


@partial(mx.compile, shapeless=True)
def relu_squared(x: mx.array) -> mx.array:
    """ReLU-squared activation: ``relu(x)^2``.

    Compiled with ``mx.compile`` for performance. Produces sparser
    activations than standard ReLU.

    Args:
        x (mx.array): Input tensor of any shape.

    Returns:
        mx.array: Element-wise ``max(0, x)^2``.
    """
    return nn.relu(x).square()


class NemotronLayerNorm1P(nn.LayerNorm):
    """LayerNorm with weights offset by +1 (saved as offset from 1).

    Checkpoint stores ``weight = actual_weight - 1``, so at runtime we
    compute ``layer_norm(x, weight + 1, bias)``. This allows the weights
    to be initialized near zero while the effective scaling starts at 1.

    Example:
        >>> norm = NemotronLayerNorm1P(4096)
        >>> out = norm(mx.zeros((1, 128, 4096)))
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Apply LayerNorm with weight offset by +1.

        Args:
            x (mx.array): Input tensor.

        Returns:
            mx.array: Normalized tensor.
        """
        weight = self.weight + 1 if "weight" in self else None
        bias = self.bias if "bias" in self else None
        return mx.fast.layer_norm(x, weight, bias, self.eps)


class NemotronAttention(nn.Module):
    """Multi-head attention with partial RoPE for Nemotron.

    Only the first ``partial_rotary_factor * head_dim`` dimensions of
    each head receive rotary positional embeddings. The remaining
    dimensions are left unrotated.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.

    Example:
        >>> config = NemotronConfig(partial_rotary_factor=0.5)
        >>> attn = NemotronAttention(config)
        >>> out = attn(mx.zeros((1, 128, 6144)))
    """

    def __init__(self, config: NemotronConfig):
        """Initialize Nemotron attention with partial RoPE.

        Args:
            config (NemotronConfig): Model configuration with
                ``partial_rotary_factor`` controlling how many head
                dimensions receive RoPE.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Partial RoPE: only first (partial_rotary_factor * head_dim) dimensions
        rope_dims = int(config.partial_rotary_factor * self.head_dim)

        # Handle rope_scaling for Nemotron (supports linear scaling)
        rope_scaling = config.rope_scaling
        if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        self.rope = get_rope(
            dims=rope_dims,
            base=config.rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
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
        """Compute attention forward pass with partial RoPE.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
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


class NemotronMLP(nn.Module):
    """MLP with ReLU-squared activation for Nemotron.

    Computes ``down_proj(relu_squared(up_proj(x)))`` -- a simpler non-gated
    architecture unlike SwiGLU.

    Attributes:
        down_proj: Down-projection linear layer.
        up_proj: Up-projection linear layer.

    Example:
        >>> config = NemotronConfig()
        >>> mlp = NemotronMLP(config)
        >>> out = mlp(mx.zeros((1, 128, 6144)))
    """

    def __init__(self, config: NemotronConfig):
        """Initialize Nemotron ReLU^2 MLP.

        Args:
            config (NemotronConfig): Model configuration.
        """
        super().__init__()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute ReLU^2 MLP forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(relu_squared(self.up_proj(hidden_states)))


class NemotronDecoderLayer(nn.Module):
    """Single Nemotron decoder layer with NemotronLayerNorm1P.

    Uses LayerNorm with +1 weight offset (NemotronLayerNorm1P) for
    pre-norm before attention and MLP sub-layers.

    Attributes:
        self_attn: Nemotron attention with partial RoPE.
        mlp: ReLU^2 MLP module.
        input_layernorm: Pre-attention NemotronLayerNorm1P.
        post_attention_layernorm: Pre-MLP NemotronLayerNorm1P.

    Example:
        >>> config = NemotronConfig()
        >>> layer = NemotronDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 6144)))
    """

    def __init__(self, config: NemotronConfig):
        """Initialize Nemotron decoder layer.

        Args:
            config (NemotronConfig): Model configuration.
        """
        super().__init__()
        self.self_attn = NemotronAttention(config)
        self.mlp = NemotronMLP(config)
        self.input_layernorm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)

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
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=NemotronConfig, model_type="nemotron")
class NemotronModel(EasyMLXBaseModule):
    """Base Nemotron transformer model with ReLU^2, partial RoPE, and LayerNorm1P.

    Attributes:
        config_class: Associated configuration class (``NemotronConfig``).
        embed_tokens: Token embedding layer.
        layers: List of Nemotron decoder layers.
        norm: Final NemotronLayerNorm1P.

    Example:
        >>> config = NemotronConfig()
        >>> model = NemotronModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronConfig

    def __init__(self, config: NemotronConfig):
        """Initialize Nemotron base model.

        Args:
            config (NemotronConfig): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = NemotronLayerNorm1P(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the Nemotron base model.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings.
            cache_views (list[CacheView] | None): Per-layer KV cache views.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Normalized hidden states of shape ``(B, L, D)``.

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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove unused rotary embedding inverse frequency buffers.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Filtered weight dictionary.
        """
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}


@register_module(task_type=TaskType.CAUSAL_LM, config=NemotronConfig, model_type="nemotron")
class NemotronForCausalLM(BaseCausalLMModule[NemotronModel, NemotronConfig]):
    """Nemotron causal language model with LM head.

    Attributes:
        config_class: Associated configuration class (``NemotronConfig``).

    Example:
        >>> config = NemotronConfig()
        >>> model = NemotronForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronConfig

    def __init__(self, config: NemotronConfig):
        """Initialize Nemotron causal LM.

        Args:
            config (NemotronConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=NemotronModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("NemotronForCausalLM", "NemotronModel")
