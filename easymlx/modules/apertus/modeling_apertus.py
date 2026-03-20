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

"""Apertus MLX model implementation for serving and inference.

Apertus features:
  - XieLU activation (learnable piecewise-quadratic / exponential)
  - Non-gated MLP (up -> activation -> down)
  - QK-norm before RoPE
  - Separate attention_layernorm / feedforward_layernorm naming
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

from .apertus_configuration import ApertusConfig

CacheView = TransformerCacheView | PageCacheView


# ---------------------------------------------------------------------------
# XieLU activation (matches upstream mlx-lm apertus)
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def _xielu(x: mx.array, alpha_p: mx.array, alpha_n: mx.array, beta: mx.array, eps: mx.array) -> mx.array:
    """Compute the XieLU activation element-wise.

    For positive inputs: ``softplus(alpha_p) * x^2 + beta * x``.
    For negative inputs: ``(expm1(min(x, eps)) - x) * (beta + softplus(alpha_n)) + beta * x``.

    Args:
        x: Input tensor.
        alpha_p: Learnable positive-region curvature parameter (pre-softplus).
        alpha_n: Learnable negative-region curvature parameter (pre-softplus).
        beta: Linear slope parameter.
        eps: Clamping epsilon for numerical stability on the negative branch.

    Returns:
        Activated tensor of the same shape as ``x``.
    """
    alpha_p_ = nn.softplus(alpha_p)
    alpha_n_ = beta + nn.softplus(alpha_n)
    return mx.where(
        x > 0,
        alpha_p_ * mx.square(x) + beta * x,
        (mx.expm1(mx.minimum(x, eps)) - x) * alpha_n_ + beta * x,
    )


class XieLU(nn.Module):
    """Learnable piecewise-quadratic/exponential activation used by Apertus.

    Combines a quadratic positive branch with an exponential-like
    negative branch, both with learnable curvature parameters.

    Attributes:
        alpha_p: Positive-region curvature (stored in pre-softplus space).
        alpha_n: Negative-region curvature (stored in pre-softplus space).
        beta: Linear slope parameter.
        eps: Clamping epsilon for numerical stability.

    Example::

        >>> act = XieLU()
        >>> out = act(mx.array([-1.0, 0.0, 1.0]))
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
    ):
        """Initialize the XieLU activation.

        Args:
            alpha_p_init: Initial value for the positive curvature
                (in output space, converted to pre-softplus). Defaults to 0.8.
            alpha_n_init: Initial value for the negative curvature
                (in output space). Defaults to 0.8.
            beta: Linear slope parameter. Defaults to 0.5.
            eps: Clamping epsilon for the negative branch. Defaults to -1e-6.
        """
        super().__init__()
        alpha_p_tensor = mx.array(alpha_p_init)
        alpha_n_tensor = mx.array(alpha_n_init - beta)
        self.alpha_p = mx.log(mx.exp(alpha_p_tensor) - 1)
        self.alpha_n = mx.log(mx.exp(alpha_n_tensor) - 1)
        self.beta = mx.array(beta)
        self.eps = mx.array(eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply XieLU activation element-wise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Activated tensor of the same shape.
        """
        return _xielu(x, self.alpha_p, self.alpha_n, self.beta, self.eps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert. If ``None``, returns ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class ApertusAttention(nn.Module):
    """Grouped-query attention for Apertus with QK-norm before RoPE.

    Applies RMSNorm to Q and K before rotary position embedding.
    Uses bias-free projections by default.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor (``head_dim ** -0.5``).
        q_proj: Query linear projection.
        k_proj: Key linear projection.
        v_proj: Value linear projection.
        o_proj: Output linear projection.
        q_norm: RMSNorm applied to queries.
        k_norm: RMSNorm applied to keys.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = ApertusConfig(hidden_size=2048, num_attention_heads=16)
        >>> attn = ApertusAttention(config)
    """

    def __init__(self, config: ApertusConfig):
        """Initialize Apertus attention.

        Args:
            config: Model configuration containing attention hyperparameters.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
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
        """Run attention forward pass with QK-norm and RoPE.

        Args:
            hidden_states: Input hidden states of shape
                ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        # QK-norm before RoPE
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


# ---------------------------------------------------------------------------
# MLP (non-gated, XieLU activation)
# ---------------------------------------------------------------------------


class ApertusMLP(nn.Module):
    """Non-gated MLP with XieLU activation for Apertus.

    Unlike SwiGLU, this is a simple two-layer MLP:
    ``down_proj(xielu(up_proj(x)))``.

    Attributes:
        up_proj: Linear up projection.
        down_proj: Linear down projection.
        act_fn: XieLU activation function.

    Example::

        >>> config = ApertusConfig(hidden_size=2048, intermediate_size=8192)
        >>> mlp = ApertusMLP(config)
    """

    def __init__(self, config: ApertusConfig):
        """Initialize the Apertus MLP.

        Args:
            config: Model configuration with ``hidden_size``,
                ``intermediate_size``, and ``mlp_bias``.
        """
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = XieLU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply the non-gated MLP with XieLU activation.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class ApertusDecoderLayer(nn.Module):
    """Single Apertus decoder layer with separate attention and feedforward norms.

    Uses ``attention_layernorm`` before attention and ``feedforward_layernorm``
    before MLP, with residual connections.

    Attributes:
        self_attn: QK-norm attention module.
        mlp: Non-gated MLP with XieLU activation.
        attention_layernorm: RMSNorm before attention.
        feedforward_layernorm: RMSNorm before MLP.

    Example::

        >>> config = ApertusConfig(hidden_size=2048)
        >>> layer = ApertusDecoderLayer(config)
    """

    def __init__(self, config: ApertusConfig):
        """Initialize the Apertus decoder layer.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__()
        self.self_attn = ApertusAttention(config)
        self.mlp = ApertusMLP(config)
        self.attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the decoder layer forward pass.

        Args:
            hidden_states: Input hidden states of shape
                ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output hidden states of the same shape.
        """
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.feedforward_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.BASE_MODULE, config=ApertusConfig, model_type="apertus")
class ApertusModel(EasyMLXBaseModule):
    """Base Apertus transformer model for inference.

    Implements a decoder-only transformer with QK-norm attention,
    XieLU activation in a non-gated MLP, and RMSNorm normalization.

    Attributes:
        config_class: The configuration class (``ApertusConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``ApertusDecoderLayer`` decoder blocks.
        norm: Final RMS normalization.

    Example::

        >>> config = ApertusConfig(vocab_size=32000, hidden_size=2048)
        >>> model = ApertusModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = ApertusConfig

    def __init__(self, config: ApertusConfig):
        """Initialize the Apertus base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [ApertusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the transformer forward pass.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``
                or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings to use instead of
                ``input_ids``.
            cache_views: Per-layer KV cache views. Length must match
                ``num_hidden_layers``.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states tensor of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the
                number of layers.
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
                attention_mask_arr = _as_int_array(attention_mask)
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
        """Transform checkpoint weights for compatibility.

        Removes rotary embedding buffers and squeezes XieLU learnable
        parameters (``alpha_p``, ``alpha_n``) from checkpoint shape
        to scalar tensors.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k or "rope.inv_freq" in k:
                continue
            if k.endswith("alpha_p") or k.endswith("alpha_n"):
                v = v.squeeze()
            sanitized[k] = v
        return sanitized


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.CAUSAL_LM, config=ApertusConfig, model_type="apertus")
class ApertusForCausalLM(BaseCausalLMModule[ApertusModel, ApertusConfig]):
    """Apertus model with a causal language modeling head.

    Wraps ``ApertusModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``ApertusConfig``).

    Example::

        >>> config = ApertusConfig(vocab_size=32000, hidden_size=2048)
        >>> model = ApertusForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
        >>> output.logits.shape
        (1, 3, 32000)
    """

    config_class = ApertusConfig

    def __init__(self, config: ApertusConfig):
        """Initialize the Apertus causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=ApertusModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("ApertusForCausalLM", "ApertusModel")
