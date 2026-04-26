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

"""Gemma2 MLX implementation (serving/inference only).

Structure:
  Gemma2Config -> Gemma2Attention -> Gemma2MLP -> Gemma2DecoderLayer -> Gemma2Model -> Gemma2ForCausalLM

Key differences from Gemma:
  - 4 norms per layer (input, post_attention, pre_feedforward, post_feedforward)
  - Attention logit softcapping
  - Final logit softcapping
  - query_pre_attn_scalar for attention scale
  - GELU approx activation
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .gemma2_configuration import Gemma2Config

CacheView = TransformerCacheView | PageCacheView


class Gemma2Attention(nn.Module):
    """Multi-head attention for the Gemma2 model with softcapping.

    Uses ``query_pre_attn_scalar`` for attention scaling instead of
    ``1/sqrt(head_dim)``, computing scale as
    ``1/sqrt(query_pre_attn_scalar)``. Stores ``attn_logit_softcapping``
    for downstream softcapping of attention logits.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Attention scale derived from ``query_pre_attn_scalar``.
        attn_logit_softcapping: Softcapping value for attention logits.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> attn = Gemma2Attention(Gemma2Config(hidden_size=64, num_attention_heads=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma2Config):
        """Initialize Gemma2 attention layer.

        Args:
            config: Model configuration containing attention parameters
                including ``query_pre_attn_scalar`` and
                ``attn_logit_softcapping``.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = 1.0 / (config.query_pre_attn_scalar**0.5)
        self.attn_logit_softcapping = config.attn_logit_softcapping

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

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
        """Compute multi-head attention with RoPE.

        Args:
            hidden_states: Input tensor of shape
                ``[batch, seq_len, hidden_size]``.
            mask: Attention mask (float array, string, or None).
            cache_view: Optional KV cache for autoregressive decoding.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Output tensor of shape ``[batch, seq_len, hidden_size]``.
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


class Gemma2MLP(nn.Module):
    """Gated feed-forward network with GELU approximate activation for Gemma2.

    Uses the approximate GELU variant (``gelu_approx``) as the gating
    function: ``down_proj(gelu_approx(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear projection for the gating branch.
        up_proj: Linear projection for the value branch.
        down_proj: Linear projection back to hidden size.

    Example::

        >>> mlp = Gemma2MLP(Gemma2Config(hidden_size=64, intermediate_size=128))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma2Config):
        """Initialize Gemma2 MLP.

        Args:
            config: Model configuration with ``hidden_size``,
                ``intermediate_size``, and ``mlp_bias``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply gated MLP with approximate GELU activation.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Output of the same shape.
        """
        return self.down_proj(nn.gelu_approx(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Gemma2DecoderLayer(nn.Module):
    """Single transformer decoder layer for the Gemma2 model.

    Uses four RMSNorm layers per block: ``input_layernorm`` and
    ``post_attention_layernorm`` wrap the attention sub-layer, while
    ``pre_feedforward_layernorm`` and ``post_feedforward_layernorm``
    wrap the MLP sub-layer. The post-norms are applied to the
    sub-layer output before the residual addition.

    Attributes:
        self_attn: Gemma2 attention with softcapping.
        mlp: GELU-approximate gated MLP.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm after attention output.
        pre_feedforward_layernorm: RMSNorm before MLP.
        post_feedforward_layernorm: RMSNorm after MLP output.

    Example::

        >>> layer = Gemma2DecoderLayer(Gemma2Config(hidden_size=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma2Config):
        """Initialize Gemma2 decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = Gemma2Attention(config)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through one Gemma2 decoder layer.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache for decoding.
            cache_metadata: Optional page metadata for paged attention.

        Returns:
            Hidden states after attention and MLP with residuals.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Gemma2Config, model_type="gemma2")
class Gemma2Model(EasyMLXBaseModule):
    """Base Gemma2 transformer with scaled embeddings and 4-norm layers.

    Implements the Gemma2 architecture with embedding scaling by
    ``sqrt(hidden_size)``, four normalization layers per decoder block,
    and ``query_pre_attn_scalar``-based attention scaling.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: Stack of ``Gemma2DecoderLayer`` instances.
        norm: Final RMSNorm applied to transformer output.

    Example::

        >>> model = Gemma2Model(Gemma2Config(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = Gemma2Config

    def __init__(self, config: Gemma2Config):
        """Initialize Gemma2 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the full Gemma2 transformer.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides token lookup).
            cache_views: Per-layer KV cache views for decoding.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Normalized hidden states of shape
            ``[batch, seq_len, hidden_size]``.

        Raises:
            ValueError: If ``cache_views`` length does not match layers.
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

        hidden_states = hidden_states * (self.config.hidden_size**0.5)

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


@register_module(task_type=TaskType.CAUSAL_LM, config=Gemma2Config, model_type="gemma2")
class Gemma2ForCausalLM(BaseCausalLMModule[Gemma2Model, Gemma2Config]):
    """Gemma2 causal language model with final logit softcapping.

    Wraps ``Gemma2Model`` and applies ``final_logit_softcapping`` to the
    output logits using ``tanh(logits / cap) * cap``. This bounds the
    output logits to ``[-cap, cap]`` to stabilize generation.

    Attributes:
        config_class: ``Gemma2Config``.
        model: The underlying ``Gemma2Model`` base model.

    Example::

        >>> model = Gemma2ForCausalLM(Gemma2Config(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = Gemma2Config

    def __init__(self, config: Gemma2Config):
        """Initialize Gemma2 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits with softcapping.

        Applies ``final_logit_softcapping`` if nonzero, bounding logits
        to ``[-cap, cap]`` via ``tanh(logits / cap) * cap``.

        Args:
            hidden_states: Transformer output of shape
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Logits of shape ``[batch, seq_len, vocab_size]``.
        """
        logits = super().compute_lm_logits(hidden_states)
        if self.config.final_logit_softcapping:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


__all__ = ("Gemma2ForCausalLM", "Gemma2Model")
