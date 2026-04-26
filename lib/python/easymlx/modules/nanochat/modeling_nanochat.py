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

"""Nanochat MLX implementation (serving/inference only).

Structure mirrors EasyDeL's nanochat:
  NanochatConfig -> NanochatAttention -> NanochatMLP -> NanochatDecoderLayer
  -> NanochatModel -> NanochatForCausalLM

Nanochat features functional (parameter-free) RMSNorm, Q/K RMSNorm after
RoPE, ReLU^2 activation, and logit soft-capping.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from .nanochat_configuration import NanochatConfig

CacheView = TransformerCacheView | PageCacheView


def _functional_rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Apply RMSNorm without learnable parameters.

    Unlike ``nn.RMSNorm``, this function has no trainable weight or bias.
    It normalizes by the root-mean-square of the input.

    Args:
        x (mx.array): Input tensor of any shape.
        eps (float): Epsilon for numerical stability.

    Returns:
        mx.array: Normalized tensor with same shape as input.
    """
    return mx.fast.rms_norm(x, None, eps)


def _apply_rotary_emb(x: mx.array, offset: int, freqs: mx.array) -> mx.array:
    """Apply RoPE using precomputed negated frequencies.

    Args:
        x (mx.array): Input tensor of shape ``(B, H, L, D)`` or similar.
        offset (int): Position offset for autoregressive decoding.
        freqs (mx.array): Precomputed negated frequency array.

    Returns:
        mx.array: Tensor with rotary embeddings applied.
    """
    head_dim = x.shape[-1]
    return mx.fast.rope(
        x,
        dims=head_dim,
        traditional=False,
        base=None,
        freqs=freqs,
        scale=1.0,
        offset=offset,
    )


class NanochatAttention(nn.Module):
    """Nanochat attention with Q/K RMSNorm applied after RoPE.

    Uses precomputed negated RoPE frequencies and functional RMSNorm
    (no learnable parameters) on queries and keys after rope application.
    This is unique -- most architectures apply QK norm before RoPE.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.
        hidden_size: Model hidden dimensionality.
        rms_norm_eps: Epsilon for functional RMSNorm.

    Example:
        >>> config = NanochatConfig()
        >>> attn = NanochatAttention(config)
        >>> out = attn(mx.zeros((1, 128, 1280)))
    """

    def __init__(self, config: NanochatConfig):
        """Initialize Nanochat attention.

        Args:
            config (NanochatConfig): Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.c_q = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        half_d = self.head_dim // 2
        self._rope_freqs = -mx.exp(mx.arange(0.0, half_d, dtype=mx.float32) * (math.log(config.rope_theta) / half_d))

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
        """Compute attention with RoPE followed by Q/K RMSNorm.

        Applies RoPE manually, then functional RMSNorm on Q/K, before
        passing to the attention performer without built-in rope.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)`` or
                ``(num_tokens, D)`` for paged layout.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output tensor matching input leading dimensions.
        """
        lead = hidden_states.shape[:-1]
        q = self.c_q(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.c_k(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.c_v(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if q.ndim == 3:
            offset = cache_view.offset if cache_view is not None else 0
            q = q[:, None, :, :]
            k = k[:, None, :, :]
            q = _apply_rotary_emb(q.transpose(0, 2, 1, 3), offset, self._rope_freqs).transpose(0, 2, 1, 3)[:, 0]
            k = _apply_rotary_emb(k.transpose(0, 2, 1, 3), offset, self._rope_freqs).transpose(0, 2, 1, 3)[:, 0]
            q = _functional_rms_norm(q, self.rms_norm_eps)
            k = _functional_rms_norm(k, self.rms_norm_eps)
        else:
            offset = cache_view.offset if cache_view is not None else 0

            qt = q.transpose(0, 2, 1, 3)
            kt = k.transpose(0, 2, 1, 3)
            qt = _apply_rotary_emb(qt, offset, self._rope_freqs)
            kt = _apply_rotary_emb(kt, offset, self._rope_freqs)

            qt = _functional_rms_norm(qt, self.rms_norm_eps)
            kt = _functional_rms_norm(kt, self.rms_norm_eps)

            q = qt.transpose(0, 2, 1, 3)
            k = kt.transpose(0, 2, 1, 3)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=None,
        )
        return self.c_proj(attn.reshape(*lead, -1))


class NanochatMLP(nn.Module):
    """ReLU-squared MLP for Nanochat.

    Computes ``c_proj(relu(c_fc(x))^2)`` -- uses squared ReLU activation
    instead of SwiGLU or GELU, which provides a sparser activation pattern.

    Attributes:
        c_fc: Up-projection linear layer.
        c_proj: Down-projection linear layer.

    Example:
        >>> config = NanochatConfig()
        >>> mlp = NanochatMLP(config)
        >>> out = mlp(mx.zeros((1, 128, 1280)))
    """

    def __init__(self, config: NanochatConfig):
        """Initialize Nanochat ReLU^2 MLP.

        Args:
            config (NanochatConfig): Model configuration.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute ReLU^2 MLP forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        x = self.c_fc(hidden_states)
        x = nn.relu(x) ** 2
        return self.c_proj(x)


class NanochatDecoderLayer(nn.Module):
    """Single Nanochat decoder layer with functional (parameter-free) RMSNorm.

    Uses functional RMSNorm (no learnable weight/bias) for pre-norm before
    both attention and MLP sub-layers.

    Attributes:
        attn: Nanochat attention module.
        mlp: ReLU^2 MLP module.
        rms_norm_eps: Epsilon for functional RMSNorm.

    Example:
        >>> config = NanochatConfig()
        >>> layer = NanochatDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 1280)))
    """

    def __init__(self, config: NanochatConfig):
        """Initialize Nanochat decoder layer.

        Args:
            config (NanochatConfig): Model configuration.
        """
        super().__init__()
        self.attn = NanochatAttention(config)
        self.mlp = NanochatMLP(config)
        self.rms_norm_eps = config.rms_norm_eps

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

        hidden_states = hidden_states + self.attn(
            _functional_rms_norm(hidden_states, self.rms_norm_eps),
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = hidden_states + self.mlp(_functional_rms_norm(hidden_states, self.rms_norm_eps))
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=NanochatConfig, model_type="nanochat")
class NanochatModel(EasyMLXBaseModule):
    """Base Nanochat transformer model with functional RMSNorm.

    Applies functional RMSNorm after token embedding and before the final
    output. Uses ``wte`` (weight-tied embedding) naming convention and ``h``
    for decoder layers.

    Attributes:
        config_class: Associated configuration class (``NanochatConfig``).
        wte: Token embedding layer.
        h: List of Nanochat decoder layers.
        rms_norm_eps: Epsilon for functional RMSNorm.

    Example:
        >>> config = NanochatConfig()
        >>> model = NanochatModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = NanochatConfig

    def __init__(self, config: NanochatConfig):
        """Initialize Nanochat base model.

        Args:
            config (NanochatConfig): Model configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = [NanochatDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.rms_norm_eps = config.rms_norm_eps

    @property
    def layers(self):
        """Return decoder layers (alias for ``self.h``).

        Returns:
            list[NanochatDecoderLayer]: The decoder layer list.
        """
        return self.h

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the Nanochat base model.

        Applies functional RMSNorm after embedding and before output.

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
        if cache_views is not None and len(cache_views) != len(self.h):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.wte(input_ids)

        hidden_states = _functional_rms_norm(hidden_states, self.rms_norm_eps)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.h):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        hidden_states = _functional_rms_norm(hidden_states, self.rms_norm_eps)
        return hidden_states


@register_module(task_type=TaskType.CAUSAL_LM, config=NanochatConfig, model_type="nanochat")
class NanochatForCausalLM(BaseCausalLMModule[NanochatModel, NanochatConfig]):
    """Nanochat causal language model with logit soft-capping.

    Applies ``cap * tanh(logits / cap)`` to output logits using
    ``config.logits_soft_cap``. Embeddings are tied by default.

    Attributes:
        config_class: Associated configuration class (``NanochatConfig``).

    Example:
        >>> config = NanochatConfig()
        >>> model = NanochatForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = NanochatConfig

    def __init__(self, config: NanochatConfig):
        """Initialize Nanochat causal LM.

        Args:
            config (NanochatConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=NanochatModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
            logit_cap=config.logits_soft_cap,
        )

    def get_embedding(self) -> nn.Embedding:
        """Return the token embedding layer for weight tying.

        Returns:
            nn.Embedding: The ``wte`` embedding from the base model.
        """
        return self.base_model.wte


__all__ = ("NanochatForCausalLM", "NanochatModel")
