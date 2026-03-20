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

"""AFM7 MLX model implementation for serving and inference.

AFM7 (Apple Foundation Model 7) features standard transformer layers
followed by KV-reuse layers that share cached KV states from the last
standard transformer layer. Uses QK-norm and SwiGLU MLP.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .afm7_configuration import Afm7Config

CacheView = TransformerCacheView | PageCacheView


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


class Afm7Attention(nn.Module):
    """Multi-head grouped-query attention for AFM7 with QK-norm and RoPE.

    Applies separate Q, K, V projections with RMSNorm on Q and K before
    rotary position embedding. Uses grouped-query attention (GQA) when
    ``num_kv_heads < num_heads``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor (``head_dim ** -0.5``).
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
        out_proj: Linear output projection.
        q_norm: RMSNorm applied to query heads.
        k_norm: RMSNorm applied to key heads.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = Afm7Config(hidden_dim=2048, num_heads=16)
        >>> attn = Afm7Attention(config)
        >>> out = attn(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: Afm7Config):
        """Initialize AFM7 attention.

        Args:
            config: Model configuration containing ``hidden_dim``,
                ``num_heads``, ``num_kv_heads``, ``head_dim``, and
                ``rope_theta``.
        """
        super().__init__()
        dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
        )
        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the attention forward pass with QK-norm and RoPE.

        Projects input to Q, K, V, applies RMSNorm to Q and K, then
        computes scaled dot-product attention with rotary embeddings.

        Args:
            x: Input hidden states of shape ``(batch, seq_len, hidden_dim)``
                or ``(seq_len, hidden_dim)`` for paged attention.
            mask: Optional attention mask. Can be an ``mx.array`` or a
                string sentinel for special mask behavior.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        lead = x.shape[:-1]

        q = self.q_proj(x).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)

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
        return self.out_proj(attn.reshape(*lead, -1))


class Afm7MLP(nn.Module):
    """SwiGLU feed-forward network for AFM7.

    Uses SiLU-gated linear unit: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear gate projection.
        down_proj: Linear down projection.
        up_proj: Linear up projection.

    Example::

        >>> config = Afm7Config(hidden_dim=2048)
        >>> mlp = Afm7MLP(config)
        >>> out = mlp(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: Afm7Config):
        """Initialize the AFM7 SwiGLU MLP.

        Args:
            config: Model configuration containing ``hidden_dim`` and
                ``hidden_dim_scale_factor`` to compute intermediate size.
        """
        super().__init__()
        dim = config.hidden_dim
        hidden_dim = int(dim * config.hidden_dim_scale_factor)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SwiGLU MLP to the input.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.

        Returns:
            Output tensor of the same shape as input.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Afm7TransformerBlock(nn.Module):
    """Standard transformer decoder block for AFM7.

    Applies pre-norm attention followed by pre-norm SwiGLU MLP with
    residual connections.

    Attributes:
        self_attn: Multi-head attention with QK-norm.
        mlp: SwiGLU feed-forward network.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example::

        >>> config = Afm7Config(hidden_dim=2048, num_heads=16)
        >>> block = Afm7TransformerBlock(config)
        >>> out = block(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: Afm7Config):
        """Initialize the AFM7 transformer block.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__()
        self.self_attn = Afm7Attention(config)
        self.mlp = Afm7MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the decoder block forward pass.

        Applies pre-norm attention with residual, then pre-norm MLP
        with residual.

        Args:
            x: Input hidden states of shape ``(batch, seq_len, hidden_dim)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output hidden states of the same shape as input.
        """
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache_view=cache_view, cache_metadata=cache_metadata)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


@register_module(task_type=TaskType.BASE_MODULE, config=Afm7Config, model_type="afm7")
class Afm7Model(EasyMLXBaseModule):
    """Base AFM7 transformer model with KV-reuse layers.

    Implements a decoder-only transformer with standard layers followed
    by KV-reuse layers. Uses RMSNorm normalization, QK-norm attention,
    SwiGLU MLP, and rotary position embeddings.

    Attributes:
        config_class: The configuration class (``Afm7Config``).
        embed_tokens: Token embedding layer mapping vocab IDs to vectors.
        layers: List of ``Afm7TransformerBlock`` decoder blocks. Total
            count is ``num_layers + num_kv_reuse_layers``.
        norm: Final RMS normalization applied to the last hidden state.

    Example::

        >>> config = Afm7Config(vocab_size=32000, hidden_dim=2048)
        >>> model = Afm7Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        (1, 3, 2048)
    """

    config_class = Afm7Config

    def __init__(self, config: Afm7Config):
        """Initialize the AFM7 base model.

        Args:
            config: Model configuration containing architecture
                hyperparameters such as ``hidden_dim``, ``num_layers``,
                ``num_kv_reuse_layers``, and ``num_heads``.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        total_layers = config.num_layers + config.num_kv_reuse_layers
        self.layers = [Afm7TransformerBlock(config) for _ in range(total_layers)]
        self.norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

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

        Embeds input tokens, applies all decoder layers with optional
        KV caching, and returns normalized hidden states.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``
                or ``(seq_len,)`` (auto-batched when no paged metadata).
            attention_mask: Optional boolean/int mask of shape
                ``(batch, seq_len)`` where 1 indicates attending positions.
            input_embeddings: Pre-computed embeddings to use instead of
                ``input_ids``. Shape ``(batch, seq_len, hidden_dim)``.
            cache_views: Per-layer KV cache views for autoregressive
                generation. Length must match ``num_hidden_layers``.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Hidden states tensor of shape ``(batch, seq_len, hidden_dim)``
            after final layer normalization.

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
        """Transform checkpoint weights for compatibility with this model.

        Removes precomputed rotary embedding buffers (``rotary_emb.inv_freq``
        and ``rope.inv_freq``) that are recomputed at initialization.

        Args:
            weights: Raw checkpoint weight dictionary mapping parameter
                names to ``mx.array`` tensors.

        Returns:
            Cleaned weight dictionary with incompatible keys removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Afm7Config, model_type="afm7")
class Afm7ForCausalLM(BaseCausalLMModule[Afm7Model, Afm7Config]):
    """AFM7 model with a causal language modeling head.

    Wraps ``Afm7Model`` with a linear projection to vocabulary logits.
    Supports tied word embeddings (sharing weights between the input
    embedding and the output projection).

    Attributes:
        config_class: The configuration class (``Afm7Config``).

    Example::

        >>> config = Afm7Config(vocab_size=32000, hidden_dim=2048)
        >>> model = Afm7ForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
        >>> output.logits.shape
        (1, 3, 32000)
    """

    config_class = Afm7Config

    def __init__(self, config: Afm7Config):
        """Initialize the AFM7 causal language model.

        Args:
            config: Model configuration. Uses ``tie_word_embeddings``
                to determine whether to share input/output weights.
        """
        super().__init__(
            config=config,
            base_model_class=Afm7Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transform checkpoint weights for compatibility.

        Applies base sanitization, then remaps upstream naming
        conventions: ``model.embedding`` to ``model.embed_tokens``
        and ``model.output_norm`` to ``model.norm``.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with keys remapped.
        """
        weights = super().sanitize(weights)
        # Map upstream model.embedding -> model.embed_tokens
        embed_key = "model.embedding.weight"
        target_key = "model.embed_tokens.weight"
        if embed_key in weights and target_key not in weights:
            weights[target_key] = weights.pop(embed_key)
        # Map upstream model.output_norm -> model.norm
        for suffix in ["weight", "bias"]:
            old = f"model.output_norm.{suffix}"
            new = f"model.norm.{suffix}"
            if old in weights and new not in weights:
                weights[new] = weights.pop(old)
        return weights


__all__ = ("Afm7ForCausalLM", "Afm7Model")
