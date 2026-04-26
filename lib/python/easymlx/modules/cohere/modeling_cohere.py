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

"""Cohere MLX model implementation for serving and inference."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .cohere_configuration import CohereConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class CohereLayerNorm2D(nn.Module):
    """Per-head LayerNorm for Q/K normalization in Cohere models.

    Applies LayerNorm independently per head and rescales with a
    learned 2D weight of shape ``(num_heads, head_dim)``.

    Attributes:
        weight: Learnable weight of shape ``(d1, d2)``.
        eps: LayerNorm epsilon.

    Example::

        >>> norm = CohereLayerNorm2D(32, 128, eps=1e-5)
    """

    def __init__(self, d1: int, d2: int, eps: float):
        """Initialize the 2D LayerNorm.

        Args:
            d1: First dimension (number of heads).
            d2: Second dimension (head dim).
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.weight = mx.zeros((d1, d2))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply per-head LayerNorm.

        Args:
            x: Input tensor of shape ``(..., num_heads, head_dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)


class CohereMLP(nn.Module):
    """SwiGLU feed-forward network for the Cohere model.

    Uses ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: CohereConfig):
        """Initialize the Cohere MLP.

        Args:
            config: Model configuration with ``hidden_size`` and
                ``intermediate_size``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class CohereAttention(nn.Module):
    """Multi-head attention with traditional RoPE and optional per-head Q/K LayerNorm.

    Uses traditional (not interleaved) RoPE layout and optional
    ``CohereLayerNorm2D`` for Q/K normalization.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        use_qk_norm: Whether per-head QK LayerNorm is applied.
        q_proj: Query linear projection.
        k_proj: Key linear projection.
        v_proj: Value linear projection.
        o_proj: Output linear projection.
        q_norm: Optional per-head LayerNorm for queries.
        k_norm: Optional per-head LayerNorm for keys.
        rope: Traditional rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = CohereConfig(hidden_size=4096, num_attention_heads=32)
        >>> attn = CohereAttention(config)
    """

    def __init__(self, config: CohereConfig):
        """Initialize Cohere attention.

        Args:
            config: Model configuration with attention hyperparameters.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = CohereLayerNorm2D(self.num_heads, self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = CohereLayerNorm2D(self.num_kv_heads, self.head_dim, eps=config.layer_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
            scaling_config=None,
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
        """Run attention forward pass with traditional RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
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


class CohereDecoderLayer(nn.Module):
    """Single Cohere decoder layer with parallel residual connections.

    Uses a single LayerNorm before both attention and MLP. The
    attention and MLP outputs are added together with the residual
    in a single step: ``output = attn(norm(x)) + mlp(norm(x)) + x``.

    Attributes:
        self_attn: Cohere attention module.
        mlp: SwiGLU MLP.
        input_layernorm: LayerNorm applied before both attention and MLP.

    Example::

        >>> config = CohereConfig(hidden_size=4096)
        >>> layer = CohereDecoderLayer(config)
    """

    def __init__(self, config: CohereConfig):
        """Initialize the Cohere decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = CohereAttention(config)
        self.mlp = CohereMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the parallel-residual decoder layer forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        h = self.input_layernorm(hidden_states)
        attn_h = self.self_attn(
            h,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        ff_h = self.mlp(h)
        return attn_h + ff_h + hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=CohereConfig, model_type="cohere")
class CohereModel(EasyMLXBaseModule):
    """Base Cohere (Command R) transformer model for inference.

    Implements a decoder-only transformer with parallel residual
    connections, traditional RoPE, optional per-head QK LayerNorm,
    and SwiGLU MLP.

    Attributes:
        config_class: The configuration class (``CohereConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``CohereDecoderLayer`` decoder blocks.
        norm: Final LayerNorm normalization.

    Example::

        >>> config = CohereConfig(vocab_size=256000, hidden_size=4096)
        >>> model = CohereModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = CohereConfig

    def __init__(self, config: CohereConfig):
        """Initialize the Cohere base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [CohereDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias)

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
            input_ids: Integer token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

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
        """Remove rotary embedding buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=CohereConfig, model_type="cohere")
class CohereForCausalLM(BaseCausalLMModule[CohereModel, CohereConfig]):
    """Cohere model with a causal language modeling head.

    Wraps ``CohereModel`` with a linear projection to vocabulary logits.
    Applies ``logit_scale`` to the output logits. Supports tied word
    embeddings.

    Attributes:
        config_class: The configuration class (``CohereConfig``).

    Example::

        >>> config = CohereConfig(vocab_size=256000, hidden_size=4096)
        >>> model = CohereForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = CohereConfig

    def __init__(self, config: CohereConfig):
        """Initialize the Cohere causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=CohereModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits with Cohere's logit scaling.

        Applies the base logit computation, then multiplies by
        ``config.logit_scale``.

        Args:
            hidden_states: Final hidden states of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Scaled logits of shape ``(batch, seq_len, vocab_size)``.
        """
        logits = super().compute_lm_logits(hidden_states)
        logits = logits * self.config.logit_scale
        return logits


__all__ = ("CohereForCausalLM", "CohereModel")
