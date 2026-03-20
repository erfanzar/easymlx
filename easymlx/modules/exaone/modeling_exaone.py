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

"""Exaone MLX model implementation for serving and inference.

Llama-like architecture with RoPE, RMSNorm, and SwiGLU MLP.
Upstream uses field name ``layer_norm_epsilon`` despite using RMSNorm.
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

from .exaone_configuration import ExaoneConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an optional array-like to an ``mx.array`` of int32.

    Args:
        values: Array-like values, or ``None``.

    Returns:
        An ``mx.array`` with dtype int32, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class ExaoneMLP(nn.Module):
    """SwiGLU MLP for the Exaone model.

    Follows upstream naming: ``c_fc_0`` (gate), ``c_fc_1`` (up), ``c_proj`` (down).

    Example::

        >>> mlp = ExaoneMLP(config)
        >>> output = mlp(hidden_states)
    """

    def __init__(self, config: ExaoneConfig):
        """Initialize the MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.c_fc_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.c_fc_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run the SwiGLU MLP forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.c_proj(nn.silu(self.c_fc_0(hidden_states)) * self.c_fc_1(hidden_states))


class ExaoneAttention(nn.Module):
    """Multi-head attention with RoPE for the Exaone model.

    Follows upstream naming with ``out_proj`` instead of ``o_proj``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
    """

    def __init__(self, config: ExaoneConfig):
        """Initialize Exaone attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

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
        """Run the attention forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.out_proj(attn.reshape(*lead, -1))


class ExaoneDecoderLayer(nn.Module):
    """Single Exaone decoder layer.

    Follows upstream naming: ``ln_1``, ``attn``, ``ln_2``, ``mlp``.
    Uses pre-norm residual architecture.
    """

    def __init__(self, config: ExaoneConfig):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ExaoneAttention(config)
        self.ln_2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = ExaoneMLP(config)

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
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=ExaoneConfig, model_type="exaone")
class ExaoneModel(EasyMLXBaseModule):
    """Base Exaone transformer model.

    Follows upstream naming: ``wte`` (embeddings), ``h`` (layers), ``ln_f``
    (final norm). Properties ``layers`` and ``embed_tokens`` provide standard
    EasyMLX interface aliases.

    Attributes:
        config_class: The configuration class (``ExaoneConfig``).
        wte: Token embedding (upstream naming).
        h: List of decoder layers (upstream naming).
        ln_f: Final RMS normalization.

    Example::

        >>> config = ExaoneConfig()
        >>> model = ExaoneModel(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = ExaoneConfig

    def __init__(self, config: ExaoneConfig):
        """Initialize the Exaone base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = [ExaoneDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    @property
    def layers(self):
        """Return the decoder layers (alias for ``self.h``)."""
        return self.h

    @property
    def embed_tokens(self):
        """Return the embedding layer (alias for ``self.wte``)."""
        return self.wte

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
            attention_mask: Optional mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings instead of ``input_ids``.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length mismatches layer count.
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

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.h):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.ln_f(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Filter out rotary embedding inverse frequency keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with inv_freq keys removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=ExaoneConfig, model_type="exaone")
class ExaoneForCausalLM(BaseCausalLMModule[ExaoneModel, ExaoneConfig]):
    """Exaone causal language model.

    Uses ``transformer`` as the base model name for upstream weight
    compatibility.

    Example::

        >>> model = ExaoneForCausalLM(ExaoneConfig())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = ExaoneConfig

    def __init__(self, config: ExaoneConfig):
        """Initialize the Exaone causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=ExaoneModel,
            base_model_name="transformer",
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("ExaoneForCausalLM", "ExaoneModel")
