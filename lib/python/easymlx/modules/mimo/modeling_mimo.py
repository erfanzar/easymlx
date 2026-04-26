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

"""MiMo MLX model implementation for serving and inference.

MiMo is a standard Llama-style transformer with biased QKV projections
and multi-token prediction layers (stripped at load time).
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

from .mimo_configuration import MiMoConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like value to an int32 mx.array.

    Args:
        values: Input values to convert, or ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class MiMoAttention(nn.Module):
    """MiMo multi-head attention with biased QKV projections and RoPE.

    Unlike standard Llama attention, MiMo uses bias in Q, K, and V
    projections while the output projection remains bias-free.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        rope: Rotary position embedding.

    Example:
        >>> config = MiMoConfig(hidden_size=64, num_attention_heads=4)
        >>> attn = MiMoAttention(config)
    """

    def __init__(self, config: MiMoConfig):
        """Initialize MiMo attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

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
        """Compute attention with biased QKV projections and RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
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
        return self.o_proj(attn.reshape(*lead, -1))


class MiMoMLP(nn.Module):
    """MiMo SwiGLU feed-forward MLP.

    Standard SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.

    Example:
        >>> config = MiMoConfig(hidden_size=64, intermediate_size=128)
        >>> mlp = MiMoMLP(config)
    """

    def __init__(self, config: MiMoConfig):
        """Initialize MiMo MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiMoDecoderLayer(nn.Module):
    """Single MiMo decoder layer with pre-norm and residual connections.

    Attributes:
        self_attn: MiMo attention with biased QKV.
        mlp: SwiGLU MLP.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.

    Example:
        >>> config = MiMoConfig(hidden_size=64, num_attention_heads=4)
        >>> layer = MiMoDecoderLayer(config)
    """

    def __init__(self, config: MiMoConfig):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = MiMoAttention(config)
        self.mlp = MiMoMLP(config)
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
        """Run the decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=MiMoConfig, model_type="mimo")
class MiMoModel(EasyMLXBaseModule):
    """Base MiMo Llama-style transformer model with biased QKV.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        layers: List of ``MiMoDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = MiMoConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MiMoModel(config)
    """

    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        """Initialize the base MiMo model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiMoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the forward pass through the MiMo backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged attention metadata.

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
        """Sanitize upstream weights for MiMo.

        Removes rotary embedding inverse frequency keys, MTP layer weights,
        and tied LM head weights when applicable.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        return {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
            and not k.startswith("model.mtp_layers.")
            and not (self.config.tie_word_embeddings and k == "lm_head.weight")
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=MiMoConfig, model_type="mimo")
class MiMoForCausalLM(BaseCausalLMModule[MiMoModel, MiMoConfig]):
    """MiMo causal language model with an LM head.

    Wraps ``MiMoModel`` with a language modeling head for next-token
    prediction.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = MiMoConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MiMoForCausalLM(config)
    """

    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        """Initialize the MiMo causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MiMoModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.base_model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("MiMoForCausalLM", "MiMoModel")
