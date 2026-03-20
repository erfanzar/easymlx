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

"""MiniCPM MLX implementation (serving/inference only).

Structure mirrors EasyDeL's minicpm:
  MiniCPMConfig -> MiniCPMAttention -> MiniCPMMLP -> MiniCPMDecoderLayer
  -> MiniCPMModel -> MiniCPMForCausalLM

MiniCPM uses depth and embedding scaling for improved small model training.
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

from .minicpm_configuration import MiniCPMConfig

CacheView = TransformerCacheView | PageCacheView


class MiniCPMAttention(nn.Module):
    """Multi-head attention for MiniCPM with RoPE.

    Standard multi-head attention with optional GQA, configurable head
    dimension, and RoPE positional encoding.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        rope: Rotary position embedding.

    Example:
        >>> config = MiniCPMConfig(hidden_size=64, num_attention_heads=4)
        >>> attn = MiniCPMAttention(config)
    """

    def __init__(self, config: MiniCPMConfig):
        """Initialize MiniCPM attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
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
        """Compute multi-head attention with RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

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


class MiniCPMMLP(nn.Module):
    """SwiGLU feed-forward network for MiniCPM.

    Standard SwiGLU: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.

    Example:
        >>> config = MiniCPMConfig(hidden_size=64, intermediate_size=128)
        >>> mlp = MiniCPMMLP(config)
    """

    def __init__(self, config: MiniCPMConfig):
        """Initialize MiniCPM MLP.

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


class MiniCPMDecoderLayer(nn.Module):
    """Single MiniCPM decoder layer with depth scaling.

    Applies a depth-dependent scaling factor of
    ``scale_depth / sqrt(num_hidden_layers)`` to both the attention
    and MLP residual connections, improving training stability for
    small models.

    Attributes:
        self_attn: MiniCPM attention sub-layer.
        mlp: SwiGLU MLP sub-layer.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.
        scale_depth: Depth scaling factor.
        num_hidden_layers: Total number of layers (used for scaling).

    Example:
        >>> config = MiniCPMConfig(hidden_size=64, num_attention_heads=4, scale_depth=1.4)
        >>> layer = MiniCPMDecoderLayer(config)
    """

    def __init__(self, config: MiniCPMConfig):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = MiniCPMAttention(config)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the decoder layer with depth-scaled residuals.

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
        attn_out = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + attn_out * (self.scale_depth / self.num_hidden_layers**0.5)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states) * (self.scale_depth / self.num_hidden_layers**0.5)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=MiniCPMConfig, model_type="minicpm")
class MiniCPMModel(EasyMLXBaseModule):
    """Base MiniCPM transformer model with embedding scaling.

    Multiplies token embeddings by ``scale_emb`` before passing through
    the decoder layers. Each layer applies depth-scaled residual connections.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        scale_emb: Embedding scaling factor.
        layers: List of ``MiniCPMDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = MiniCPMConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MiniCPMModel(config)
    """

    config_class = MiniCPMConfig

    def __init__(self, config: MiniCPMConfig):
        """Initialize the base MiniCPM model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.scale_emb = config.scale_emb
        self.layers = [MiniCPMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the forward pass with embedding scaling and depth-scaled layers.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``
                and skips embedding scaling).
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
            hidden_states = self.embed_tokens(input_ids) * self.scale_emb

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


@register_module(task_type=TaskType.CAUSAL_LM, config=MiniCPMConfig, model_type="minicpm")
class MiniCPMForCausalLM(BaseCausalLMModule[MiniCPMModel, MiniCPMConfig]):
    """MiniCPM causal language model.

    When not tying embeddings, applies hidden_size / dim_model_base scaling
    before the LM head, matching the upstream MiniCPM convention.
    """

    config_class = MiniCPMConfig

    def __init__(self, config: MiniCPMConfig):
        """Initialize the MiniCPM causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MiniCPMModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits with optional hidden_size/dim_model_base scaling.

        When embeddings are not tied, divides hidden states by
        ``hidden_size / dim_model_base`` before the LM head projection,
        matching the upstream MiniCPM convention.

        Args:
            hidden_states: Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        if not self._tie_word_embeddings:
            hidden_states = hidden_states / (self.config.hidden_size / self.config.dim_model_base)
        return super().compute_lm_logits(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights, creating lm_head from embed_tokens if needed.

        When ``tie_word_embeddings`` is ``False`` and no ``lm_head.weight``
        is present, copies the embedding weight to the LM head.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = super().sanitize(weights)
        if "lm_head.weight" not in weights and not self._tie_word_embeddings:
            embed_key = "model.embed_tokens.weight"
            if embed_key in weights:
                weights["lm_head.weight"] = weights[embed_key]
        return weights


__all__ = ("MiniCPMForCausalLM", "MiniCPMModel")
