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

"""MiniCPM3 MLX implementation (serving/inference only).

Structure:
  MiniCPM3Config -> MiniCPM3Attention (MLA) -> MiniCPM3MLP
  -> MiniCPM3DecoderLayer -> MiniCPM3Model -> MiniCPM3ForCausalLM

Key features:
  - Multi-head Latent Attention (MLA) with compressed KV projections
  - SuScaledRoPE for extended context
  - Depth scaling (scale_depth / sqrt(num_hidden_layers))
  - Embedding scaling (scale_emb)
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
from easymlx.layers.rotary import SuScaledRoPE
from easymlx.modules._base import BaseCausalLMModule

from .minicpm3_configuration import MiniCPM3Config

CacheView = TransformerCacheView | PageCacheView


class MiniCPM3Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for MiniCPM3.

    Uses LoRA-style compressed KV projections similar to DeepSeek V2:
      - Query: x -> q_a_proj -> q_a_layernorm -> q_b_proj -> split(nope, pe)
      - KV: x -> kv_a_proj_with_mqa -> split(compressed_kv, k_pe)
             compressed_kv -> kv_a_layernorm -> kv_b_proj -> split(k_nope, values)

    The compressed projections reduce KV cache memory by using a low-rank
    bottleneck (``kv_lora_rank``) instead of storing full key/value states.

    Attributes:
        num_heads: Number of attention heads.
        qk_rope_head_dim: RoPE portion of query/key head dimension.
        qk_nope_head_dim: Non-RoPE portion of query/key head dimension.
        kv_lora_rank: Rank for KV LoRA compression.
        q_lora_rank: Rank for query LoRA compression.
        hidden_size: Model hidden dimensionality.
        v_head_dim: Value head dimensionality (hidden_size // num_heads).
        q_head_dim: Total query head dim (qk_nope_head_dim + qk_rope_head_dim).
        scale: Attention scaling factor (1 / sqrt(q_head_dim)).

    Example:
        >>> config = MiniCPM3Config(hidden_size=2560, num_attention_heads=40)
        >>> attn = MiniCPM3Attention(config)
        >>> output = attn(mx.zeros((1, 128, 2560)))  # (B, L, D)
    """

    def __init__(self, config: MiniCPM3Config):
        """Initialize MiniCPM3 MLA attention.

        Args:
            config (MiniCPM3Config): Model configuration with MLA parameters
                including ``q_lora_rank``, ``kv_lora_rank``, ``qk_nope_head_dim``,
                and ``qk_rope_head_dim``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.q_lora_rank = int(config.q_lora_rank)
        self.hidden_size = int(config.hidden_size)

        self.v_head_dim = self.hidden_size // self.num_heads
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.q_head_dim ** (-0.5)

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        rope_scaling = config.rope_scaling or {}
        self.rope = SuScaledRoPE(
            dims=config.qk_rope_head_dim,
            base=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
            short_factor=rope_scaling.get("short_factor", 1.0),
            long_factor=rope_scaling.get("long_factor", 1.0),
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
        """Compute MLA attention forward pass.

        Compresses queries through LoRA bottleneck, compresses KV through
        shared MQA projection, applies SuScaledRoPE, then runs attention.

        Args:
            hidden_states (mx.array): Input tensor of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view for autoregressive
                decoding.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output tensor of shape ``(B, L, D)``.
        """
        B, L, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        offset = cache_view.offset if cache_view is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)

        k_pe = mx.broadcast_to(k_pe, (B, self.num_heads, L, self.qk_rope_head_dim))

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)

        if cache_view is not None:
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)

        output = self.attention_performer.forward(
            queries,
            keys,
            values,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniCPM3MLP(nn.Module):
    """SwiGLU feed-forward network for MiniCPM3.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gating linear projection.
        up_proj: Up-projection linear layer.
        down_proj: Down-projection linear layer.

    Example:
        >>> config = MiniCPM3Config(hidden_size=2560, intermediate_size=6400)
        >>> mlp = MiniCPM3MLP(config)
        >>> out = mlp(mx.zeros((1, 128, 2560)))  # (B, L, D)
    """

    def __init__(self, config: MiniCPM3Config):
        """Initialize MiniCPM3 SwiGLU MLP.

        Args:
            config (MiniCPM3Config): Model configuration with ``hidden_size``
                and ``intermediate_size``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states (mx.array): Input tensor of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiniCPM3DecoderLayer(nn.Module):
    """Single MiniCPM3 decoder layer with depth scaling.

    Applies pre-norm (RMSNorm) before attention and MLP, with residual
    connections scaled by ``scale_depth / sqrt(num_hidden_layers)``.

    Attributes:
        self_attn: MLA attention module.
        mlp: SwiGLU MLP module.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.
        scale_depth: Depth scaling factor from config.
        num_hidden_layers: Total number of layers (for scaling computation).

    Example:
        >>> config = MiniCPM3Config()
        >>> layer = MiniCPM3DecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 2560)))
    """

    def __init__(self, config: MiniCPM3Config):
        """Initialize MiniCPM3 decoder layer.

        Args:
            config (MiniCPM3Config): Model configuration.
        """
        super().__init__()
        self.self_attn = MiniCPM3Attention(config)
        self.mlp = MiniCPM3MLP(config)
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
        """Forward pass through a single decoder layer.

        Applies attention with depth-scaled residual, then MLP with
        depth-scaled residual. Scale factor is
        ``scale_depth / sqrt(num_hidden_layers)``.

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
        attn_out = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + attn_out * (self.scale_depth / (self.num_hidden_layers**0.5))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states) * (self.scale_depth / (self.num_hidden_layers**0.5))
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=MiniCPM3Config, model_type="minicpm3")
class MiniCPM3Model(EasyMLXBaseModule):
    """Base MiniCPM3 transformer model with MLA and embedding scaling.

    Token embeddings are scaled by ``config.scale_emb`` before being
    passed through the decoder layers. Uses Multi-head Latent Attention
    (MLA) with LoRA-compressed KV projections.

    Attributes:
        config_class: Associated configuration class (``MiniCPM3Config``).
        embed_tokens: Token embedding layer.
        layers: List of MiniCPM3 decoder layers.
        norm: Final RMSNorm.

    Example:
        >>> config = MiniCPM3Config()
        >>> model = MiniCPM3Model(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))  # (1, 3, hidden_size)
    """

    config_class = MiniCPM3Config

    def __init__(self, config: MiniCPM3Config):
        """Initialize MiniCPM3 base model.

        Args:
            config (MiniCPM3Config): Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiniCPM3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the MiniCPM3 base model.

        Embeds tokens (with ``scale_emb`` scaling), builds the attention mask,
        runs all decoder layers, and applies final normalization.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings,
                bypasses token embedding lookup if provided.
            cache_views (list[CacheView] | None): Per-layer KV cache views
                for autoregressive generation.
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
            hidden_states = self.embed_tokens(input_ids) * self.config.scale_emb

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


@register_module(task_type=TaskType.CAUSAL_LM, config=MiniCPM3Config, model_type="minicpm3")
class MiniCPM3ForCausalLM(BaseCausalLMModule[MiniCPM3Model, MiniCPM3Config]):
    """MiniCPM3 causal language model.

    When not tying embeddings, applies ``hidden_size / dim_model_base`` scaling
    before the LM head, matching the upstream MiniCPM3 convention.

    Attributes:
        config_class: Associated configuration class (``MiniCPM3Config``).

    Example:
        >>> config = MiniCPM3Config()
        >>> model = MiniCPM3ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))  # (1, 3, vocab_size)
    """

    config_class = MiniCPM3Config

    def __init__(self, config: MiniCPM3Config):
        """Initialize MiniCPM3 causal LM.

        Args:
            config (MiniCPM3Config): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MiniCPM3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute language model logits with optional scaling.

        When embeddings are not tied, scales hidden states by
        ``hidden_size / dim_model_base`` before the LM head projection.

        Args:
            hidden_states (mx.array): Hidden states of shape ``(B, L, D)``.

        Returns:
            mx.array: Logits of shape ``(B, L, vocab_size)``.
        """
        if not self._tie_word_embeddings:
            hidden_states = hidden_states / (self.config.hidden_size / self.config.dim_model_base)
        return super().compute_lm_logits(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream checkpoint weights.

        If ``lm_head.weight`` is missing and embeddings are not tied,
        copies ``model.embed_tokens.weight`` to ``lm_head.weight``.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Sanitized weight dictionary.
        """
        weights = super().sanitize(weights)
        if "lm_head.weight" not in weights and not self._tie_word_embeddings:
            embed_key = "model.embed_tokens.weight"
            if embed_key in weights:
                weights["lm_head.weight"] = weights[embed_key]
        return weights


__all__ = ("MiniCPM3ForCausalLM", "MiniCPM3Model")
