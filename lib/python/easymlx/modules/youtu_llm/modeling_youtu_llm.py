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

"""YouTu LLM MLX model implementation for serving and inference.

YouTu LLM uses Multi-head Latent Attention (MLA) with optional
q_lora_rank compression, kv_b_proj for key/value projection from
compressed KV latent, and a standard SwiGLU MLP.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .youtu_llm_configuration import YouTuLLMConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an ``mx.array`` of dtype ``int32``.

    Args:
        values: Input values. Accepts ``mx.array``, sequences, or ``None``.

    Returns:
        An ``mx.array`` with ``int32`` dtype, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class YouTuLLMAttention(nn.Module):
    """Multi-head Latent Attention (MLA) for YouTu LLM.

    Implements MLA where key/value projections pass through a low-rank
    latent bottleneck (``kv_lora_rank``). The compressed latent is
    normalized with RMSNorm before being expanded via ``kv_b_proj`` into
    full-rank K and V. Query projections optionally use a similar
    two-stage LoRA compression (``q_lora_rank``). Each query/key head
    is split into a non-RoPE portion (``qk_nope_head_dim``) and a RoPE
    portion (``qk_rope_head_dim``), which are concatenated after
    independent processing.

    Attributes:
        config: Model configuration.
        hidden_size: Dimensionality of input hidden states.
        num_heads: Number of query attention heads.
        q_lora_rank: Query LoRA rank (``None`` for direct projection).
        qk_rope_head_dim: Dimensions per head with RoPE.
        kv_lora_rank: KV latent compression rank.
        v_head_dim: Value head dimensionality.
        qk_nope_head_dim: Dimensions per head without RoPE.
        q_head_dim: Total query head dim (``qk_nope_head_dim + qk_rope_head_dim``).
        scale: Attention scaling factor.
        q_proj: Direct query projection (when ``q_lora_rank`` is ``None``).
        q_a_proj: First-stage query LoRA projection (when ``q_lora_rank`` is set).
        q_a_layernorm: RMSNorm after first-stage query compression.
        q_b_proj: Second-stage query LoRA projection.
        kv_a_proj_with_mqa: Combined KV latent + RoPE key projection.
        kv_a_layernorm: RMSNorm for KV latent normalization.
        kv_b_proj: Expansion from KV latent to full K_nope and V.
        o_proj: Output projection.
        rope: RoPE module operating on ``qk_rope_head_dim`` dimensions.

    Example:
        >>> config = YouTuLLMConfig(kv_lora_rank=512, q_lora_rank=1536)
        >>> attn = YouTuLLMAttention(config)
        >>> out = attn(mx.zeros((1, 10, 2048)))
    """

    def __init__(self, config: YouTuLLMConfig):
        """Initialize MLA attention.

        Args:
            config: Model configuration specifying MLA dimensions,
                LoRA ranks, RoPE parameters, and bias settings.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=config.attention_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
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

        self.rope = get_rope(
            dims=self.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute Multi-head Latent Attention.

        The forward pass:
        1. Optionally compress queries via LoRA (q_a_proj -> norm -> q_b_proj).
        2. Split queries into nope and RoPE portions.
        3. Compress KV via kv_a_proj_with_mqa, split into latent + k_pe.
        4. Expand latent via kv_b_proj into k_nope and values.
        5. Apply RoPE to q_pe and k_pe, repeat k_pe across heads.
        6. Concatenate nope/pe portions and compute scaled dot-product attention.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask (boolean array or ``None``).
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _D = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        offset = cache_view.offset if cache_view is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)

        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        if cache_view is not None:
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)

        scores = (queries * self.scale) @ keys.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class YouTuLLMMLP(nn.Module):
    """SwiGLU feed-forward network for YouTu LLM.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear projection for the SiLU gate.
        up_proj: Linear projection for the element-wise product branch.
        down_proj: Linear projection back to ``hidden_size``.

    Example:
        >>> config = YouTuLLMConfig(hidden_size=2048, intermediate_size=6144)
        >>> mlp = YouTuLLMMLP(config)
    """

    def __init__(self, config: YouTuLLMConfig):
        """Initialize the SwiGLU MLP.

        Args:
            config: Model configuration specifying hidden/intermediate
                sizes and bias settings.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class YouTuLLMDecoderLayer(nn.Module):
    """Single YouTu LLM decoder layer with MLA and SwiGLU MLP.

    Applies pre-norm MLA attention followed by pre-norm SwiGLU MLP,
    each with a residual connection.

    Attributes:
        self_attn: Multi-head Latent Attention sub-layer.
        mlp: SwiGLU feed-forward sub-layer.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before the MLP.

    Example:
        >>> config = YouTuLLMConfig(hidden_size=2048)
        >>> layer = YouTuLLMDecoderLayer(config)
    """

    def __init__(self, config: YouTuLLMConfig):
        """Initialize a YouTu LLM decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = YouTuLLMAttention(config)
        self.mlp = YouTuLLMMLP(config)
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
        """Run the decoder layer forward pass.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.

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


@register_module(task_type=TaskType.BASE_MODULE, config=YouTuLLMConfig, model_type="youtu_llm")
class YouTuLLMModel(EasyMLXBaseModule):
    """Base YouTu LLM transformer model with Multi-head Latent Attention.

    A decoder-only transformer using MLA for compressed KV projections,
    RMSNorm, SwiGLU MLP, and RoPE. Supports both batched and paged
    attention modes.

    Attributes:
        config_class: Associated configuration class (``YouTuLLMConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``YouTuLLMDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = YouTuLLMConfig(hidden_size=2048, num_hidden_layers=4)
        >>> model = YouTuLLMModel(config)
    """

    config_class = YouTuLLMConfig

    def __init__(self, config: YouTuLLMConfig):
        """Initialize the base YouTu LLM model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [YouTuLLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the forward pass through all decoder layers.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of layers.
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
        """Remove non-persistent rotary embedding buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=YouTuLLMConfig, model_type="youtu_llm")
class YouTuLLMForCausalLM(BaseCausalLMModule[YouTuLLMModel, YouTuLLMConfig]):
    """YouTu LLM causal language model with an LM head.

    Wraps ``YouTuLLMModel`` with a linear language-model head for
    next-token prediction. Supports weight tying.

    Attributes:
        config_class: Associated configuration class (``YouTuLLMConfig``).

    Example:
        >>> config = YouTuLLMConfig(hidden_size=2048, num_hidden_layers=4)
        >>> model = YouTuLLMForCausalLM(config)
    """

    config_class = YouTuLLMConfig

    def __init__(self, config: YouTuLLMConfig):
        """Initialize the causal LM wrapper.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=YouTuLLMModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("YouTuLLMForCausalLM", "YouTuLLMModel")
