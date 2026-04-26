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

"""Hunyuan V1 Dense MLX implementation (serving/inference only).

Structure:
  HunyuanV1DenseConfig -> HunyuanV1DenseDynamicNTKAlphaRoPE
  -> HunyuanV1DenseAttention -> HunyuanV1DenseMLP
  -> HunyuanV1DenseDecoderLayer -> HunyuanV1DenseModel
  -> HunyuanV1DenseForCausalLM

Key features:
  - Dynamic NTK-Alpha RoPE scaling
  - QK normalization (per-head)
  - Standard dense transformer (no MoE)
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
from easymlx.modules._base import BaseCausalLMModule

from .hunyuan_v1_dense_configuration import HunyuanV1DenseConfig

CacheView = TransformerCacheView | PageCacheView


class HunyuanV1DenseDynamicNTKAlphaRoPE(nn.Module):
    """Dynamic NTK-Alpha scaled rotary positional embedding for Hunyuan V1 Dense.

    Applies frequency-domain scaling: ``base' = base * alpha^(dims / (dims - 2))``.

    Attributes:
        dims: Number of rotary embedding dimensions.

    Example:
        >>> rope = HunyuanV1DenseDynamicNTKAlphaRoPE(dims=64, scaling_alpha=2.0)
        >>> x = mx.zeros((1, 8, 10, 64))
        >>> out = rope(x, offset=0)
        >>> out.shape
        [1, 8, 10, 64]
    """

    def __init__(self, dims: int, base: float = 10000, scaling_alpha: float = 1.0):
        """Initialize HunyuanV1DenseDynamicNTKAlphaRoPE.

        Args:
            dims: Number of rotary embedding dimensions (must be even).
            base: Base frequency for RoPE.
            scaling_alpha: NTK-Alpha scaling factor (1.0 = no scaling).
        """
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        """Apply rotary positional embedding.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, dims)``.
            offset: Positional offset for autoregressive decoding.

        Returns:
            Tensor with rotary embeddings applied, same shape as *x*.
        """
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class HunyuanV1DenseAttention(nn.Module):
    """Multi-head attention with optional QK normalization for Hunyuan V1 Dense.

    Uses GQA with Dynamic NTK-Alpha RoPE and optional per-head RMSNorm
    on queries and keys for training stability.

    Attributes:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        use_qk_norm: Whether QK normalization is applied.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Dynamic NTK-Alpha RoPE.
        attention_performer: Attention computation backend.

    Example:
        >>> config = HunyuanV1DenseConfig(hidden_size=256, num_attention_heads=8)
        >>> attn = HunyuanV1DenseAttention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HunyuanV1DenseConfig):
        """Initialize HunyuanV1DenseAttention.

        Args:
            config: Hunyuan V1 Dense model configuration.
        """
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim if config.head_dim is not None else (dim // self.n_heads)
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.attention_bias)

        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, config.rms_norm_eps)

        scaling_alpha = 1.0
        if config.rope_scaling and "alpha" in config.rope_scaling:
            scaling_alpha = config.rope_scaling["alpha"]

        self.rope = HunyuanV1DenseDynamicNTKAlphaRoPE(
            self.head_dim,
            base=config.rope_theta,
            scaling_alpha=scaling_alpha,
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
        """Compute attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
        """
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache_view.offset if cache_view is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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


class HunyuanV1DenseMLP(nn.Module):
    """SwiGLU feed-forward network for Hunyuan V1 Dense.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection (no bias).
        down_proj: Down projection (no bias).
        up_proj: Up projection (no bias).

    Example:
        >>> config = HunyuanV1DenseConfig(hidden_size=256, intermediate_size=512)
        >>> mlp = HunyuanV1DenseMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HunyuanV1DenseConfig):
        """Initialize HunyuanV1DenseMLP.

        Args:
            config: Hunyuan V1 Dense model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class HunyuanV1DenseDecoderLayer(nn.Module):
    """Single Hunyuan V1 Dense decoder layer.

    Applies pre-norm attention followed by pre-norm SwiGLU MLP with
    residual connections.

    Attributes:
        self_attn: Attention sub-layer.
        mlp: SwiGLU MLP sub-layer.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = HunyuanV1DenseConfig(hidden_size=256, num_attention_heads=8)
        >>> layer = HunyuanV1DenseDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HunyuanV1DenseConfig):
        """Initialize HunyuanV1DenseDecoderLayer.

        Args:
            config: Hunyuan V1 Dense model configuration.
        """
        super().__init__()
        self.self_attn = HunyuanV1DenseAttention(config)
        self.mlp = HunyuanV1DenseMLP(config)
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
        """Execute the decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=HunyuanV1DenseConfig, model_type="hunyuan_v1_dense")
class HunyuanV1DenseModel(EasyMLXBaseModule):
    """Base Hunyuan V1 Dense transformer model (no LM head).

    A dense (non-MoE) variant of Hunyuan using Dynamic NTK-Alpha RoPE
    and optional QK normalization with a standard SwiGLU MLP.

    Attributes:
        config_class: Associated configuration class.
        config: Model configuration.
        embed_tokens: Token embedding layer.
        layers: List of ``HunyuanV1DenseDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = HunyuanV1DenseConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = HunyuanV1DenseModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = HunyuanV1DenseConfig

    def __init__(self, config: HunyuanV1DenseConfig):
        """Initialize HunyuanV1DenseModel.

        Args:
            config: Hunyuan V1 Dense model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [HunyuanV1DenseDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the Hunyuan V1 Dense transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; if provided,
                ``input_ids`` is ignored.
            cache_views: Per-layer KV cache views for autoregressive decoding.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=HunyuanV1DenseConfig, model_type="hunyuan_v1_dense")
class HunyuanV1DenseForCausalLM(BaseCausalLMModule[HunyuanV1DenseModel, HunyuanV1DenseConfig]):
    """Hunyuan V1 Dense transformer with a causal language modeling head.

    Wraps ``HunyuanV1DenseModel`` and adds an LM head for next-token
    prediction.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = HunyuanV1DenseConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = HunyuanV1DenseForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = HunyuanV1DenseConfig

    def __init__(self, config: HunyuanV1DenseConfig):
        """Initialize HunyuanV1DenseForCausalLM.

        Args:
            config: Hunyuan V1 Dense model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=HunyuanV1DenseModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("HunyuanV1DenseForCausalLM", "HunyuanV1DenseModel")
