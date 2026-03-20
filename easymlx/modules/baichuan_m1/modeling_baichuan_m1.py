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

"""Baichuan M1 MLX model implementation for serving and inference.

Baichuan M1 uses:
- Packed QKV projection (W_pack)
- Custom 1-D convolution on K/V with a window of 2
- Sliding window attention on designated layers
- Standard SwiGLU MLP
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from .baichuan_m1_configuration import BaichuanM1Config

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


class BaichuanM1Attention(nn.Module):
    """Attention for Baichuan M1 with packed QKV and KV convolution.

    Uses a single fused ``W_pack`` projection for Q, K, V, followed by
    custom 1-D convolution on K and V with a window of 2. Supports
    sliding window attention on designated layers and GQA with
    different head counts for SWA layers.

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer in the stack.
        is_swa: Whether this layer uses sliding window attention.
        num_heads: Number of query attention heads (may differ for SWA layers).
        num_kv_heads: Number of KV heads (may differ for SWA layers).
        hidden_size: Model hidden dimensionality.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        W_pack: Fused QKV linear projection.
        o_proj: Output linear projection.
        rope: Rotary position embedding module.
        conv_window: Convolution window size (always 2).
        conv_k: Learnable 1-D convolution weights for keys.
        conv_v: Learnable 1-D convolution weights for values.

    Example::

        >>> config = BaichuanM1Config(hidden_size=2048)
        >>> attn = BaichuanM1Attention(config, layer_idx=0)
    """

    def __init__(self, config: BaichuanM1Config, layer_idx: int):
        """Initialize Baichuan M1 attention.

        Args:
            config: Model configuration with attention hyperparameters.
            layer_idx: Index of this layer, used to determine if sliding
                window attention is applied.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.is_swa = layer_idx in config.sliding_window_layers
        self.num_heads = (
            config.num_swa_attention_heads
            if self.is_swa and config.num_swa_attention_heads
            else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.num_swa_key_value_heads
            if self.is_swa and config.num_swa_key_value_heads
            else config.num_key_value_heads
        )

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.W_pack = nn.Linear(
            config.hidden_size,
            self.hidden_size + 2 * self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

        self.conv_window = config.conv_window
        self.conv_k = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))
        self.conv_v = mx.zeros((1, 1, self.num_kv_heads, 1, self.conv_window))

    def _custom_convolution(
        self,
        u: mx.array,
        weights: mx.array,
        state: mx.array | None = None,
    ) -> mx.array:
        """Apply custom 1-D convolution with window size 2.

        Convolves the input with learned weights: ``u_prev * w0 + u * w1``.

        Args:
            u: Input tensor of shape ``(B, H, L, D)``.
            weights: Convolution weights of shape
                ``(1, 1, num_kv_heads, 1, conv_window)``.
            state: Previous timestep state for causal convolution.
                If ``None``, uses zeros.

        Returns:
            Convolved tensor of the same shape as ``u``.
        """
        B, H, L, D = u.shape
        weights = weights.reshape((1, H, self.conv_window, 1, 1))
        w0 = weights[:, :, 0]
        w1 = weights[:, :, 1]
        if state is None:
            state = mx.zeros((B, H, 1, D), u.dtype)
        if L > 1:
            u_prev = mx.concatenate([state, u[:, :, :-1]], axis=2)
        else:
            u_prev = state
        return u_prev * w0 + u * w1

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the attention forward pass with packed QKV and KV convolution.

        Projects input through fused QKV, applies 1-D convolution on K/V,
        rotary embeddings, optional KV caching, GQA repeat, and scaled
        dot-product attention.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, D = hidden_states.shape

        proj = self.W_pack(hidden_states)
        q, k, v = mx.split(proj, (D, D + self.num_kv_heads * self.head_dim), axis=-1)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply custom convolution on K and V
        k = self._custom_convolution(k, self.conv_k)
        v = self._custom_convolution(v, self.conv_v)

        offset = cache_view.offset if cache_view is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache_view is not None:
            k, v, _ = cache_view.concatenate_to_cache(k, v)

        # Repeat KV heads for GQA
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        scores = (q * self.scale) @ k.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ v

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class BaichuanM1MLP(nn.Module):
    """SwiGLU feed-forward network for Baichuan M1.

    Uses ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: BaichuanM1Config):
        """Initialize the Baichuan M1 MLP.

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


class BaichuanM1DecoderLayer(nn.Module):
    """Single Baichuan M1 decoder layer with pre-norm residual connections.

    Attributes:
        self_attn: Packed QKV attention with KV convolution.
        mlp: SwiGLU feed-forward network.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
    """

    def __init__(self, config: BaichuanM1Config, layer_idx: int):
        """Initialize the Baichuan M1 decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack.
        """
        super().__init__()
        self.self_attn = BaichuanM1Attention(config, layer_idx)
        self.mlp = BaichuanM1MLP(config)
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
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output hidden states of the same shape.
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


@register_module(task_type=TaskType.BASE_MODULE, config=BaichuanM1Config, model_type="baichuan_m1")
class BaichuanM1Model(EasyMLXBaseModule):
    """Base Baichuan M1 transformer model for inference.

    Implements a decoder-only transformer with packed QKV projections,
    1-D KV convolution, sliding window attention on designated layers,
    and SwiGLU MLP.

    Attributes:
        config_class: The configuration class (``BaichuanM1Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``BaichuanM1DecoderLayer`` decoder blocks.
        norm: Final RMS normalization.
        swa_layers: Set of layer indices that use sliding window attention.

    Example::

        >>> config = BaichuanM1Config(vocab_size=102400, hidden_size=2048)
        >>> model = BaichuanM1Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = BaichuanM1Config

    def __init__(self, config: BaichuanM1Config):
        """Initialize the Baichuan M1 base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BaichuanM1DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.swa_layers = set(config.sliding_window_layers)

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
        """Transform checkpoint weights for compatibility.

        Normalizes the ``lm_head`` weight matrix (L2-norm per row) when
        not quantized, and removes rotary embedding buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with normalized lm_head and
            rotary buffers removed.
        """
        is_quantized = "lm_head.scales" in weights
        if not is_quantized and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.astype(mx.float32)
            norm = mx.linalg.norm(w, axis=-1, keepdims=True)
            w = (w / (norm + 1e-7)).astype(dtype)
            weights["lm_head.weight"] = w

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=BaichuanM1Config, model_type="baichuan_m1")
class BaichuanM1ForCausalLM(BaseCausalLMModule[BaichuanM1Model, BaichuanM1Config]):
    """Baichuan M1 model with a causal language modeling head.

    Wraps ``BaichuanM1Model`` with a linear projection to vocabulary
    logits. Includes lm_head weight normalization during sanitization.

    Attributes:
        config_class: The configuration class (``BaichuanM1Config``).

    Example::

        >>> config = BaichuanM1Config(vocab_size=102400, hidden_size=2048)
        >>> model = BaichuanM1ForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = BaichuanM1Config

    def __init__(self, config: BaichuanM1Config):
        """Initialize the Baichuan M1 causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=BaichuanM1Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Normalize lm_head weights and apply base sanitization.

        L2-normalizes the ``lm_head.weight`` per row (when not quantized),
        then delegates to the base class sanitization.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        is_quantized = "lm_head.scales" in weights
        if not is_quantized and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.astype(mx.float32)
            norm = mx.linalg.norm(w, axis=-1, keepdims=True)
            w = (w / (norm + 1e-7)).astype(dtype)
            weights["lm_head.weight"] = w

        return super().sanitize(weights)


__all__ = ("BaichuanM1ForCausalLM", "BaichuanM1Model")
