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

"""LFM2 MLX model implementation for serving and inference.

LFM2 is a hybrid architecture mixing short convolution layers with
full attention layers. The conv layers use a gated depthwise conv1d,
while attention layers use standard RoPE-based multi-head attention
with QK layernorm.
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

from .lfm2_configuration import Lfm2Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like value to an int32 mx.array.

    Args:
        values: Input values to convert. Can be ``None``, an ``mx.array``,
            or any iterable convertible to an array.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None`` if the input is ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Lfm2Attention(nn.Module):
    """LFM2 multi-head attention with QK layernorm and RoPE.

    Applies separate RMSNorm to query and key projections before computing
    attention scores, and uses RoPE for positional encoding.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor (``head_dim ** -0.5``).
        q_layernorm: RMSNorm applied to query projections.
        k_layernorm: RMSNorm applied to key projections.
        rope: Rotary position embedding module.
        attention_performer: Attention computation engine.

    Example:
        >>> config = Lfm2Config(hidden_size=64, num_attention_heads=4)
        >>> attn = Lfm2Attention(config)
    """

    def __init__(self, config: Lfm2Config):
        """Initialize LFM2 attention layer.

        Args:
            config: Model configuration containing attention hyperparameters.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Compute attention-weighted output with QK layernorm and RoPE.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask. Can be an ``mx.array`` boolean mask,
                a string hint, or ``None`` for no masking.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        queries = self.q_layernorm(queries)
        keys = self.k_layernorm(keys)

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


class Lfm2MLP(nn.Module):
    """LFM2 SwiGLU feed-forward network with auto-adjusted dimensions.

    Computes ``w2(silu(w1(x)) * w3(x))``. When ``block_auto_adjust_ff_dim``
    is enabled in the config, the intermediate dimension is adjusted using
    the 2/3 heuristic and aligned to ``block_multiple_of``.

    Attributes:
        w1: Gate projection linear layer.
        w3: Up projection linear layer.
        w2: Down projection linear layer.

    Example:
        >>> config = Lfm2Config(block_dim=64, block_ff_dim=128)
        >>> mlp = Lfm2MLP(config)
    """

    def __init__(self, config: Lfm2Config):
        """Initialize LFM2 MLP.

        Args:
            config: Model configuration with MLP dimension parameters.
        """
        super().__init__()
        ff_dim = config.block_ff_dim
        if config.block_auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if config.block_ffn_dim_multiplier is not None:
                ff_dim = int(config.block_ffn_dim_multiplier * ff_dim)
            ff_dim = config.block_multiple_of * ((ff_dim + config.block_multiple_of - 1) // config.block_multiple_of)
        self.w1 = nn.Linear(config.block_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(config.block_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, config.block_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU feed-forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., block_dim)``.

        Returns:
            Output tensor of shape ``(..., block_dim)``.
        """
        return self.w2(nn.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class Lfm2ShortConv(nn.Module):
    """LFM2 gated depthwise short convolution layer.

    Projects input to three gates (B, C, x), applies a depthwise conv1d on
    the gated signal ``B * x``, then gates the output with C before final
    projection.

    Attributes:
        hidden_size: Model hidden dimension.
        L_cache: Convolution kernel size.
        bias: Whether projections use bias.
        conv: Depthwise 1D convolution module.
        in_proj: Input projection to 3x hidden_size (B, C, x gates).
        out_proj: Output projection back to hidden_size.

    Example:
        >>> config = Lfm2Config(hidden_size=64, conv_L_cache=4)
        >>> conv_layer = Lfm2ShortConv(config)
    """

    def __init__(self, config: Lfm2Config):
        """Initialize LFM2 short convolution layer.

        Args:
            config: Model configuration with convolution hyperparameters.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Apply gated depthwise short convolution.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Unused; accepted for API compatibility with attention layers.
            cache_view: Unused; accepted for API compatibility.
            cache_metadata: Unused; accepted for API compatibility.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        BCx = self.in_proj(hidden_states)
        B_gate, C_gate, x = mx.split(BCx, 3, axis=-1)
        Bx = B_gate * x

        Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])
        conv_out = self.conv(Bx)
        y = C_gate * conv_out
        return self.out_proj(y)


class Lfm2DecoderLayer(nn.Module):
    """Single LFM2 decoder layer selecting between attention and convolution.

    Each layer contains either a full attention sub-layer or a gated short
    convolution sub-layer, followed by a SwiGLU MLP. Both paths use
    pre-normalization with RMSNorm and residual connections.

    Attributes:
        is_attention_layer: Whether this layer uses attention (vs. convolution).
        feed_forward: SwiGLU MLP sub-layer.
        operator_norm: Pre-operator RMSNorm.
        ffn_norm: Pre-FFN RMSNorm.

    Example:
        >>> config = Lfm2Config(hidden_size=64, num_hidden_layers=2)
        >>> layer = Lfm2DecoderLayer(config, layer_idx=0)
    """

    def __init__(self, config: Lfm2Config, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack, used to determine
                whether to use attention or convolution.
        """
        super().__init__()
        self.is_attention_layer = layer_idx in config.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config)

        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the decoder layer (attention or conv, then MLP).

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask (passed to the attention sub-layer).
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        normed = self.operator_norm(hidden_states)

        if self.is_attention_layer:
            hidden_states = residual + self.self_attn(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        else:
            hidden_states = residual + self.conv(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )

        residual = hidden_states
        hidden_states = residual + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Lfm2Config, model_type="lfm2")
class Lfm2Model(EasyMLXBaseModule):
    """Base LFM2 hybrid conv-attention transformer model.

    Embeds input tokens, passes them through a stack of decoder layers
    (interleaving convolution and attention based on ``full_attn_idxs``),
    and applies a final RMSNorm.

    Attributes:
        config_class: The associated configuration class (``Lfm2Config``).
        embed_tokens: Token embedding table.
        layers: List of ``Lfm2DecoderLayer`` modules.
        embedding_norm: Final RMSNorm applied to the output.

    Example:
        >>> config = Lfm2Config(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = Lfm2Model(config)
    """

    config_class = Lfm2Config

    def __init__(self, config: Lfm2Config):
        """Initialize the base LFM2 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Lfm2DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the LFM2 backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings. If provided, ``input_ids``
                is ignored.
            cache_views: Per-layer KV cache views for incremental decoding.
                Length must match the number of layers.
            cache_metadata: Paged attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` after
            final normalization.

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

        return self.embedding_norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream weights for LFM2 compatibility.

        Transposes conv1d weights from ``(out, kernel, in)`` to
        ``(out, in, kernel)`` format when needed.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary with corrected conv1d weight shapes.
        """
        sanitized = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)
            sanitized[name] = param
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=Lfm2Config, model_type="lfm2")
class Lfm2ForCausalLM(BaseCausalLMModule[Lfm2Model, Lfm2Config]):
    """LFM2 causal language model with an LM head.

    Wraps ``Lfm2Model`` with a language modeling head for next-token
    prediction. Supports tied input/output embeddings.

    Attributes:
        config_class: The associated configuration class (``Lfm2Config``).

    Example:
        >>> config = Lfm2Config(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = Lfm2ForCausalLM(config)
    """

    config_class = Lfm2Config

    def __init__(self, config: Lfm2Config):
        """Initialize the LFM2 causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Lfm2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights by delegating to the base model and parent.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.base_model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("Lfm2ForCausalLM", "Lfm2Model")
