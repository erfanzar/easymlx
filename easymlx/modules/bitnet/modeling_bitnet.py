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

"""BitNet MLX implementation (serving/inference only).

Structure:
  BitNetConfig -> BitNetAttention -> BitNetMLP -> BitNetDecoderLayer
  -> BitNetModel -> BitNetForCausalLM

Key differences from standard transformers:
  - Uses nn.Linear (BitLinear weights handled at load time)
  - ReLU^2 activation in MLP gating
  - Additional sub-norms (attn_sub_norm, ffn_sub_norm) after projections
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
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .bitnet_configuration import BitNetConfig

CacheView = TransformerCacheView | PageCacheView


class BitNetAttention(nn.Module):
    """Multi-head attention for BitNet with post-attention sub-normalization.

    Applies standard GQA attention with RoPE, then an RMSNorm sub-norm
    on the concatenated attention output before the output projection.
    BitLinear ternary weights are handled at load time.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query linear projection.
        k_proj: Key linear projection.
        v_proj: Value linear projection.
        o_proj: Output linear projection.
        attn_sub_norm: RMSNorm applied after attention concatenation.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = BitNetConfig(hidden_size=2048, num_attention_heads=32)
        >>> attn = BitNetAttention(config)
    """

    def __init__(self, config: BitNetConfig):
        """Initialize BitNet attention.

        Args:
            config: Model configuration with attention hyperparameters.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.attn_sub_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        """Run attention forward pass with post-attention sub-norm.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
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
        output = attn.reshape(*lead, -1)
        output = self.attn_sub_norm(output)
        return self.o_proj(output)


class BitNetMLP(nn.Module):
    """Feed-forward network with ReLU^2 gating and FFN sub-norm for BitNet.

    Uses ReLU squared activation: ``relu(gate_proj(x))^2 * up_proj(x)``,
    followed by an RMSNorm sub-norm before the down projection.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
        ffn_sub_norm: RMSNorm applied before down projection.

    Example::

        >>> config = BitNetConfig(hidden_size=2048, intermediate_size=5504)
        >>> mlp = BitNetMLP(config)
    """

    def __init__(self, config: BitNetConfig):
        """Initialize the BitNet MLP.

        Args:
            config: Model configuration with ``hidden_size``,
                ``intermediate_size``, and ``mlp_bias``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.ffn_sub_norm = nn.RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply ReLU^2-gated MLP with FFN sub-norm.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        x = nn.relu(self.gate_proj(hidden_states)) ** 2 * self.up_proj(hidden_states)
        x = self.ffn_sub_norm(x)
        return self.down_proj(x)


class BitNetDecoderLayer(nn.Module):
    """Single BitNet decoder layer with pre-norm residual connections.

    Attributes:
        self_attn: BitNet attention with sub-norm.
        mlp: ReLU^2-gated MLP with FFN sub-norm.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
    """

    def __init__(self, config: BitNetConfig):
        """Initialize the BitNet decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
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


@register_module(task_type=TaskType.BASE_MODULE, config=BitNetConfig, model_type="bitnet")
class BitNetModel(EasyMLXBaseModule):
    """Base BitNet transformer model for inference.

    Implements a decoder-only transformer with ternary BitLinear
    weights (handled at load time), ReLU^2 MLP gating, and
    post-projection sub-norms in both attention and MLP.

    Attributes:
        config_class: The configuration class (``BitNetConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``BitNetDecoderLayer`` decoder blocks.
        norm: Final RMS normalization.

    Example::

        >>> config = BitNetConfig(vocab_size=32000, hidden_size=2048)
        >>> model = BitNetModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = BitNetConfig

    def __init__(self, config: BitNetConfig):
        """Initialize the BitNet base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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


@register_module(task_type=TaskType.CAUSAL_LM, config=BitNetConfig, model_type="bitnet")
class BitNetForCausalLM(BaseCausalLMModule[BitNetModel, BitNetConfig]):
    """BitNet model with a causal language modeling head.

    Wraps ``BitNetModel`` with a linear projection to vocabulary logits.
    Supports tied word embeddings.

    Attributes:
        config_class: The configuration class (``BitNetConfig``).

    Example::

        >>> config = BitNetConfig(vocab_size=32000, hidden_size=2048)
        >>> model = BitNetForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
        >>> output.logits.shape
        (1, 3, 32000)
    """

    config_class = BitNetConfig

    def __init__(self, config: BitNetConfig):
        """Initialize the BitNet causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=BitNetModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )


__all__ = ("BitNetForCausalLM", "BitNetModel")
