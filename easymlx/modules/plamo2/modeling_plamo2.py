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

"""PLaMo2 MLX implementation (serving/inference only).

Structure:
  Plamo2Config -> Plamo2Attention / Plamo2Mamba -> Plamo2MLP -> Plamo2DecoderLayer
  -> Plamo2Model -> Plamo2ForCausalLM

Key features:
  - Hybrid Attention + Mamba: some layers use standard attention, others use Mamba SSM.
  - Custom RMSNorm (offset-based: weight + offset).
  - QK RMSNorm (weight-less) + learnable scaling.
  - Fused QKV projection for attention layers.
  - SwiGLU MLP with fused gate+up projection.
  - Conv1d weight sanitization (axis conversion).
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

from .plamo2_configuration import Plamo2Config

CacheView = TransformerCacheView | PageCacheView


class Plamo2RMSNorm(nn.Module):
    """PLaMo2-style RMSNorm with configurable weight offset.

    The effective normalization weight is ``self.weight + offset``, allowing
    the norm to be initialized near identity (offset=1.0) or with reduced
    scale (e.g. offset=0.2 for post-mixer norms).

    Attributes:
        weight: Learnable weight parameter initialized to zeros.
        variance_epsilon: Epsilon for numerical stability.
        offset: Constant added to weight before normalization.

    Example:
        >>> norm = Plamo2RMSNorm(4096, eps=1e-6, offset=1.0)
        >>> out = norm(hidden_states)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, offset: float = 1.0):
        """Initialize PLaMo2 RMSNorm.

        Args:
            hidden_size: Number of features.
            eps: Epsilon for numerical stability.
            offset: Constant added to the learned weight.
        """
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply offset-based RMSNorm.

        Args:
            hidden_states: Input tensor of arbitrary shape with last
                dimension equal to ``hidden_size``.

        Returns:
            Normalized tensor with the same shape.
        """
        return mx.fast.rms_norm(hidden_states, self.weight + self.offset, self.variance_epsilon)


class Plamo2Attention(nn.Module):
    """PLaMo2 attention with fused QKV, weight-less QK RMSNorm, and learnable scaling.

    Uses a single fused ``qkv_proj`` for Q, K, and V. Applies weight-less
    RMSNorm to Q and K followed by learnable per-head scaling weights
    (``q_weight``, ``k_weight``).

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention logit scaling factor.
        qkv_proj: Fused Q/K/V linear projection.
        o_proj: Output projection.
        q_weight: Learnable per-head Q scaling.
        k_weight: Learnable per-head K scaling.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example:
        >>> attn = Plamo2Attention(config)
        >>> out = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: Plamo2Config):
        """Initialize PLaMo2 attention layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size_per_head)
        self.scale = self.head_dim**-0.5

        q_dim = self.num_heads * self.head_dim
        k_dim = self.num_kv_heads * self.head_dim
        v_dim = self.num_kv_heads * self.head_dim
        self.q_proj_dim = q_dim
        self.k_proj_dim = k_dim

        self.qkv_proj = nn.Linear(config.hidden_size, q_dim + k_dim + v_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_weight = mx.ones((self.num_heads, self.head_dim))
        self.k_weight = mx.ones((self.num_kv_heads, self.head_dim))

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=config.rope_traditional,
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
        """Compute attention with fused QKV, QK RMSNorm, and learnable scaling.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(qkv, [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim], axis=-1)

        q = q.reshape(*lead, self.num_heads, self.head_dim)
        k = k.reshape(*lead, self.num_kv_heads, self.head_dim)
        v = v.reshape(*lead, self.num_kv_heads, self.head_dim)

        # QK RMSNorm (weight-less) + learnable scaling
        q = mx.fast.rms_norm(q, weight=None, eps=1e-6) * self.q_weight
        k = mx.fast.rms_norm(k, weight=None, eps=1e-6) * self.k_weight

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


class Plamo2MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection for PLaMo2.

    Uses a single ``gate_up_proj`` that outputs gate and up values
    concatenated, which are then split for the SwiGLU computation.

    Attributes:
        gate_up_proj: Fused gate + up projection.
        down_proj: Down projection.

    Example:
        >>> mlp = Plamo2MLP(config)
        >>> out = mlp(hidden_states)
    """

    def __init__(self, config: Plamo2Config):
        """Initialize PLaMo2 MLP.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply fused gate+up -> SwiGLU -> down projection.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        h = self.gate_up_proj(hidden_states)
        gate, up = mx.split(h, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class Plamo2DecoderLayer(nn.Module):
    """PLaMo2 decoder layer (attention-only variant).

    Uses separate pre/post norms for the mixer (attention) and MLP
    sub-layers, each with different offset values to control initial
    residual scaling.

    Attributes:
        is_mamba: Always False for the attention-only variant.
        mixer: Attention module.
        mlp: Feed-forward network.
        pre_mixer_norm: Pre-attention RMSNorm (offset=1.0).
        post_mixer_norm: Post-attention RMSNorm (offset=1/5).
        pre_mlp_norm: Pre-MLP RMSNorm (offset=1.0).
        post_mlp_norm: Post-MLP RMSNorm (offset=1/5^1.5).

    Example:
        >>> layer = Plamo2DecoderLayer(config)
        >>> out = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: Plamo2Config):
        """Initialize PLaMo2 decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.is_mamba = False
        self.mixer = Plamo2Attention(config)
        self.mlp = Plamo2MLP(config)
        self.pre_mixer_norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0)
        self.post_mixer_norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5)
        self.pre_mlp_norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0)
        self.post_mlp_norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5))

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer with pre/post norms and residual connections.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)

        attn_out = self.mixer(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        attn_out = self.post_mixer_norm(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        mlp_out = self.post_mlp_norm(mlp_out)
        return residual + mlp_out


@register_module(task_type=TaskType.BASE_MODULE, config=Plamo2Config, model_type="plamo2")
class Plamo2Model(EasyMLXBaseModule):
    """Base PLaMo2 transformer model (attention-only path for EasyMLX).

    Note: The upstream PLaMo2 supports hybrid Attention+Mamba layers.
    This implementation provides the attention-only path suitable for
    standard transformer inference. Mamba SSM layers are not yet
    supported in EasyMLX.
    """

    config_class = Plamo2Config

    def __init__(self, config: Plamo2Config):
        """Initialize the base PLaMo2 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Plamo2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = Plamo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final offset-based RMSNorm.

        Raises:
            ValueError: If ``cache_views`` length does not match the
                number of layers.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=Plamo2Config, model_type="plamo2")
class Plamo2ForCausalLM(BaseCausalLMModule[Plamo2Model, Plamo2Config]):
    """PLaMo2 model with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class (``Plamo2Config``).

    Example:
        >>> model = Plamo2ForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = Plamo2Config

    def __init__(self, config: Plamo2Config):
        """Initialize the PLaMo2 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Plamo2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights including conv1d axis transposition.

        Transposes conv1d weights from upstream ``(channels, 1, kernel)``
        layout to the ``(channels, kernel, 1)`` layout expected by MLX.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Sanitized weight dictionary with transposed conv1d weights.
        """
        weights = super().sanitize(weights)
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                sanitized[k] = v.moveaxis(2, 1)
            else:
                sanitized[k] = v
        return sanitized


__all__ = ("Plamo2ForCausalLM", "Plamo2Model")
