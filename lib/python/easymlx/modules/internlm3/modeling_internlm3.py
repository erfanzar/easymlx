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

"""InternLM3 MLX implementation (serving/inference only).

Structure mirrors EasyDeL's internlm3:
  InternLM3Config -> InternLM3Attention -> InternLM3MLP -> InternLM3DecoderLayer
  -> InternLM3Model -> InternLM3ForCausalLM

InternLM3 is similar to InternLM2 but uses separate Q/K/V projections
with optional QKV bias.
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

from .internlm3_configuration import InternLM3Config

CacheView = TransformerCacheView | PageCacheView


class InternLM3Attention(nn.Module):
    """InternLM3 attention with separate Q/K/V projections.

    Unlike InternLM2 which uses a fused QKV projection, InternLM3 has
    independent ``q_proj``, ``k_proj``, and ``v_proj`` linear layers with
    optional QKV bias. Supports DynamicNTKScaling RoPE.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = InternLM3Config(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> attn = InternLM3Attention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM3Config):
        """Initialize InternLM3Attention.

        Args:
            config: InternLM3 model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        qkv_bias = config.qkv_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=qkv_bias)

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
        """Compute attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
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


class InternLM3MLP(nn.Module):
    """SwiGLU feed-forward network for InternLM3.

    Uses standard ``gate_proj``/``up_proj``/``down_proj`` naming with
    optional bias controlled by ``config.bias``.

    Attributes:
        gate_proj: Gate projection.
        down_proj: Down projection.
        up_proj: Up projection.

    Example:
        >>> config = InternLM3Config(hidden_size=256, intermediate_size=512)
        >>> mlp = InternLM3MLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM3Config):
        """Initialize InternLM3MLP.

        Args:
            config: InternLM3 model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class InternLM3DecoderLayer(nn.Module):
    """Single InternLM3 decoder layer.

    Pre-norm architecture with separate Q/K/V attention and SwiGLU MLP.

    Attributes:
        self_attn: InternLM3 attention sub-layer.
        mlp: SwiGLU MLP sub-layer.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = InternLM3Config(hidden_size=256, num_attention_heads=8)
        >>> layer = InternLM3DecoderLayer(config)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM3Config):
        """Initialize InternLM3DecoderLayer.

        Args:
            config: InternLM3 model configuration.
        """
        super().__init__()
        self.self_attn = InternLM3Attention(config)
        self.mlp = InternLM3MLP(config)
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


@register_module(task_type=TaskType.BASE_MODULE, config=InternLM3Config, model_type="internlm3")
class InternLM3Model(EasyMLXBaseModule):
    """Base InternLM3 transformer model (no LM head).

    InternLM3 uses separate Q/K/V projections (unlike InternLM2's fused
    ``wqkv``), optional QKV bias, and DynamicNTKScaling RoPE.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding layer.
        layers: List of ``InternLM3DecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = InternLM3Config(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = InternLM3Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = InternLM3Config

    def __init__(self, config: InternLM3Config):
        """Initialize InternLM3Model.

        Args:
            config: InternLM3 model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [InternLM3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the InternLM3 transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove rotary embedding inverse-frequency buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Filtered weight dictionary.
        """
        return {
            k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k and "attention.rope.inv_freq" not in k
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=InternLM3Config, model_type="internlm3")
class InternLM3ForCausalLM(BaseCausalLMModule[InternLM3Model, InternLM3Config]):
    """InternLM3 transformer with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = InternLM3Config(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = InternLM3ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = InternLM3Config

    def __init__(self, config: InternLM3Config):
        """Initialize InternLM3ForCausalLM.

        Args:
            config: InternLM3 model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=InternLM3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove rotary inv_freq buffers and optionally drop tied LM head weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """
        weights = {
            k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k and "attention.rope.inv_freq" not in k
        }
        if self._tie_word_embeddings:
            weights.pop(f"{self._lm_head_name}.weight", None)
        return weights


__all__ = ("InternLM3ForCausalLM", "InternLM3Model")
