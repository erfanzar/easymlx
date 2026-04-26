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

"""InternLM2 MLX implementation (serving/inference only).

Structure mirrors EasyDeL's internlm2:
  InternLM2Config -> InternLM2Attention -> InternLM2MLP -> InternLM2DecoderLayer
  -> InternLM2Model -> InternLM2ForCausalLM

InternLM2 uses a fused QKV projection and DynamicNTKScaling RoPE.
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

from .internlm2_configuration import InternLM2Config

CacheView = TransformerCacheView | PageCacheView


class InternLM2Attention(nn.Module):
    """InternLM2 attention with fused QKV projection (wqkv).

    Uses a single fused linear projection ``wqkv`` that outputs all Q, K, V
    states in a single matmul for efficiency. The fused output is split
    using the GQA group structure: each KV group contains
    ``num_kv_groups`` query heads plus one K and one V head.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        num_kv_groups: Number of query heads per KV group.
        wqkv: Fused QKV linear projection.
        wo: Output linear projection.
        rope: Rotary position embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = InternLM2Config(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> attn = InternLM2Attention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM2Config):
        """Initialize InternLM2Attention.

        Args:
            config: InternLM2 model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.wqkv = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.bias,
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.bias)

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
        """Compute attention forward pass with fused QKV.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]

        qkv_states = self.wqkv(hidden_states)
        qkv_states = qkv_states.reshape(*lead, -1, 2 + self.num_kv_groups, self.head_dim)

        queries = qkv_states[..., : self.num_kv_groups, :]
        queries = queries.reshape(*lead, -1, self.head_dim)
        keys = qkv_states[..., -2, :]
        values = qkv_states[..., -1, :]

        queries = queries.reshape(*lead, self.num_heads, self.head_dim)
        keys = keys.reshape(*lead, self.num_kv_heads, self.head_dim)
        values = values.reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.wo(attn.reshape(*lead, -1))


class InternLM2MLP(nn.Module):
    """SwiGLU feed-forward network for InternLM2.

    Uses the ``w1``/``w2``/``w3`` naming convention from the InternLM2
    architecture: ``w2(silu(w1(x)) * w3(x))``.

    Attributes:
        w1: Gate projection (no bias).
        w2: Down projection (no bias).
        w3: Up projection (no bias).

    Example:
        >>> config = InternLM2Config(hidden_size=256, intermediate_size=512)
        >>> mlp = InternLM2MLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM2Config):
        """Initialize InternLM2MLP.

        Args:
            config: InternLM2 model configuration.
        """
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.w2(nn.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class InternLM2DecoderLayer(nn.Module):
    """Single InternLM2 decoder layer.

    Pre-norm architecture with fused-QKV attention and SwiGLU MLP.
    Uses ``attention_norm`` and ``ffn_norm`` naming.

    Attributes:
        attention: InternLM2 attention sub-layer.
        feed_forward: SwiGLU MLP sub-layer.
        attention_norm: RMSNorm before attention.
        ffn_norm: RMSNorm before MLP.

    Example:
        >>> config = InternLM2Config(hidden_size=256, num_attention_heads=8)
        >>> layer = InternLM2DecoderLayer(config)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: InternLM2Config):
        """Initialize InternLM2DecoderLayer.

        Args:
            config: InternLM2 model configuration.
        """
        super().__init__()
        self.attention = InternLM2Attention(config)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = residual + self.attention(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=InternLM2Config, model_type="internlm2")
class InternLM2Model(EasyMLXBaseModule):
    """Base InternLM2 transformer model (no LM head).

    InternLM2 uses a fused QKV projection (``wqkv``) and
    ``w1``/``w2``/``w3`` MLP naming. Token embeddings use
    ``tok_embeddings`` instead of ``embed_tokens``.

    Attributes:
        config_class: Associated configuration class.
        tok_embeddings: Token embedding layer.
        layers: List of ``InternLM2DecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = InternLM2Config(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = InternLM2Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = InternLM2Config

    def __init__(self, config: InternLM2Config):
        """Initialize InternLM2Model.

        Args:
            config: InternLM2 model configuration.
        """
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the InternLM2 transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
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
            hidden_states = self.tok_embeddings(input_ids)

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
        """Remove rotary embedding inverse-frequency buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Filtered weight dictionary without rotary inv_freq entries.
        """
        return {
            k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k and "attention.rope.inv_freq" not in k
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=InternLM2Config, model_type="internlm2")
class InternLM2ForCausalLM(BaseCausalLMModule[InternLM2Model, InternLM2Config]):
    """InternLM2 transformer with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = InternLM2Config(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = InternLM2ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = InternLM2Config

    def __init__(self, config: InternLM2Config):
        """Initialize InternLM2ForCausalLM.

        Args:
            config: InternLM2 model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=InternLM2Model,
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


__all__ = ("InternLM2ForCausalLM", "InternLM2Model")
