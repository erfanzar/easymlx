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

"""PhiMoE MLX model implementation for serving and inference.

Structure mirrors EasyDeL's phimoe:
  PhimoeConfig -> PhimoeAttention -> PhimoeSparseMoeBlock
  -> PhimoeDecoderLayer -> PhimoeModel -> PhimoeForCausalLM

Uses LayerNorm (not RMSNorm) and supports SuScaledRoPE.
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
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .phimoe_configuration import PhimoeConfig

CacheView = TransformerCacheView | PageCacheView


class PhimoeAttention(nn.Module):
    """Multi-head attention with grouped-query attention for PhiMoE.

    Supports SuScaledRoPE via the rope_scaling config.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
    """

    def __init__(self, config: PhimoeConfig):
        """Initialize PhiMoE attention layer.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``num_attention_heads`` is not divisible by
                ``num_key_value_heads``.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Compute multi-head attention with SuScaledRoPE support.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
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


class PhimoeSparseMoeBlock(nn.Module):
    """Sparse Mixture-of-Experts block with SwitchGLU for PhiMoE.

    Attributes:
        num_experts: Total number of experts.
        num_experts_per_tok: Number of experts activated per token.
        gate: Router linear projection.
        switch_mlp: SwitchGLU expert bank.
    """

    def __init__(self, config: PhimoeConfig):
        """Initialize PhiMoE sparse MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, self.num_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to top-k experts and aggregate results.

        Args:
            hidden_states: Input tensor of shape
                ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of the same shape as input, weighted by
            softmax router scores.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, orig_shape[-1])

        gate_logits = self.gate(hidden_states)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gate_logits, kth=k - 1, axis=-1)[:, :k])
        scores = mx.take_along_axis(gate_logits, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(orig_shape)


class PhimoeDecoderLayer(nn.Module):
    """Single decoder layer for PhiMoE.

    Uses LayerNorm (not RMSNorm).

    Attributes:
        self_attn: Multi-head attention module.
        block_sparse_moe: Sparse MoE block.
        input_layernorm: Pre-attention layer normalization.
        post_attention_layernorm: Post-attention layer normalization.
    """

    def __init__(self, config: PhimoeConfig):
        """Initialize PhiMoE decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = PhimoeAttention(config)
        self.block_sparse_moe = PhimoeSparseMoeBlock(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run one decoder layer with attention + MoE residual connections.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
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
        hidden_states = residual + self.block_sparse_moe(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=PhimoeConfig, model_type="phimoe")
class PhimoeModel(EasyMLXBaseModule):
    """Base PhiMoE transformer model.

    Uses LayerNorm throughout.

    Attributes:
        config_class: The associated configuration class (``PhimoeConfig``).
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final layer normalization.
    """

    config_class = PhimoeConfig

    def __init__(self, config: PhimoeConfig):
        """Initialize the base PhiMoE model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhimoeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            input_embeddings: Pre-computed embeddings (overrides
                ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final LayerNorm.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=PhimoeConfig, model_type="phimoe")
class PhimoeForCausalLM(BaseCausalLMModule[PhimoeModel, PhimoeConfig]):
    """PhiMoE model with a causal language modeling head.

    Includes a ``sanitize`` method that stacks per-expert weights from
    upstream checkpoints into the ``SwitchGLU`` format expected by
    EasyMLX.

    Attributes:
        config_class: Associated configuration class (``PhimoeConfig``).

    Example:
        >>> model = PhimoeForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = PhimoeConfig

    def __init__(self, config: PhimoeConfig):
        """Initialize the PhiMoE causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=PhimoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict) -> dict:
        """Sanitize upstream weights by stacking per-expert parameters.

        Transforms individual expert weights from
        ``model.layers.N.block_sparse_moe.experts.E.{w1,w2,w3}.weight``
        into stacked ``model.layers.N.block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}.weight``.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.block_sparse_moe.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{n}.{k}")
                            for e in range(self.config.num_local_experts)
                        ]
                        weights[f"{prefix}.block_sparse_moe.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return weights


__all__ = ("PhimoeForCausalLM", "PhimoeModel")
