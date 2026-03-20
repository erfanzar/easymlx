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

"""Mixtral MLX model implementation for serving and inference.

Structure mirrors EasyDeL's mixtral:
  MixtralConfig -> MixtralAttention -> MixtralSparseMoeBlock
  -> MixtralDecoderLayer -> MixtralModel -> MixtralForCausalLM

Unified ``__call__`` at every level -- cache_view is either
``TransformerCacheView`` (standard) or ``PageCacheView`` (paged serving).
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

from .mixtral_configuration import MixtralConfig

CacheView = TransformerCacheView | PageCacheView


class MixtralAttention(nn.Module):
    """Multi-head attention with grouped-query attention for Mixtral.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
    """

    def __init__(self, config: MixtralConfig):
        """Initialize Mixtral GQA attention.

        Args:
            config (MixtralConfig): Model configuration.

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

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

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
        """Compute GQA attention forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)`` or
                ``(num_tokens, D)`` for paged layout.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output tensor matching input leading dimensions.
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


class MixtralSparseMoeBlock(nn.Module):
    """Sparse Mixture-of-Experts block with SwitchGLU for Mixtral.

    Attributes:
        num_experts: Total number of experts.
        num_experts_per_tok: Number of experts activated per token.
        gate: Router linear projection.
        switch_mlp: SwitchGLU expert bank.
    """

    def __init__(self, config: MixtralConfig):
        """Initialize Mixtral sparse MoE block.

        Args:
            config (MixtralConfig): Model configuration with expert settings.
        """
        super().__init__()
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, self.num_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to top-k experts with softmax scoring.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.

        Returns:
            mx.array: Expert-weighted output of shape ``(B, L, D)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, orig_shape[-1])

        gate_logits = self.gate(hidden_states)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gate_logits, kth=k - 1, axis=-1)[:, :k])
        scores = mx.softmax(gate_logits, axis=-1, precise=True)
        scores = mx.take_along_axis(scores, inds, axis=-1)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(orig_shape)


class MixtralDecoderLayer(nn.Module):
    """Single decoder layer for Mixtral.

    Attributes:
        self_attn: Multi-head attention module.
        block_sparse_moe: Sparse MoE block.
        input_layernorm: Pre-attention RMS normalization.
        post_attention_layernorm: Post-attention RMS normalization.
    """

    def __init__(self, config: MixtralConfig):
        """Initialize Mixtral decoder layer.

        Args:
            config (MixtralConfig): Model configuration.
        """
        super().__init__()
        self.self_attn = MixtralAttention(config)
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
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
        """Forward pass through the decoder layer.

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


@register_module(task_type=TaskType.BASE_MODULE, config=MixtralConfig, model_type="mixtral")
class MixtralModel(EasyMLXBaseModule):
    """Base Mixtral transformer model.

    Attributes:
        config_class: The associated configuration class (``MixtralConfig``).
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
    """

    config_class = MixtralConfig

    def __init__(self, config: MixtralConfig):
        """Initialize Mixtral base model.

        Args:
            config (MixtralConfig): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MixtralDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the Mixtral base model.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings.
            cache_views (list[CacheView] | None): Per-layer KV cache views.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=MixtralConfig, model_type="mixtral")
class MixtralForCausalLM(BaseCausalLMModule[MixtralModel, MixtralConfig]):
    """Mixtral model with a causal language modeling head.

    Attributes:
        config_class: The associated configuration class (``MixtralConfig``).
    """

    config_class = MixtralConfig

    def __init__(self, config: MixtralConfig):
        """Initialize Mixtral causal LM.

        Args:
            config (MixtralConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MixtralModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict) -> dict:
        """Sanitize upstream weights by stacking per-expert parameters.

        Transforms individual expert weights from
        ``model.layers.N.block_sparse_moe.experts.E.{w1,w2,w3}.weight``
        into stacked ``model.layers.N.block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}.weight``.
        Also removes ``rotary_emb.inv_freq`` keys and handles weight tying.

        Args:
            weights (dict): Raw checkpoint weight dictionary.

        Returns:
            dict: Sanitized weight dictionary with stacked expert weights.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

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


__all__ = ("MixtralForCausalLM", "MixtralModel")
