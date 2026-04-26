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

"""MiniMax MLX implementation (serving/inference only).

Structure:
  MiniMaxConfig -> MiniMaxAttention -> MiniMaxSparseMoeBlock
  -> MiniMaxDecoderLayer -> MiniMaxModel -> MiniMaxForCausalLM

Key features:
  - Sparse MoE with sigmoid gating and e_score_correction_bias
  - QK normalization
  - SwitchGLU expert computation
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

from .minimax_configuration import MiniMaxConfig

CacheView = TransformerCacheView | PageCacheView


class MiniMaxAttention(nn.Module):
    """Multi-head attention with optional QK normalization for MiniMax.

    When ``use_qk_norm`` is enabled, RMSNorm is applied to the full Q and K
    projections before reshaping into heads.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
        use_qk_norm: Whether QK normalization is enabled.

    Example:
        >>> config = MiniMaxConfig(use_qk_norm=True)
        >>> attn = MiniMaxAttention(config)
        >>> out = attn(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: MiniMaxConfig):
        """Initialize MiniMax attention.

        Args:
            config (MiniMaxConfig): Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim * self.num_heads, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim * self.num_kv_heads, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=config.rotary_dim,
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
        """Compute attention forward pass with optional QK normalization.

        Args:
            hidden_states (mx.array): Input tensor of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output tensor of shape ``(B, L, D)``.
        """
        lead = hidden_states.shape[:-1]

        queries, keys, values = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        q = queries.reshape(*lead, self.num_heads, self.head_dim)
        k = keys.reshape(*lead, self.num_kv_heads, self.head_dim)
        v = values.reshape(*lead, self.num_kv_heads, self.head_dim)

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


class MiniMaxSparseMoeBlock(nn.Module):
    """Sparse MoE block with sigmoid gating for MiniMax.

    Uses sigmoid scoring instead of softmax, with an additive
    ``e_score_correction_bias`` for expert selection. After selecting the
    top-k experts, the original (unbiased) sigmoid scores are used for
    weighted combination, normalized to sum to 1.

    Attributes:
        num_experts_per_tok: Number of experts activated per token.
        num_local_experts: Total number of routing experts.
        gate: Router linear projection.
        switch_mlp: SwitchGLU expert bank.
        e_score_correction_bias: Additive bias for expert selection scores.

    Example:
        >>> config = MiniMaxConfig(num_local_experts=8, num_experts_per_tok=2)
        >>> moe = MiniMaxSparseMoeBlock(config)
        >>> out = moe(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: MiniMaxConfig):
        """Initialize MiniMax sparse MoE block.

        Args:
            config (MiniMaxConfig): Model configuration with expert settings.
        """
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_local_experts = config.num_local_experts
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, config.num_local_experts)
        self.e_score_correction_bias = mx.zeros((config.num_local_experts,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute weighted output.

        Computes sigmoid scores, adds correction bias for selection,
        selects top-k experts, then uses original scores for weighting.

        Args:
            hidden_states (mx.array): Input tensor of shape ``(B, L, D)``.

        Returns:
            mx.array: Output tensor of shape ``(B, L, D)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, orig_shape[-1])

        gates = self.gate(hidden_states)
        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(hidden_states.dtype)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(orig_shape)


class MiniMaxDecoderLayer(nn.Module):
    """Single MiniMax decoder layer with pre-norm and sparse MoE.

    Attributes:
        self_attn: Multi-head attention module with optional QK norm.
        block_sparse_moe: Sparse MoE block with sigmoid gating.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MoE RMSNorm.

    Example:
        >>> config = MiniMaxConfig()
        >>> layer = MiniMaxDecoderLayer(config)
        >>> out = layer(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: MiniMaxConfig):
        """Initialize MiniMax decoder layer.

        Args:
            config (MiniMaxConfig): Model configuration.
        """
        super().__init__()
        self.self_attn = MiniMaxAttention(config)
        self.block_sparse_moe = MiniMaxSparseMoeBlock(config)
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
        """Forward pass through a single MiniMax decoder layer.

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


@register_module(task_type=TaskType.BASE_MODULE, config=MiniMaxConfig, model_type="minimax")
class MiniMaxModel(EasyMLXBaseModule):
    """Base MiniMax transformer model with sparse MoE and QK normalization.

    Attributes:
        config_class: Associated configuration class (``MiniMaxConfig``).
        embed_tokens: Token embedding layer.
        layers: List of MiniMax decoder layers.
        norm: Final RMSNorm.

    Example:
        >>> config = MiniMaxConfig()
        >>> model = MiniMaxModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = MiniMaxConfig

    def __init__(self, config: MiniMaxConfig):
        """Initialize MiniMax base model.

        Args:
            config (MiniMaxConfig): Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiniMaxDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass through the MiniMax base model.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=MiniMaxConfig, model_type="minimax")
class MiniMaxForCausalLM(BaseCausalLMModule[MiniMaxModel, MiniMaxConfig]):
    """MiniMax model with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class (``MiniMaxConfig``).

    Example:
        >>> config = MiniMaxConfig()
        >>> model = MiniMaxForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = MiniMaxConfig

    def __init__(self, config: MiniMaxConfig):
        """Initialize MiniMax causal LM.

        Args:
            config (MiniMaxConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MiniMaxModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream weights: dequantize FP8, restructure MoE experts.

        Performs two transformations:
        1. Dequantizes FP8 weights using block-wise scaling (block size 128).
        2. Stacks per-expert weights from
           ``experts.E.{w1,w2,w3}.weight`` into
           ``switch_mlp.{gate_proj,down_proj,up_proj}.weight``.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Sanitized weight dictionary.
        """
        weights = super().sanitize(weights)

        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                if wk in weights:
                    weight = weights[wk]
                    weight = mx.from_fp8(weight, dtype=mx.bfloat16)
                    bs = 128
                    m, n = weight.shape
                    pad_bottom = (-m) % bs
                    pad_side = (-n) % bs
                    weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
                    weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
                    weight = (weight * scale_inv[:, None, :, None]).reshape(m + pad_bottom, n + pad_side)
                    new_weights[wk] = weight[:m, :n].astype(mx.bfloat16)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.{k}")
                            for e in range(self.config.num_local_experts)
                        ]
                        weights[f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.{k}"] = mx.stack(to_join)

        return weights


__all__ = ("MiniMaxForCausalLM", "MiniMaxModel")
