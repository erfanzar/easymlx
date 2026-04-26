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

"""Bailing MoE MLX model implementation for serving and inference.

Structure mirrors the upstream Bailing MoE architecture with fused QKV
projection, group expert selection, shared experts, and optional QK
normalization.
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
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .bailing_moe_configuration import BailingMoeConfig

CacheView = TransformerCacheView | PageCacheView


class BailingMoeAttention(nn.Module):
    """Fused QKV attention for Bailing MoE with optional QK normalization.

    Uses a single fused ``query_key_value`` projection for Q, K, V,
    with optional RMSNorm on Q and K. Supports partial rotary embeddings
    via ``partial_rotary_factor``.

    Attributes:
        use_qk_norm: Whether QK normalization is applied.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        query_key_value: Fused QKV linear projection.
        dense: Output linear projection.
        query_layernorm: Optional RMSNorm for queries.
        key_layernorm: Optional RMSNorm for keys.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = BailingMoeConfig(hidden_size=2048)
        >>> attn = BailingMoeAttention(config)
    """

    def __init__(self, config: BailingMoeConfig):
        """Initialize Bailing MoE attention.

        Args:
            config: Model configuration with attention hyperparameters.
        """
        super().__init__()
        self.use_qk_norm = config.use_qk_norm
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_dim = (
            config.rotary_dim if config.rotary_dim is not None else int(self.head_dim * config.partial_rotary_factor)
        )
        self.rope = get_rope(
            dims=rope_dim,
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
        """Run the fused QKV attention forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]

        qkv = self.query_key_value(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q.reshape(*lead, self.num_heads, self.head_dim)
        k = k.reshape(*lead, self.num_kv_heads, self.head_dim)
        v = v.reshape(*lead, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.dense(attn.reshape(*lead, -1))


class BailingMoeDenseMLP(nn.Module):
    """SiLU-gated feed-forward MLP for dense and shared expert layers.

    Uses ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Linear gate projection.
        up_proj: Linear up projection.
        down_proj: Linear down projection.
    """

    def __init__(self, config: BailingMoeConfig, intermediate_size: int | None = None):
        """Initialize the Bailing MoE dense MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimensionality.
                If ``None``, uses ``config.intermediate_size``.
        """
        super().__init__()
        intermediate = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=config.use_bias)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply SwiGLU MLP.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class BailingMoeSparseBlock(nn.Module):
    """Mixture-of-Experts block for Bailing MoE with group expert selection.

    Routes tokens to top-k experts via ``TopKRouter`` with configurable
    scoring function and optional shared experts.

    Attributes:
        router: Top-k expert routing module.
        switch_mlp: Fused SwitchGLU expert module.
        shared_experts: Optional shared expert MLP (always active).

    Example::

        >>> config = BailingMoeConfig(hidden_size=2048, num_experts=8)
        >>> moe = BailingMoeSparseBlock(config)
    """

    def __init__(self, config: BailingMoeConfig):
        """Initialize the Bailing MoE sparse block.

        Args:
            config: Model configuration with MoE hyperparameters.
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.score_function,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group,
            topk_group=config.topk_group,
            use_score_bias=config.moe_router_enable_expert_bias,
        )
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            bias=config.use_bias,
        )

        shared_dim = config.moe_shared_expert_intermediate_size or config.moe_intermediate_size
        if config.num_shared_experts > 0 and config.moe_router_enable_shared_expert:
            self.shared_experts = BailingMoeDenseMLP(
                config,
                intermediate_size=shared_dim * config.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of the same shape.
        """
        inds, scores = self.router(hidden_states)
        out = self.switch_mlp(hidden_states, inds)
        out = (out * scores[..., None]).sum(axis=-2).astype(out.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class BailingMoeDecoderLayer(nn.Module):
    """Single decoder layer for Bailing MoE.

    Uses pre-norm attention and pre-norm MLP/MoE with residual
    connections. Layers before ``first_k_dense_replace`` use dense MLP;
    later layers use the sparse MoE block.

    Attributes:
        attention: Fused QKV attention module.
        mlp: Dense MLP or MoE block depending on layer index.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.
    """

    def __init__(self, config: BailingMoeConfig, layer_idx: int):
        """Initialize the Bailing MoE decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer in the stack.
        """
        super().__init__()
        self.attention = BailingMoeAttention(config)
        self.mlp = (
            BailingMoeSparseBlock(config)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeDenseMLP(config)
        )
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
        hidden_states = residual + self.attention(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=BailingMoeConfig, model_type="bailing_moe")
class BailingMoeModel(EasyMLXBaseModule):
    """Base Bailing MoE transformer model.

    Attributes:
        config_class: The associated configuration class (``BailingMoeConfig``).
        word_embeddings: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
    """

    config_class = BailingMoeConfig

    def __init__(self, config: BailingMoeConfig):
        """Initialize the Bailing MoE base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BailingMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Alias for word_embeddings to match BaseCausalLMModule expectations."""
        return self.word_embeddings

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
            hidden_states = self.word_embeddings(input_ids)

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
        """Remap upstream checkpoint keys to EasyMLX parameter names.

        Stacks per-expert weights into fused SwitchGLU format, remaps
        ``gate`` to ``router``, optionally normalizes ``lm_head`` weights,
        and removes rotary embedding buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with expert weights stacked and
            keys remapped.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)

        if self.config.norm_head and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= self.config.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(self.config.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

                gate_key = f"{prefix}.mlp.gate.weight"
                router_key = f"{prefix}.mlp.router.weight"
                if gate_key in weights and router_key not in weights:
                    weights[router_key] = weights.pop(gate_key)

                bias_key = f"{prefix}.mlp.gate.bias"
                router_bias_key = f"{prefix}.mlp.router.score_correction_bias"
                if bias_key in weights and router_bias_key not in weights:
                    weights[router_bias_key] = weights.pop(bias_key)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=BailingMoeConfig, model_type="bailing_moe")
class BailingMoeForCausalLM(BaseCausalLMModule[BailingMoeModel, BailingMoeConfig]):
    """Bailing MoE model with a causal language modeling head.

    Wraps ``BailingMoeModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``BailingMoeConfig``).

    Example::

        >>> config = BailingMoeConfig(vocab_size=102400, hidden_size=2048)
        >>> model = BailingMoeForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = BailingMoeConfig

    def __init__(self, config: BailingMoeConfig):
        """Initialize the Bailing MoE causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=BailingMoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("BailingMoeForCausalLM", "BailingMoeModel")
