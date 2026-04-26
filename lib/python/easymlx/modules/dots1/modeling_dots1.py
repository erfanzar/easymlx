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

"""Dots1 MLX model implementation for serving and inference.

This module provides the Dots1 MoE architecture on MLX, featuring
QK-norm attention, grouped expert selection with sigmoid scoring,
score correction bias routing, and shared experts.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .dots1_configuration import Dots1Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an optional array-like to an ``mx.array`` of int32.

    Args:
        values: Array-like values to convert, or ``None``.

    Returns:
        An ``mx.array`` with dtype int32, or ``None`` if input is ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class Dots1Attention(nn.Module):
    """Multi-head attention for Dots1 with QK-norm and RoPE.

    Applies RMSNorm to queries and keys before computing attention, which
    stabilizes training and improves convergence for large models.

    Attributes:
        hidden_size: Hidden dimension size.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads for GQA.
        head_dim: Per-head dimension.
        scale: Attention scaling factor.
        q_norm: RMSNorm applied to query projections.
        k_norm: RMSNorm applied to key projections.

    Example::

        >>> attn = Dots1Attention(config)
        >>> output = attn(hidden_states, mask=mask)
    """

    def __init__(self, config: Dots1Config):
        """Initialize Dots1 attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the QK-normed attention forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]

        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

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


class Dots1MLP(nn.Module):
    """SiLU-gated feed-forward MLP for Dots1.

    Implements ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``.
    """

    def __init__(self, config: Dots1Config, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimension, or ``None``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=config.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Dots1MoE(nn.Module):
    """Mixture-of-Experts block for Dots1.

    Uses sigmoid scoring with score correction bias and group-based expert
    selection. Optionally adds shared expert output.

    Attributes:
        config: Model configuration.
        router: Top-k router with score bias.
        switch_mlp: Batched SwiGLU expert bank.
        shared_experts: Optional shared MLP.
    """

    def __init__(self, config: Dots1Config):
        """Initialize the MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="sigmoid",
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group,
            topk_group=config.topk_group,
            use_score_bias=True,
        )
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)

        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Dots1MLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and aggregate outputs.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            MoE output of shape ``(..., hidden_size)``.
        """
        inds, scores = self.router(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            y = y + self.shared_experts(hidden_states)
        return y


class Dots1DecoderLayer(nn.Module):
    """Single Dots1 decoder layer.

    Pre-norm residual with QK-normed attention and MoE or dense MLP.
    Layers before ``first_k_dense_replace`` use dense MLP; the rest use MoE.
    """

    def __init__(self, config: Dots1Config, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index.
        """
        super().__init__()
        self.self_attn = Dots1Attention(config)
        self.mlp = Dots1MoE(config) if layer_idx >= config.first_k_dense_replace else Dots1MLP(config)
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
            Output of shape ``(batch, seq_len, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=Dots1Config, model_type="dots1")
class Dots1Model(EasyMLXBaseModule):
    """Base Dots1 transformer model with MoE.

    Decoder-only transformer with QK-normed attention, grouped expert
    selection using sigmoid scoring with score correction bias.

    Attributes:
        config_class: The configuration class (``Dots1Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``Dots1DecoderLayer`` instances.
        norm: Final RMS normalization.

    Example::

        >>> config = Dots1Config()
        >>> model = Dots1Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = Dots1Config

    def __init__(self, config: Dots1Config):
        """Initialize the Dots1 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Dots1DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
            attention_mask: Optional mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings instead of ``input_ids``.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length mismatches layer count.
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
        """Stack per-expert weights and remap router keys.

        Transforms individual expert ``gate_proj``, ``up_proj``, ``down_proj``
        weights into stacked ``switch_mlp`` tensors. Remaps ``mlp.gate.weight``
        to ``mlp.router.weight`` and ``e_score_correction_bias`` to
        ``score_correction_bias``. Filters out rotary inv_freq keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        n_routed = self.config.n_routed_experts

        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx < self.config.first_k_dense_replace:
                continue
            prefix = f"model.layers.{layer_idx}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(n_routed)]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            gate_key = f"{prefix}.mlp.gate.weight"
            router_key = f"{prefix}.mlp.router.weight"
            if gate_key in weights and router_key not in weights:
                weights[router_key] = weights.pop(gate_key)
            bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
            router_bias_key = f"{prefix}.mlp.router.score_correction_bias"
            if bias_key in weights and router_bias_key not in weights:
                weights[router_bias_key] = weights.pop(bias_key)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Dots1Config, model_type="dots1")
class Dots1ForCausalLM(BaseCausalLMModule[Dots1Model, Dots1Config]):
    """Dots1 causal language model.

    Wraps ``Dots1Model`` with a language modeling head.

    Example::

        >>> model = Dots1ForCausalLM(Dots1Config())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = Dots1Config

    def __init__(self, config: Dots1Config):
        """Initialize the Dots1 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Dots1Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Apply base model sanitization followed by CausalLM sanitization.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        weights = self.model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("Dots1ForCausalLM", "Dots1Model")
