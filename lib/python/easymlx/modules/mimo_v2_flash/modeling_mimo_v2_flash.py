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

"""MiMo-V2-Flash MLX model implementation for serving and inference.

MiMo-V2-Flash is a DeepSeekV3-style hybrid MoE transformer featuring
sliding window attention, group-based noaux_tc routing, shared experts,
asymmetric head dimensions, and FP8 weight dequantization at load time.
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

from .mimo_v2_flash_configuration import MiMoV2FlashConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like value to an int32 mx.array.

    Args:
        values: Input values to convert, or ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class MiMoV2FlashAttention(nn.Module):
    """MiMo-V2-Flash attention with sliding window and partial RoPE.

    Supports two modes: full attention and sliding window attention (SWA),
    each with potentially different head counts, head dimensions, and
    RoPE frequencies. Uses asymmetric Q/K and V head dimensions.

    Attributes:
        is_sliding_window: Whether this layer uses SWA.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Q/K head dimension.
        v_head_dim: V head dimension.
        scale: Attention scaling factor.

    Example:
        >>> config = MiMoV2FlashConfig(hidden_size=64, num_attention_heads=4)
        >>> attn = MiMoV2FlashAttention(config, is_sliding_window=False)
    """

    def __init__(self, config: MiMoV2FlashConfig, is_sliding_window: bool):
        """Initialize MiMo-V2-Flash attention.

        Args:
            config: Model configuration.
            is_sliding_window: Whether this layer uses sliding window attention.
        """
        super().__init__()
        self.is_sliding_window = is_sliding_window

        if is_sliding_window:
            self.num_heads = int(config.swa_num_attention_heads)
            self.num_kv_heads = int(config.swa_num_key_value_heads)
            head_dim = config.swa_head_dim
            v_head_dim = config.swa_v_head_dim
            rope_theta = config.swa_rope_theta
        else:
            self.num_heads = int(config.num_attention_heads)
            self.num_kv_heads = int(config.num_key_value_heads)
            head_dim = config.head_dim
            v_head_dim = config.v_head_dim
            rope_theta = config.rope_theta

        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * v_head_dim, config.hidden_size, bias=False)

        rope_dims = int(config.partial_rotary_factor * head_dim)
        self.rope = get_rope(
            dims=rope_dims,
            base=rope_theta,
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
        """Compute attention with partial RoPE and asymmetric head dims.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.v_head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class MiMoV2FlashMLP(nn.Module):
    """Dense SwiGLU MLP for MiMo-V2-Flash.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.

    Example:
        >>> config = MiMoV2FlashConfig(hidden_size=64, intermediate_size=128)
        >>> mlp = MiMoV2FlashMLP(config)
    """

    def __init__(self, config: MiMoV2FlashConfig, intermediate_size: int | None = None):
        """Initialize MiMo-V2-Flash MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for the intermediate dimension.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiMoV2FlashMoE(nn.Module):
    """MoE block for MiMo-V2-Flash with noaux_tc group routing and shared experts.

    Uses ``TopKRouter`` with group-based routing and optional shared
    experts that are always active alongside the routed experts.

    Attributes:
        config: Model configuration reference.
        switch_mlp: SwitchGLU module for routed experts.
        gate: Top-k router with group routing.
        shared_experts: Optional shared MLP (if ``n_shared_experts`` is set).

    Example:
        >>> config = MiMoV2FlashConfig(
        ...     hidden_size=64, n_routed_experts=8,
        ...     moe_intermediate_size=32, num_experts_per_tok=2,
        ... )
        >>> moe = MiMoV2FlashMoE(config)
    """

    def __init__(self, config: MiMoV2FlashConfig):
        """Initialize the MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor or 1.0,
            n_group=config.n_group,
            topk_group=config.topk_group,
            use_score_bias=True,
        )
        if config.n_shared_experts is not None:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MiMoV2FlashMLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and add shared expert output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Combined routed and shared expert output.
        """
        inds, scores = self.gate(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class MiMoV2FlashDecoderLayer(nn.Module):
    """Single MiMo-V2-Flash decoder layer with configurable attention and MLP.

    Supports both full attention and sliding window attention, and both
    dense MLP and MoE feed-forward, based on per-layer configuration.

    Attributes:
        is_sliding_window: Whether this layer uses SWA.
        self_attn: Attention sub-layer.
        mlp: Dense MLP or MoE block.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.

    Example:
        >>> config = MiMoV2FlashConfig(hidden_size=64)
        >>> layer = MiMoV2FlashDecoderLayer(config, is_moe=False, is_sliding_window=False)
    """

    def __init__(self, config: MiMoV2FlashConfig, is_moe: bool, is_sliding_window: bool):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            is_moe: Whether this layer uses MoE feed-forward.
            is_sliding_window: Whether this layer uses sliding window attention.
        """
        super().__init__()
        self.is_sliding_window = is_sliding_window
        self.self_attn = MiMoV2FlashAttention(config, is_sliding_window)
        self.mlp = MiMoV2FlashMoE(config) if is_moe else MiMoV2FlashMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view for incremental decoding.
            cache_metadata: Paged attention metadata.

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
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=MiMoV2FlashConfig, model_type="mimo_v2_flash")
class MiMoV2FlashModel(EasyMLXBaseModule):
    """Base MiMo-V2-Flash hybrid MoE transformer model.

    Constructs decoder layers based on ``hybrid_layer_pattern`` (SWA vs
    full attention) and ``moe_layer_freq`` (dense vs MoE).

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        layers: List of ``MiMoV2FlashDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = MiMoV2FlashConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MiMoV2FlashModel(config)
    """

    config_class = MiMoV2FlashConfig

    def __init__(self, config: MiMoV2FlashConfig):
        """Initialize the base MiMo-V2-Flash model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MiMoV2FlashDecoderLayer(
                config,
                is_moe=config.moe_layer_freq[idx] == 1,
                is_sliding_window=config.hybrid_layer_pattern[idx] == 1,
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the MiMo-V2-Flash backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged attention metadata.

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
        """Sanitize upstream weights for MiMo-V2-Flash.

        Stacks per-expert weights into ``switch_mlp`` format, remaps the
        router score correction bias, and removes MTP layer weights.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        n_routed = getattr(self.config, "n_routed_experts", None)

        if n_routed is not None:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(n_routed)]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

                bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
                router_bias_key = f"{prefix}.mlp.gate.score_correction_bias"
                if bias_key in weights and router_bias_key not in weights:
                    weights[router_bias_key] = weights.pop(bias_key)

        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}


@register_module(task_type=TaskType.CAUSAL_LM, config=MiMoV2FlashConfig, model_type="mimo_v2_flash")
class MiMoV2FlashForCausalLM(BaseCausalLMModule[MiMoV2FlashModel, MiMoV2FlashConfig]):
    """MiMo-V2-Flash causal language model with an LM head.

    Wraps ``MiMoV2FlashModel`` with a language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = MiMoV2FlashConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MiMoV2FlashForCausalLM(config)
    """

    config_class = MiMoV2FlashConfig

    def __init__(self, config: MiMoV2FlashConfig):
        """Initialize the MiMo-V2-Flash causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MiMoV2FlashModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.base_model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("MiMoV2FlashForCausalLM", "MiMoV2FlashModel")
