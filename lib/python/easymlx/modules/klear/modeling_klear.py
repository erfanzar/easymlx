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

"""Klear MLX model implementation for serving and inference.

Klear features:
  - Sigmoid-gated MoE routing with expert bias
  - Shared experts mixed via a learned coefficient
  - QK-norm attention
  - Per-layer sparse/dense MLP selection
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .klear_configuration import KlearConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class KlearAttention(nn.Module):
    """Klear attention with QK-norm.

    Applies per-head RMSNorm to queries and keys before attention
    computation. Uses GQA with RoPE.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: RMSNorm for queries.
        k_norm: RMSNorm for keys.
        rope: Rotary position embedding.
        attention_performer: Attention computation backend.

    Example:
        >>> config = KlearConfig(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> attn = KlearAttention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KlearConfig):
        """Initialize KlearAttention.

        Args:
            config: Klear model configuration.

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

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Compute QK-normed attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

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


class KlearMLP(nn.Module):
    """Dense SwiGLU MLP.

    Attributes:
        gate_proj: Gate projection (no bias).
        down_proj: Down projection (no bias).
        up_proj: Up projection (no bias).

    Example:
        >>> mlp = KlearMLP(dim=256, hidden_dim=512)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, dim: int, hidden_dim: int):
        """Initialize KlearMLP.

        Args:
            dim: Input and output dimensionality.
            hidden_dim: Intermediate dimensionality.
        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class KlearSparseMoeBlock(nn.Module):
    """Sigmoid-gated MoE with shared experts and coefficient mixing.

    Uses sigmoid activation for routing (not softmax), with an additive
    expert bias correction. The shared expert output is mixed with the
    routed output using a learned 2-way softmax coefficient.

    Attributes:
        norm_topk_prob: Whether to normalize top-k probabilities.
        num_experts: Total number of routing experts.
        top_k: Number of experts activated per token.
        gate: Router linear projection.
        experts: SwitchGLU expert bank.
        shared_experts: Shared MLP (always active).
        coefficient: 2-way softmax coefficient for mixing.
        expert_bias: Additive bias for expert routing.

    Example:
        >>> config = KlearConfig(hidden_size=256, num_experts=4,
        ...     num_experts_per_tok=2, moe_intermediate_size=512,
        ...     n_shared_experts=1)
        >>> moe = KlearSparseMoeBlock(config)
        >>> out = moe(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KlearConfig):
        """Initialize KlearSparseMoeBlock.

        Args:
            config: Klear model configuration.
        """
        super().__init__()
        self.norm_topk_prob = config.norm_topk_prob
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)
        self.shared_experts = KlearMLP(
            config.hidden_size,
            hidden_dim=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.coefficient = nn.Linear(config.hidden_size, 2)
        self.expert_bias = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute coefficient-mixed output.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``, a coefficient-
            weighted mix of routed expert output and shared expert output.
        """
        routing_weights = mx.sigmoid(self.gate(hidden_states).astype(mx.float32))
        biased_weights = routing_weights + self.expert_bias.reshape((1, 1, -1))
        k = self.top_k
        inds = mx.argpartition(-biased_weights, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(routing_weights, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(hidden_states.dtype)
        expert_out = self.experts(hidden_states, inds)
        y_experts = (expert_out * scores[..., None]).sum(axis=-2)
        coef = mx.softmax(self.coefficient(hidden_states), axis=-1, precise=True)
        shared = self.shared_experts(hidden_states)
        y = y_experts * coef[..., :1] + shared * coef[..., 1:]
        return y


class KlearDecoderLayer(nn.Module):
    """Single Klear decoder layer with conditional MoE.

    The feed-forward block is either a dense MLP (for layers in
    ``mlp_only_layers``) or a sigmoid-gated sparse MoE block, selected
    based on the layer index and ``decoder_sparse_step``.

    Attributes:
        self_attn: QK-normed attention sub-layer.
        mlp: Feed-forward sub-layer (KlearMLP or KlearSparseMoeBlock).
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = KlearConfig(hidden_size=256, num_attention_heads=8,
        ...     num_experts=4, num_experts_per_tok=2)
        >>> layer = KlearDecoderLayer(config, layer_idx=0)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KlearConfig, layer_idx: int):
        """Initialize KlearDecoderLayer.

        Args:
            config: Klear model configuration.
            layer_idx: Layer index for MoE scheduling.
        """
        super().__init__()
        self.self_attn = KlearAttention(config)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = KlearSparseMoeBlock(config)
        else:
            self.mlp = KlearMLP(config.hidden_size, config.intermediate_size)

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


@register_module(task_type=TaskType.BASE_MODULE, config=KlearConfig, model_type="Klear")
class KlearModel(EasyMLXBaseModule):
    """Base Klear transformer model with sigmoid-gated MoE (no LM head).

    Stacks ``num_hidden_layers`` KlearDecoderLayer instances with
    per-layer sparse/dense MLP selection, QK-norm attention, and
    coefficient-mixed MoE with shared experts.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding layer.
        layers: List of ``KlearDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = KlearConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8,
        ...     num_experts=4, num_experts_per_tok=2)
        >>> model = KlearModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = KlearConfig

    def __init__(self, config: KlearConfig):
        """Initialize KlearModel.

        Args:
            config: Klear model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [KlearDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
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
        """Run the Klear transformer forward pass.

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
        """Stack per-expert weights into SwitchGLU format and remove rotary buffers.

        If individual expert weights (``experts.0.gate_proj.weight``, etc.)
        are present, they are stacked into a single tensor for the SwitchGLU
        layer. Rotary embedding inverse-frequency buffers are also removed.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary ready for model loading.
        """
        sanitized: dict[str, mx.array] = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k or "rope.inv_freq" in k:
                continue
            sanitized[k] = v

        if "model.layers.0.mlp.experts.0.gate_proj.weight" in sanitized:
            num_layers = len(self.layers)
            num_experts = self.config.num_experts
            for l_idx in range(num_layers):
                prefix = f"model.layers.{l_idx}.mlp.experts"
                if f"{prefix}.0.gate_proj.weight" not in sanitized:
                    continue
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    stacked = [sanitized.pop(f"{prefix}.{e}.{name}.weight") for e in range(num_experts)]
                    sanitized[f"{prefix}.{name}.weight"] = mx.stack(stacked)

        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=KlearConfig, model_type="Klear")
class KlearForCausalLM(BaseCausalLMModule[KlearModel, KlearConfig]):
    """Klear transformer with a causal language modeling head.

    Wraps ``KlearModel`` and adds an LM head for next-token prediction.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = KlearConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8,
        ...     num_experts=4, num_experts_per_tok=2)
        >>> model = KlearForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = KlearConfig

    def __init__(self, config: KlearConfig):
        """Initialize KlearForCausalLM.

        Args:
            config: Klear model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=KlearModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("KlearForCausalLM", "KlearModel")
