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

"""GraniteMoE MLX implementation (serving/inference only).

Structure mirrors EasyDeL's granite with MoE:
  GraniteMoeConfig -> GraniteMoeAttention -> GraniteMoeMoE
  -> GraniteMoeDecoderLayer -> GraniteMoeModel -> GraniteMoeForCausalLM

GraniteMoE extends the Granite architecture with Mixture-of-Experts
using SwitchGLU, plus custom scaling multipliers.
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

from .granitemoe_configuration import GraniteMoeConfig

CacheView = TransformerCacheView | PageCacheView


class GraniteMoeAttention(nn.Module):
    """Multi-head attention with Granite's attention_multiplier scaling.

    Uses ``attention_multiplier`` directly as the attention scale
    factor (not divided by ``sqrt(head_dim)`` further).

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scale (``attention_multiplier``).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        rope: Rotary position embedding.
        attention_performer: Attention backend.

    Example::

        >>> attn = GraniteMoeAttention(GraniteMoeConfig(hidden_size=64, num_attention_heads=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeConfig):
        """Initialize GraniteMoE attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = config.attention_multiplier

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

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
        """Compute attention.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Output of shape ``[batch, seq_len, hidden_size]``.
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


class GraniteMoeTopKGating(nn.Module):
    """Top-K expert gating for GraniteMoE.

    Routes each token to the top-K experts based on softmax-normalized
    gating logits. The gating layer is a bias-free linear projection.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        layer: Linear gating projection.

    Args:
        input_size: Input feature dimensionality.
        num_experts: Total number of experts.
        top_k: Number of experts per token.

    Example::

        >>> gate = GraniteMoeTopKGating(64, num_experts=8, top_k=2)
        >>> indices, weights = gate(mx.zeros((1, 8, 64)))
    """

    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """Initialize top-K gating.

        Args:
            input_size: Input feature dimensionality.
            num_experts: Total number of experts.
            top_k: Number of experts per token.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def __call__(self, hidden_states: mx.array):
        """Compute top-K expert selection.

        Args:
            hidden_states: Input of shape ``[..., input_size]``.

        Returns:
            Tuple of (expert_indices, gate_weights) where indices
            has shape ``[..., top_k]`` and weights are softmax-normalized.
        """
        logits = self.layer(hidden_states)
        top_k_idx = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        top_k_logits = mx.take_along_axis(logits, top_k_idx, axis=-1)
        top_k_gates = mx.softmax(top_k_logits.astype(mx.float32), axis=-1)
        return top_k_idx, top_k_gates


class GraniteMoeMoE(nn.Module):
    """Mixture-of-Experts block using SwitchGLU for GraniteMoE.

    Routes each token to top-K experts via ``GraniteMoeTopKGating``,
    processes through ``SwitchGLU`` expert MLPs, and combines outputs
    using gating weights.

    Attributes:
        switch_mlp: SwitchGLU expert MLP block.
        router: Top-K gating module.

    Example::

        >>> moe = GraniteMoeMoE(GraniteMoeConfig(hidden_size=64, num_local_experts=4))
        >>> out = moe(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeConfig):
        """Initialize GraniteMoE MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, config.num_local_experts)
        self.router = GraniteMoeTopKGating(
            input_size=config.hidden_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to experts and combine outputs.

        Args:
            x: Input of shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Weighted expert output of the same shape.
        """
        token_ids, gates = self.router(x)
        y = self.switch_mlp(x, token_ids)
        return (y * gates[..., None]).sum(axis=-2).astype(y.dtype)


class GraniteMoeDecoderLayer(nn.Module):
    """Single GraniteMoE decoder layer with MoE and residual_multiplier.

    Uses sparse MoE instead of a dense MLP, with ``residual_multiplier``
    scaling both attention and MoE outputs before residual addition.

    Attributes:
        self_attn: GraniteMoE attention.
        block_sparse_moe: MoE block with SwitchGLU experts.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MoE.
        residual_multiplier: Scale factor for sub-layer outputs.

    Example::

        >>> layer = GraniteMoeDecoderLayer(GraniteMoeConfig(hidden_size=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GraniteMoeConfig):
        """Initialize GraniteMoE decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.self_attn = GraniteMoeAttention(config)
        self.block_sparse_moe = GraniteMoeMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with MoE and residual_multiplier scaling.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after attention and MoE.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = residual + attn_out * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.block_sparse_moe(hidden_states) * self.residual_multiplier
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=GraniteMoeConfig, model_type="granitemoe")
class GraniteMoeModel(EasyMLXBaseModule):
    """Base GraniteMoE transformer with MoE and embedding_multiplier.

    Each decoder layer uses sparse MoE routing with SwitchGLU experts.
    Token embeddings are scaled by ``embedding_multiplier``.

    Attributes:
        embed_tokens: Token embedding layer.
        embedding_multiplier: Scale factor for embeddings.
        layers: Stack of ``GraniteMoeDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example::

        >>> model = GraniteMoeModel(GraniteMoeConfig(vocab_size=256, hidden_size=64))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GraniteMoeConfig

    def __init__(self, config: GraniteMoeConfig):
        """Initialize GraniteMoE base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_multiplier = config.embedding_multiplier
        self.layers = [GraniteMoeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Forward pass with MoE layers and embedding scaling.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata.

        Returns:
            Normalized hidden states.

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
            hidden_states = self.embed_tokens(input_ids) * self.embedding_multiplier

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
        """Handle MoE weight transformations for SwitchGLU format.

        Performs three transformations:

        1. Splits fused ``input_linear`` weights into separate
           ``gate_proj`` and ``up_proj`` for the SwitchGLU format.
        2. Stacks individual per-expert weights (``experts.N.gate_proj``)
           into batched tensors (``switch_mlp.gate_proj``).
        3. Removes tied ``lm_head`` and ``rotary_emb.inv_freq`` keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary in SwitchGLU format.
        """
        if "model.layers.0.block_sparse_moe.input_linear.weight" in weights:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}.block_sparse_moe"
                key = f"{prefix}.input_linear.weight"
                if key in weights:
                    value = weights.pop(key)
                    gate_proj, up_proj = mx.split(value, 2, axis=1)
                    weights[key.replace("input_linear", "switch_mlp.gate_proj")] = gate_proj
                    weights[key.replace("input_linear", "switch_mlp.up_proj")] = up_proj
                out_key = f"{prefix}.output_linear.weight"
                if out_key in weights:
                    weights[out_key.replace("output_linear", "switch_mlp.down_proj")] = weights.pop(out_key)

        # Stack individual expert weights if present
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                first_key = f"{prefix}.switch_mlp.experts.0.{m}.weight"
                if first_key in weights:
                    to_join = []
                    for e in range(self.config.num_local_experts):
                        ek = f"{prefix}.switch_mlp.experts.{e}.{m}.weight"
                        if ek in weights:
                            to_join.append(weights.pop(ek))
                    if to_join:
                        weights[f"{prefix}.switch_mlp.{m}.weight"] = mx.stack(to_join)

        if getattr(self.config, "tie_word_embeddings", True):
            weights.pop("lm_head.weight", None)

        # Filter rotary inv_freq
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        return weights


@register_module(task_type=TaskType.CAUSAL_LM, config=GraniteMoeConfig, model_type="granitemoe")
class GraniteMoeForCausalLM(BaseCausalLMModule[GraniteMoeModel, GraniteMoeConfig]):
    """GraniteMoE causal language model with logits_scaling.

    Wraps ``GraniteMoeModel`` and divides output logits by
    ``logits_scaling``.

    Attributes:
        config_class: ``GraniteMoeConfig``.

    Example::

        >>> model = GraniteMoeForCausalLM(GraniteMoeConfig(vocab_size=256, hidden_size=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GraniteMoeConfig

    def __init__(self, config: GraniteMoeConfig):
        super().__init__(
            config=config,
            base_model_class=GraniteMoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project to logits and divide by ``logits_scaling``.

        Args:
            hidden_states: Transformer output.

        Returns:
            Scaled logits of shape ``[batch, seq_len, vocab_size]``.
        """
        logits = super().compute_lm_logits(hidden_states)
        return logits / self.config.logits_scaling


__all__ = ("GraniteMoeForCausalLM", "GraniteMoeModel")
