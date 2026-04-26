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

"""ExaoneMoE MLX implementation (serving/inference only).

Structure mirrors the upstream ExaoneMoE architecture:
  ExaoneMoeConfig -> ExaoneMoeAttention -> ExaoneMoeMLP -> ExaoneMoeMoE
  -> ExaoneMoeDecoderLayer -> ExaoneMoeModel -> ExaoneMoeForCausalLM

ExaoneMoE uses per-layer MoE/dense selection, sliding window attention,
Q/K RMSNorm, shared experts, and group-based expert routing.
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

from .exaone_moe_configuration import ExaoneMoeConfig

CacheView = TransformerCacheView | PageCacheView


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Select experts using group-based scoring with sigmoid.

    Applies sigmoid to gate logits, adds score correction bias, optionally
    filters by expert groups (keeping only ``topk_group`` groups), then
    selects the top-k experts per token.

    Args:
        gates: Gate logits of shape ``(..., num_experts)``.
        e_score_correction_bias: Per-expert correction bias of shape
            ``(num_experts,)``.
        top_k: Number of experts to select per token.
        n_group: Number of expert groups for group filtering.
        topk_group: Number of groups to keep.
        routed_scaling_factor: Scaling factor for final scores.
        norm_topk_prob: Whether to normalize selected probabilities.

    Returns:
        A tuple of ``(indices, scores)`` where indices has shape
        ``(..., top_k)`` and scores has shape ``(..., top_k)``.
    """
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)

    scores = scores * routed_scaling_factor
    return inds, scores


class ExaoneMoeGate(nn.Module):
    """Expert routing gate with group selection for ExaoneMoE.

    Stores a weight matrix and score correction bias, and delegates to
    ``_group_expert_select`` for group-based expert selection.

    Attributes:
        top_k: Number of experts per token.
        n_routed_experts: Total number of routed experts.
        weight: Gate weight matrix of shape ``(num_experts, hidden_size)``.
        e_score_correction_bias: Per-expert correction bias.
    """

    def __init__(self, config: ExaoneMoeConfig):
        """Initialize the gate.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute expert indices and scores for the input.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            A tuple of ``(indices, scores)`` for selected experts.
        """
        return _group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class ExaoneMoeMLP(nn.Module):
    """SwiGLU feed-forward MLP for ExaoneMoE dense layers and shared experts."""

    def __init__(self, config: ExaoneMoeConfig, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimension, or ``None``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass.

        Args:
            x: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ExaoneMoeMoE(nn.Module):
    """Mixture-of-Experts block with shared experts for ExaoneMoE.

    Uses group-based expert routing via ``ExaoneMoeGate`` and a ``SwitchGLU``
    expert bank. Optionally adds shared expert output.

    Attributes:
        switch_mlp: Batched SwiGLU expert bank.
        gate: Expert routing gate.
        shared_experts: Optional shared MLP.
    """

    def __init__(self, config: ExaoneMoeConfig):
        """Initialize the MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        )
        self.gate = ExaoneMoeGate(config)
        self.shared_experts = (
            ExaoneMoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            )
            if config.num_shared_experts is not None and config.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens through experts and aggregate outputs.

        Args:
            x: Input of shape ``(..., hidden_size)``.

        Returns:
            MoE output of shape ``(..., hidden_size)``.
        """
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class ExaoneMoeAttention(nn.Module):
    """Multi-head attention with Q/K RMSNorm and optional RoPE for ExaoneMoE.

    RoPE is applied only to sliding window attention layers (and all layers
    when no sliding window layers exist). Global attention layers without
    RoPE rely on the learned position information from other layers.

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimension.
        is_sliding_window: Whether this layer uses sliding window attention.
        use_rope: Whether RoPE is applied to this layer.
        q_norm: Q projection RMSNorm.
        k_norm: K projection RMSNorm.
    """

    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        """Initialize ExaoneMoE attention.

        RoPE is used for sliding window layers, and also for all layers when
        no sliding window layers are present in the architecture.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index (determines sliding window behavior).
        """
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.is_sliding_window = config.layer_types[layer_idx] == "sliding_attention"
        self.apply_rope_all_layers = "sliding_attention" not in config.layer_types
        self.use_rope = self.is_sliding_window or self.apply_rope_all_layers

        if self.use_rope:
            self.rope = get_rope(
                dims=self.head_dim,
                base=config.rope_theta,
                traditional=False,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.rope = None

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
        """Run the attention forward pass with Q/K RMSNorm.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        lead = hidden_states.shape[:-1]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        q = self.q_norm(queries.reshape(*lead, self.num_heads, -1))
        k = self.k_norm(keys.reshape(*lead, self.num_kv_heads, -1))
        v = values.reshape(*lead, self.num_kv_heads, -1)

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


class ExaoneMoeDecoderLayer(nn.Module):
    """Single ExaoneMoE decoder layer with per-layer MoE/dense selection.

    Each layer decides between MoE and dense MLP based on ``is_moe_layer[layer_idx]``,
    and between sliding window and global attention based on ``layer_types[layer_idx]``.

    Attributes:
        self_attn: Attention module.
        mlp: MoE or dense MLP.
        is_sliding_window: Whether this layer uses sliding window attention.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.
    """

    def __init__(self, config: ExaoneMoeConfig, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index.
        """
        super().__init__()
        self.self_attn = ExaoneMoeAttention(config, layer_idx)
        self.mlp = ExaoneMoeMoE(config) if config.is_moe_layer[layer_idx] else ExaoneMoeMLP(config)
        self.is_sliding_window = self.self_attn.is_sliding_window

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
            mask: Optional attention mask (sliding window or global).
            cache_view: KV cache view.
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


@register_module(task_type=TaskType.BASE_MODULE, config=ExaoneMoeConfig, model_type="exaone_moe")
class ExaoneMoeModel(EasyMLXBaseModule):
    """Base ExaoneMoE transformer model with sliding window attention.

    Combines per-layer MoE/dense selection, sliding window attention with
    Q/K RMSNorm, shared experts, and group-based expert routing. Builds
    separate masks for sliding window and global attention layers.

    Attributes:
        config_class: The configuration class (``ExaoneMoeConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``ExaoneMoeDecoderLayer`` instances.
        norm: Final RMS normalization.
        swa_idx: Index of first sliding window layer, or ``None``.
        ga_idx: Index of first global attention layer, or ``None``.
        window_size: Sliding window size.

    Example::

        >>> config = ExaoneMoeConfig()
        >>> model = ExaoneMoeModel(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = ExaoneMoeConfig

    def __init__(self, config: ExaoneMoeConfig):
        """Initialize the ExaoneMoE base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [ExaoneMoeDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.swa_idx = None
        self.ga_idx = None
        for i, layer in enumerate(self.layers):
            if layer.is_sliding_window and self.swa_idx is None:
                self.swa_idx = i
            if not layer.is_sliding_window and self.ga_idx is None:
                self.ga_idx = i

        self.window_size = config.sliding_window

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the transformer forward pass with sliding window support.

        Builds separate masks for sliding window and global attention layers.
        Sliding window layers receive restricted cache metadata.

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

        global_mask: mx.array | str | None = None
        swa_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                global_mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if self.swa_idx is not None:
                    swa_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.window_size,
                    )

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.is_sliding_window:
                layer_metadata = cache_metadata.with_sliding_window(self.window_size)
            layer_mask = swa_mask if layer.is_sliding_window else global_mask
            hidden_states = layer(
                hidden_states,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Handle expert stacking and e_score_correction_bias remapping.

        Performs the following transformations:

        1. Removes multi-token prediction (MTP) weights.
        2. Remaps ``e_score_correction_bias`` to gate sub-module.
        3. Stacks individual expert weights into ``switch_mlp`` tensors.
        4. Removes ``lm_head.weight`` when embeddings are tied.
        5. Filters out rotary inv_freq keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """

        weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}

        for layer_idx in range(self.config.num_hidden_layers):
            if not self.config.is_moe_layer[layer_idx]:
                continue

            prefix = f"model.layers.{layer_idx}"

            bias_key = f"{prefix}.mlp.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = weights.pop(bias_key)

            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    last_key = f"{prefix}.mlp.experts.{self.config.num_experts - 1}.{m}.{k}"
                    if first_key in weights and last_key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(self.config.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        if getattr(self.config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)

        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        return weights


@register_module(task_type=TaskType.CAUSAL_LM, config=ExaoneMoeConfig, model_type="exaone_moe")
class ExaoneMoeForCausalLM(BaseCausalLMModule[ExaoneMoeModel, ExaoneMoeConfig]):
    """ExaoneMoE causal language model.

    Example::

        >>> model = ExaoneMoeForCausalLM(ExaoneMoeConfig())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = ExaoneMoeConfig

    def __init__(self, config: ExaoneMoeConfig):
        """Initialize the ExaoneMoE causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=ExaoneMoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("ExaoneMoeForCausalLM", "ExaoneMoeModel")
