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

"""ERNIE 4.5 MoE MLX model implementation for serving and inference.

Structure mirrors the upstream ERNIE 4.5 MoE architecture with sparse expert
routing, shared experts, and configurable gate activation.
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

from .ernie4_5_moe_configuration import Ernie45MoeConfig

CacheView = TransformerCacheView | PageCacheView


class Ernie45MoeAttention(nn.Module):
    """Multi-head attention for ERNIE 4.5 MoE with traditional RoPE."""

    def __init__(self, config: Ernie45MoeConfig):
        """Initialize ERNIE 4.5 MoE attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=True,
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
        """Run the attention forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.
            mask: Optional attention mask.
            cache_view: KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output of shape ``(..., hidden_size)``.
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


class Ernie45MoeDenseMLP(nn.Module):
    """SiLU-gated feed-forward MLP for dense layers in ERNIE 4.5 MoE."""

    def __init__(self, hidden_size: int, intermediate_size: int, use_bias: bool = False):
        """Initialize a dense MLP.

        Args:
            hidden_size: Input/output dimensionality.
            intermediate_size: Intermediate layer dimensionality.
            use_bias: Whether linear layers include bias. Defaults to ``False``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run the MLP forward pass.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Ernie45MoeSparseBlock(nn.Module):
    """Mixture-of-Experts block for ERNIE 4.5 MoE.

    Routes tokens through top-k experts and optionally adds shared expert output.
    Uses configurable gate activation (softmax or sigmoid) with normalized scores.

    Attributes:
        k: Number of experts per token.
        gate: Linear gate projection.
        switch_mlp: Batched SwiGLU expert bank.
        gate_act: Gate activation function (softmax or sigmoid).
        shared_experts: Optional shared MLP.
    """

    def __init__(self, config: Ernie45MoeConfig):
        """Initialize the sparse MoE block.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If ``moe_gate_act`` is not ``"softmax"`` or ``"sigmoid"``.
        """
        super().__init__()
        self.k = config.moe_k
        moe_intermediate = config.moe_intermediate_size if config.moe_intermediate_size else config.intermediate_size

        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            moe_intermediate,
            config.moe_num_experts,
            bias=config.use_bias,
        )

        if config.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif config.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")

        if config.moe_num_shared_experts > 0:
            shared_intermediate = moe_intermediate * config.moe_num_shared_experts
            self.shared_experts = Ernie45MoeDenseMLP(config.hidden_size, shared_intermediate, config.use_bias)
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through top-k experts and aggregate.

        Args:
            hidden_states: Input of shape ``(..., hidden_size)``.

        Returns:
            MoE output of shape ``(..., hidden_size)``.
        """
        gates = self.gate(hidden_states)
        gates = self.gate_act(gates.astype(mx.float32))

        k = self.k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            y = y + self.shared_experts(hidden_states)

        return y


class Ernie45MoeDecoderLayer(nn.Module):
    """Single decoder layer for ERNIE 4.5 MoE.

    Uses MoE block or dense MLP based on layer index and MoE configuration.
    """

    def __init__(self, config: Ernie45MoeConfig, layer_idx: int):
        """Initialize a decoder layer.

        Determines whether to use MoE or dense MLP based on layer index,
        ``moe_layer_start_index``, ``moe_layer_end_index``, and ``moe_layer_interval``.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index.
        """
        super().__init__()
        self.self_attn = Ernie45MoeAttention(config)

        moe_layer_start = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        if config.moe_layer_end_index is None:
            moe_layer_end = config.num_hidden_layers - 1
        else:
            moe_layer_end = (
                max(config.moe_layer_end_index)
                if isinstance(config.moe_layer_end_index, (tuple, list))
                else config.moe_layer_end_index
            )

        use_moe = (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start
            and layer_idx <= moe_layer_end
        )

        if use_moe:
            self.mlp = Ernie45MoeSparseBlock(config)
        else:
            self.mlp = Ernie45MoeDenseMLP(config.hidden_size, config.intermediate_size, config.use_bias)

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


@register_module(task_type=TaskType.BASE_MODULE, config=Ernie45MoeConfig, model_type="ernie4_5_moe")
class Ernie45MoeModel(EasyMLXBaseModule):
    """Base ERNIE 4.5 MoE transformer model.

    Attributes:
        config_class: The associated configuration class (``Ernie45MoeConfig``).
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMS normalization.
    """

    config_class = Ernie45MoeConfig

    def __init__(self, config: Ernie45MoeConfig):
        """Initialize the ERNIE 4.5 MoE base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Ernie45MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
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

        Removes multi-token prediction (MTP) weights and score correction bias
        keys, stacks individual expert weights into ``switch_mlp`` tensors,
        and filters out rotary inv_freq keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
            "e_score_correction_bias",
        ]
        weights = {
            key: value for key, value in weights.items() if not any(pattern in key for pattern in remove_patterns)
        }

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(self.config.moe_num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Ernie45MoeConfig, model_type="ernie4_5_moe")
class Ernie45MoeForCausalLM(BaseCausalLMModule[Ernie45MoeModel, Ernie45MoeConfig]):
    """ERNIE 4.5 MoE model with a causal language modeling head."""

    config_class = Ernie45MoeConfig

    def __init__(self, config: Ernie45MoeConfig):
        """Initialize the ERNIE 4.5 MoE causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Ernie45MoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("Ernie45MoeForCausalLM", "Ernie45MoeModel")
