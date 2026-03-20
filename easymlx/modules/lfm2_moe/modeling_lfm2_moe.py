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

"""LFM2-MoE MLX model implementation for serving and inference.

LFM2-MoE extends LFM2 with Mixture-of-Experts feed-forward layers.
Layers below ``num_dense_layers`` use dense MLPs; layers at or above
use a sparse MoE block with SwitchGLU experts and softmax gating.
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

from .lfm2_moe_configuration import Lfm2MoeConfig

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


class Lfm2MoeAttention(nn.Module):
    """LFM2-MoE multi-head attention with QK layernorm and RoPE.

    Identical architecture to ``Lfm2Attention``, applying per-head RMSNorm
    to queries and keys before computing RoPE-based attention.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.

    Example:
        >>> config = Lfm2MoeConfig(hidden_size=64, num_attention_heads=4)
        >>> attn = Lfm2MoeAttention(config)
    """

    def __init__(self, config: Lfm2MoeConfig):
        """Initialize LFM2-MoE attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = get_rope(
            dims=self.head_dim,
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
        """Compute attention with QK layernorm and RoPE.

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
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        queries = self.q_layernorm(queries)
        keys = self.k_layernorm(keys)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.out_proj(attn.reshape(*lead, -1))


class Lfm2MoeShortConv(nn.Module):
    """LFM2-MoE gated depthwise short convolution layer.

    Functionally identical to ``Lfm2ShortConv``, projecting to three
    gates (B, C, x), applying depthwise conv1d, and gating the output.

    Attributes:
        hidden_size: Model hidden dimension.
        L_cache: Convolution kernel size.

    Example:
        >>> config = Lfm2MoeConfig(hidden_size=64, conv_L_cache=4)
        >>> conv = Lfm2MoeShortConv(config)
    """

    def __init__(self, config: Lfm2MoeConfig):
        """Initialize LFM2-MoE short convolution.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Apply gated depthwise short convolution.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Unused; accepted for API compatibility.
            cache_view: Unused; accepted for API compatibility.
            cache_metadata: Unused; accepted for API compatibility.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        BCx = self.in_proj(hidden_states)
        B_gate, C_gate, x = mx.split(BCx, 3, axis=-1)
        Bx = B_gate * x

        Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])
        conv_out = self.conv(Bx)
        y = C_gate * conv_out
        return self.out_proj(y)


class Lfm2MoeMLP(nn.Module):
    """Dense SwiGLU MLP for LFM2-MoE, used in the first ``num_dense_layers``.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection linear layer.
        up_proj: Up projection linear layer.
        down_proj: Down projection linear layer.

    Example:
        >>> config = Lfm2MoeConfig(hidden_size=64, intermediate_size=128)
        >>> mlp = Lfm2MoeMLP(config)
    """

    def __init__(self, config: Lfm2MoeConfig, intermediate_size: int | None = None):
        """Initialize the dense MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for the intermediate dimension.
                If ``None``, uses ``config.intermediate_size``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute the SwiGLU feed-forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Lfm2MoeSparseMoeBlock(nn.Module):
    """Sparse Mixture-of-Experts block with softmax gating and SwitchGLU experts.

    Routes each token to the top-k experts via softmax gating, optionally
    normalizing the routing probabilities and adding an expert bias.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token.
        norm_topk_prob: Whether to normalize top-k probabilities.
        use_expert_bias: Whether to add a learned bias to expert scores.
        gate: Linear gating projection.
        switch_mlp: SwitchGLU module containing all expert weights.

    Example:
        >>> config = Lfm2MoeConfig(
        ...     hidden_size=64, num_experts=4, num_experts_per_tok=2,
        ...     moe_intermediate_size=32,
        ... )
        >>> moe = Lfm2MoeSparseMoeBlock(config)
    """

    def __init__(self, config: Lfm2MoeConfig):
        """Initialize the sparse MoE block.

        Args:
            config: Model configuration with MoE hyperparameters.
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.use_expert_bias = config.use_expert_bias

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)
        if self.use_expert_bias:
            self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to top-k experts and combine outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Weighted expert output of shape ``(..., hidden_size)``.
        """
        gates = self.gate(hidden_states).astype(mx.float32)
        gates = mx.softmax(gates, axis=-1)

        if self.use_expert_bias:
            gates = gates + self.expert_bias

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(hidden_states.dtype)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class Lfm2MoeDecoderLayer(nn.Module):
    """Single LFM2-MoE decoder layer with attention/conv and dense/MoE MLP.

    Selects between attention and convolution based on ``full_attn_idxs``,
    and between dense MLP and sparse MoE based on ``num_dense_layers``.

    Attributes:
        is_attention_layer: Whether this layer uses attention.
        feed_forward: Dense MLP or sparse MoE block.
        operator_norm: Pre-operator RMSNorm.
        ffn_norm: Pre-FFN RMSNorm.

    Example:
        >>> config = Lfm2MoeConfig(hidden_size=64, num_hidden_layers=2)
        >>> layer = Lfm2MoeDecoderLayer(config, layer_idx=0)
    """

    def __init__(self, config: Lfm2MoeConfig, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer, used to select operator type and
                dense vs. MoE feed-forward.
        """
        super().__init__()
        self.is_attention_layer = layer_idx in config.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Lfm2MoeAttention(config)
        else:
            self.conv = Lfm2MoeShortConv(config)

        self.feed_forward = (
            Lfm2MoeMLP(config, intermediate_size=config.intermediate_size)
            if layer_idx < config.num_dense_layers
            else Lfm2MoeSparseMoeBlock(config)
        )

        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

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
        normed = self.operator_norm(hidden_states)

        if self.is_attention_layer:
            hidden_states = residual + self.self_attn(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        else:
            hidden_states = residual + self.conv(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )

        residual = hidden_states
        hidden_states = residual + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=Lfm2MoeConfig, model_type="lfm2_moe")
class Lfm2MoeModel(EasyMLXBaseModule):
    """Base LFM2-MoE hybrid conv-attention transformer with MoE feed-forward.

    Embeds tokens, passes through decoder layers (with conv/attention and
    dense/MoE selection), and applies final normalization.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        layers: List of ``Lfm2MoeDecoderLayer`` modules.
        embedding_norm: Final RMSNorm.

    Example:
        >>> config = Lfm2MoeConfig(
        ...     vocab_size=1000, hidden_size=64, num_hidden_layers=2,
        ...     num_experts=4, moe_intermediate_size=32,
        ... )
        >>> model = Lfm2MoeModel(config)
    """

    config_class = Lfm2MoeConfig

    def __init__(self, config: Lfm2MoeConfig):
        """Initialize the base LFM2-MoE model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Lfm2MoeDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the LFM2-MoE backbone.

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

        return self.embedding_norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream weights for LFM2-MoE.

        Transposes conv1d weights, remaps ``w1/w2/w3`` to
        ``gate_proj/down_proj/up_proj``, and stacks per-expert weights
        into the ``switch_mlp`` format.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        sanitized = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)
            # Remap w1/w2/w3 to gate_proj/down_proj/up_proj
            replacements = {
                "w1.weight": "gate_proj.weight",
                "w2.weight": "down_proj.weight",
                "w3.weight": "up_proj.weight",
            }
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)
            sanitized[name] = param

        # Stack MoE expert weights into switch_mlp
        num_experts = getattr(self.config, "num_experts", None)
        if num_experts is not None:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"
                for n in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.feed_forward.experts.0.{n}.{k}" in sanitized:
                            to_join = [
                                sanitized.pop(f"{prefix}.feed_forward.experts.{e}.{n}.{k}") for e in range(num_experts)
                            ]
                            sanitized[f"{prefix}.feed_forward.switch_mlp.{n}.{k}"] = mx.stack(to_join)
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=Lfm2MoeConfig, model_type="lfm2_moe")
class Lfm2MoeForCausalLM(BaseCausalLMModule[Lfm2MoeModel, Lfm2MoeConfig]):
    """LFM2-MoE causal language model with an LM head.

    Wraps ``Lfm2MoeModel`` with a language modeling head for next-token
    prediction.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = Lfm2MoeConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = Lfm2MoeForCausalLM(config)
    """

    config_class = Lfm2MoeConfig

    def __init__(self, config: Lfm2MoeConfig):
        """Initialize the LFM2-MoE causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Lfm2MoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
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


__all__ = ("Lfm2MoeForCausalLM", "Lfm2MoeModel")
