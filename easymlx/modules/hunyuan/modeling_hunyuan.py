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

"""Hunyuan MLX implementation (serving/inference only).

Structure:
  HunyuanConfig -> HunyuanDynamicNTKAlphaRoPE -> HunyuanAttention
  -> HunyuanMLP -> HunyuanMoeBlock -> HunyuanDecoderLayer
  -> HunyuanModel -> HunyuanForCausalLM

Key features:
  - Dynamic NTK-Alpha RoPE scaling
  - QK normalization (per-head)
  - Cross-Layer Attention (CLA) for KV sharing
  - Mixed MLP/MoE layers with shared experts
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
from easymlx.modules._base import BaseCausalLMModule

from .hunyuan_configuration import HunyuanConfig

CacheView = TransformerCacheView | PageCacheView


def _int_or_list(arg, idx):
    """Index into a value that may be a scalar or a list.

    Args:
        arg: A scalar value or a list of values.
        idx: Index to use if *arg* is a list.

    Returns:
        ``arg[idx]`` if *arg* is a list, otherwise *arg* itself.
    """
    if isinstance(arg, list):
        return arg[idx]
    return arg


class HunyuanDynamicNTKAlphaRoPE(nn.Module):
    """Dynamic NTK-Alpha scaled rotary positional embedding.

    Applies a frequency-domain scaling to the RoPE base frequency using the
    NTK-Alpha method: ``base' = base * alpha^(dims / (dims - 2))``.
    This allows extending the effective context length without fine-tuning.

    Attributes:
        dims: Number of rotary embedding dimensions.

    Example:
        >>> rope = HunyuanDynamicNTKAlphaRoPE(dims=64, base=10000, scaling_alpha=2.0)
        >>> x = mx.zeros((1, 8, 10, 64))
        >>> out = rope(x, offset=0)
        >>> out.shape
        [1, 8, 10, 64]
    """

    def __init__(self, dims: int, base: float = 10000, scaling_alpha: float = 1.0):
        """Initialize HunyuanDynamicNTKAlphaRoPE.

        Args:
            dims: Number of rotary embedding dimensions (must be even).
            base: Base frequency for RoPE.
            scaling_alpha: NTK-Alpha scaling factor. A value of 1.0 means
                no scaling.
        """
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        """Apply rotary positional embedding to input tensor.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, dims)``.
            offset: Positional offset for autoregressive decoding.

        Returns:
            Tensor with rotary embeddings applied, same shape as *x*.
        """
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class HunyuanAttention(nn.Module):
    """Attention with optional QK normalization and Cross-Layer Attention (CLA).

    When CLA is enabled, certain layers omit their own KV projections and
    instead reuse the KV states computed by an earlier layer (controlled
    by ``cla_share_factor``). QK normalization applies per-head RMSNorm to
    queries and keys after RoPE, stabilizing training at scale.

    Attributes:
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        use_qk_norm: Whether QK normalization is applied.
        has_kv_proj: Whether this layer has its own K/V projections.
        q_proj: Query projection.
        k_proj: Key projection (absent when ``kv_proj=False``).
        v_proj: Value projection (absent when ``kv_proj=False``).
        o_proj: Output projection.
        rope: Dynamic NTK-Alpha RoPE.
        attention_performer: Attention computation backend.

    Example:
        >>> config = HunyuanConfig(hidden_size=320, num_attention_heads=8,
        ...     num_key_value_heads=4, use_qk_norm=True)
        >>> attn = HunyuanAttention(config, kv_proj=True)
        >>> h = mx.zeros((1, 10, 320))
        >>> out, kv = attn(h)
        >>> out.shape
        [1, 10, 320]
    """

    def __init__(self, config: HunyuanConfig, kv_proj: bool = True):
        """Initialize HunyuanAttention.

        Args:
            config: Hunyuan model configuration.
            kv_proj: Whether to create K/V projection layers. Set to
                ``False`` for CLA layers that share KV from a prior layer.
        """
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        head_dim = dim // self.n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.use_qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=config.attention_bias)
        if kv_proj:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=config.attention_bias)
        self.has_kv_proj = kv_proj

        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, config.rms_norm_eps)

        scaling_alpha = 1.0
        if config.rope_scaling and "alpha" in config.rope_scaling:
            scaling_alpha = config.rope_scaling["alpha"]

        self.rope = HunyuanDynamicNTKAlphaRoPE(
            head_dim,
            base=config.rope_theta,
            scaling_alpha=scaling_alpha,
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
        kv_states: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """Compute attention forward pass with optional CLA sharing.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.
            kv_states: Pre-computed ``(keys, values)`` from a CLA donor
                layer. If ``None`` and this layer has KV projections, they
                are computed from ``hidden_states``.

        Returns:
            A tuple of ``(output, kv_states)`` where *output* has shape
            ``(B, L, hidden_size)`` and *kv_states* is ``(keys, values)``
            for potential CLA sharing with subsequent layers.
        """
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        if kv_states is None:
            keys, values = self.k_proj(hidden_states), self.v_proj(hidden_states)
            kv_states = (keys, values)
        else:
            keys, values = kv_states

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache_view.offset if cache_view is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache_view is not None:
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)

        output = self.attention_performer.forward(
            queries,
            keys,
            values,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv_states


class HunyuanMLP(nn.Module):
    """SwiGLU feed-forward network for Hunyuan.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection (no bias).
        down_proj: Down projection (no bias).
        up_proj: Up projection (no bias).

    Example:
        >>> mlp = HunyuanMLP(dim=256, hidden_dim=512)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, dim: int, hidden_dim: int):
        """Initialize HunyuanMLP.

        Args:
            dim: Input and output dimensionality.
            hidden_dim: Intermediate (expanded) dimensionality.
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


class HunyuanMoeBlock(nn.Module):
    """Mixture-of-Experts block with optional shared experts for Hunyuan.

    Routes tokens to the top-k experts via a softmax gate. When
    ``use_mixed_mlp_moe`` is enabled, adds a shared MLP whose output is
    combined with the routed expert output.

    Attributes:
        use_shared_mlp: Whether shared experts are used.
        num_experts: Total number of routing experts.
        top_k: Number of experts activated per token.
        gate: Router linear projection.
        switch_mlp: SwitchGLU expert bank.
        shared_mlp: Shared MLP (only present when ``use_shared_mlp`` is True).

    Example:
        >>> config = HunyuanConfig(hidden_size=256, num_experts=4, moe_topk=2,
        ...     use_mixed_mlp_moe=True, num_shared_expert=1, intermediate_size=512)
        >>> moe = HunyuanMoeBlock(config, layer_idx=0)
        >>> out = moe(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HunyuanConfig, layer_idx: int = 0):
        """Initialize HunyuanMoeBlock.

        Args:
            config: Hunyuan model configuration.
            layer_idx: Layer index, used to index per-layer MoE parameters.
        """
        super().__init__()
        dim = config.hidden_size
        intermediate_size = config.intermediate_size
        self.use_shared_mlp = config.use_mixed_mlp_moe

        if config.use_mixed_mlp_moe:
            num_shared = _int_or_list(config.num_shared_expert, layer_idx)
            self.shared_mlp = HunyuanMLP(dim, int(intermediate_size * num_shared))

        self.num_experts = config.num_experts
        self.top_k = _int_or_list(config.moe_topk, layer_idx)

        self.gate = nn.Linear(dim, self.num_experts, bias=False)

        expert_intermediate_size = intermediate_size
        if config.moe_intermediate_size is not None:
            expert_intermediate_size = _int_or_list(config.moe_intermediate_size, layer_idx)

        self.switch_mlp = SwitchGLU(dim, expert_intermediate_size, self.num_experts)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        gates = self.gate(hidden_states)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)

        if self.use_shared_mlp:
            shared_expert_output = self.shared_mlp(hidden_states)
            y = y + shared_expert_output

        return y


class HunyuanDecoderLayer(nn.Module):
    """Single Hunyuan decoder layer with CLA and MoE support.

    Combines pre-norm attention (with optional CLA sharing) and a
    feed-forward block that is either a dense MLP (when ``num_experts == 1``)
    or a MoE block.

    Attributes:
        self_attn: Hunyuan attention sub-layer.
        mlp: Feed-forward sub-layer (HunyuanMLP or HunyuanMoeBlock).
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = HunyuanConfig(hidden_size=256, num_attention_heads=8)
        >>> layer = HunyuanDecoderLayer(config, kv_proj=True, layer_idx=0)
        >>> h = mx.zeros((1, 10, 256))
        >>> out, kv = layer(h)
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: HunyuanConfig, kv_proj: bool = True, layer_idx: int = 0):
        """Initialize HunyuanDecoderLayer.

        Args:
            config: Hunyuan model configuration.
            kv_proj: Whether the attention layer has its own KV projections.
            layer_idx: Layer index for per-layer MoE configuration.
        """
        super().__init__()
        self.self_attn = HunyuanAttention(config, kv_proj=kv_proj)

        if config.num_experts == 1:
            self.mlp = HunyuanMLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = HunyuanMoeBlock(config, layer_idx)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
        shared_kv_states: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """Execute the decoder layer.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.
            shared_kv_states: KV states shared from a CLA donor layer.

        Returns:
            A tuple of ``(output, kv_states)`` for downstream CLA sharing.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, shared_kv_states = self.self_attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            kv_states=shared_kv_states,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, shared_kv_states


@register_module(task_type=TaskType.BASE_MODULE, config=HunyuanConfig, model_type="hunyuan")
class HunyuanModel(EasyMLXBaseModule):
    """Base Hunyuan transformer model with CLA and MoE (no LM head).

    Stacks ``num_hidden_layers`` HunyuanDecoderLayer instances with
    Cross-Layer Attention (CLA) KV sharing, preceded by token embeddings
    and followed by a final RMSNorm. CLA layers share KV projections from
    the most recent ``cla_share_factor``-aligned layer.

    Attributes:
        config_class: Associated configuration class (``HunyuanConfig``).
        config: Model configuration.
        embed_tokens: Token embedding layer.
        layers: List of ``HunyuanDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = HunyuanConfig(vocab_size=1000, hidden_size=320,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = HunyuanModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 320]
    """

    config_class = HunyuanConfig

    def __init__(self, config: HunyuanConfig):
        """Initialize HunyuanModel.

        Args:
            config: Hunyuan model configuration.
        """
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            HunyuanDecoderLayer(
                config,
                kv_proj=(not config.use_cla) or (i % config.cla_share_factor == 0),
                layer_idx=i,
            )
            for i in range(config.num_hidden_layers)
        ]
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
        """Run the Hunyuan transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; if provided,
                ``input_ids`` is ignored.
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` after
            the final RMSNorm.

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
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        shared_kv_states = None
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[i]
            if (not self.config.use_cla) or (i % self.config.cla_share_factor == 0):
                shared_kv_states = None
            hidden_states, shared_kv_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
                shared_kv_states=shared_kv_states,
            )

        return self.norm(hidden_states)


@register_module(task_type=TaskType.CAUSAL_LM, config=HunyuanConfig, model_type="hunyuan")
class HunyuanForCausalLM(BaseCausalLMModule[HunyuanModel, HunyuanConfig]):
    """Hunyuan transformer with a causal language modeling head.

    Wraps ``HunyuanModel`` and adds an LM head for next-token prediction.
    Uses tied embeddings by default.

    Attributes:
        config_class: Associated configuration class (``HunyuanConfig``).

    Example:
        >>> config = HunyuanConfig(vocab_size=1000, hidden_size=320,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = HunyuanForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = HunyuanConfig

    def __init__(self, config: HunyuanConfig):
        """Initialize HunyuanForCausalLM.

        Args:
            config: Hunyuan model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=HunyuanModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream Hunyuan checkpoint weights.

        Handles three transformations:
        1. Splits fused ``qkv_proj`` into separate ``q_proj``, ``k_proj``,
           ``v_proj`` weights.
        2. Splits fused ``gate_and_up_proj`` into ``gate_proj`` and ``up_proj``.
        3. Stacks per-expert MLP weights into SwitchGLU format.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary ready for model loading.
        """
        weights = super().sanitize(weights)

        # Handle fused qkv_proj and gate_and_up_proj
        if "model.layers.0.mlp.gate_and_up_proj.weight" in weights:
            new_weights = {}
            D = self.config.hidden_size
            n_kv_heads = self.config.num_key_value_heads
            n_kv_groups = self.config.num_attention_heads // n_kv_heads
            head_dim = D // self.config.num_attention_heads
            for k, v in weights.items():
                if "qkv_proj" in k:
                    v = v.reshape(n_kv_heads, n_kv_groups + 2, head_dim, -1)
                    splits = v.split([n_kv_groups, n_kv_groups + 1], axis=1)
                    for k_up, v_new in zip(["q_proj", "k_proj", "v_proj"], splits, strict=False):
                        k_new = k.replace("qkv_proj", k_up)
                        new_weights[k_new] = mx.flatten(v_new, 0, 2)
                elif "gate_and_up_proj" in k:
                    splits = v.split(2, axis=0)
                    for k_up, v_new in zip(["up_proj", "gate_proj"], splits, strict=False):
                        k_new = k.replace("gate_and_up_proj", k_up)
                        new_weights[k_new] = v_new
                else:
                    new_weights[k] = v
            weights = new_weights

        # Handle MoE expert restacking
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}") for e in range(self.config.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)
        return weights


__all__ = ("HunyuanForCausalLM", "HunyuanModel")
