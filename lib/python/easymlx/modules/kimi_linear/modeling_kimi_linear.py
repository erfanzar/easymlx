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

"""Kimi Linear MLX model implementation for serving and inference.

Kimi Linear is a hybrid model mixing:
- Multi-head Latent Attention (MLA) layers with MultiLinear absorbed projections
- Gated delta rule linear attention (KDA) layers with short convolutions
- MoE feed-forward with sigmoid routing and shared experts

This is a faithful port of the upstream kimi_linear architecture.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .kimi_linear_configuration import KimiLinearConfig

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


class MultiLinear(nn.Module):
    """Per-head linear projection for MLA absorbed attention.

    Stores a weight tensor of shape ``(num_heads, output_dims, input_dims)``
    and applies a per-head matmul. Supports transpose mode for absorbed
    key computation.

    Attributes:
        weight: Learnable weight tensor of shape ``(num_heads, output_dims, input_dims)``.

    Example:
        >>> ml = MultiLinear(input_dims=64, output_dims=32, num_heads=8)
        >>> x = mx.zeros((1, 8, 10, 64))
        >>> out = ml(x)
        >>> out.shape
        [1, 8, 10, 32]
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        """Initialize MultiLinear.

        Args:
            input_dims: Input feature dimensionality.
            output_dims: Output feature dimensionality.
            num_heads: Number of heads.
        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_heads, output_dims, input_dims))

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        """Apply per-head linear projection.

        Args:
            x: Input tensor with a heads dimension.
            transpose: If ``True``, multiply by ``weight^T`` (standard
                projection). If ``False``, multiply by ``weight`` (used
                for absorbed key computation).

        Returns:
            Projected tensor.
        """
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight


class KimiLinearMLP(nn.Module):
    """SwiGLU MLP for Kimi Linear.

    Attributes:
        gate_proj: Gate projection (no bias).
        up_proj: Up projection (no bias).
        down_proj: Down projection (no bias).

    Example:
        >>> config = KimiLinearConfig(hidden_size=256, intermediate_size=512)
        >>> mlp = KimiLinearMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ):
        """Initialize KimiLinearMLP.

        Args:
            config: Kimi Linear model configuration.
            hidden_size: Override for input/output dimensionality.
            intermediate_size: Override for intermediate dimensionality.
        """
        super().__init__()
        dim = hidden_size or config.hidden_size
        hidden = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> tuple[mx.array, mx.array]:
    """Select top-k experts with optional grouped routing and bias correction.

    Args:
        gates: Router logits of shape ``(..., num_experts)``.
        bias: Expert score correction bias of shape ``(num_experts,)``.
        top_k: Number of experts to activate per token.
        n_group: Number of expert groups.
        topk_group: Number of top groups to keep.
        routed_scaling_factor: Scaling factor applied to final scores.
        renormalize: Whether to renormalize scores after selection.
        score_function: Activation function (``"sigmoid"`` or ``"softmax"``).

    Returns:
        Tuple of ``(indices, scores)`` where *indices* has shape
        ``(..., top_k)`` and *scores* has shape ``(..., top_k)``.
    """
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    orig_scores = scores
    scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiLinearMoE(nn.Module):
    """MoE block for Kimi Linear with sigmoid routing and shared experts.

    Uses grouped top-k expert selection with bias correction and optional
    score renormalization. Shared experts are always active and their output
    is added to the routed output.

    Attributes:
        config: Model configuration.
        gate: Router linear projection.
        switch_mlp: SwitchGLU expert bank.
        e_score_correction_bias: Expert bias correction term.
        shared_experts: Shared MLP (or ``None`` if no shared experts).

    Example:
        >>> config = KimiLinearConfig(hidden_size=256, num_experts=4,
        ...     moe_intermediate_size=512, num_experts_per_token=2,
        ...     num_shared_experts=1)
        >>> moe = KimiLinearMoE(config)
        >>> out = moe(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KimiLinearConfig):
        """Initialize KimiLinearMoE.

        Args:
            config: Kimi Linear model configuration.
        """
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        experts = config.num_experts

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, config.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if config.num_shared_experts:
            shared_hidden = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiLinearMLP(config, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        scores = self.gate(hidden_states)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.config.num_experts_per_token,
            self.config.num_expert_group,
            self.config.topk_group,
            self.config.routed_scaling_factor,
            self.config.moe_renormalize,
            self.config.moe_router_activation_func,
        )
        out = self.switch_mlp(hidden_states, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class KimiMLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) for Kimi Linear.

    Uses KV LoRA compression: projects hidden states to a low-rank latent
    space, then absorbs the up-projection into per-head ``embed_q`` and
    ``unembed_out`` matrices. Separates Q/K into non-positional (nope) and
    positional (RoPE) components for stable long-context attention.

    Attributes:
        config: Model configuration.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        qk_nope_head_dim: Non-RoPE Q/K head dimension.
        qk_rope_head_dim: RoPE Q/K head dimension.
        q_head_dim: Total Q head dimension (nope + rope).
        v_head_dim: Value head dimension.
        kv_lora_rank: KV LoRA compression rank.
        scale: Attention scaling factor.
        q_proj: Query projection.
        kv_a_proj_with_mqa: Joint KV latent + rope-key projection.
        kv_a_layernorm: Normalization for KV latent.
        embed_q: Per-head absorbed key projection.
        unembed_out: Per-head absorbed value projection.
        o_proj: Output projection.
        rope: Rotary position embedding.

    Example:
        >>> config = KimiLinearConfig(hidden_size=256, num_attention_heads=8,
        ...     kv_lora_rank=64, qk_nope_head_dim=96, qk_rope_head_dim=32,
        ...     v_head_dim=128)
        >>> attn = KimiMLAAttention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KimiLinearConfig):
        """Initialize KimiMLAAttention.

        Args:
            config: Kimi Linear model configuration.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim or config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim or config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            config.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, config.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(config.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

        self.rope = get_rope(
            dims=self.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=True,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.model_max_length,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute MLA attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view. Caches the KV latent and
                rope-key for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
        """
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        offset = cache_view.offset if cache_view is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        if cache_view is not None:
            kv_latent, k_pe, _ = cache_view.concatenate_to_cache(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        nope_scores = (q_nope * self.scale) @ k.swapaxes(-1, -2)
        scores = nope_scores + pe_scores
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ v

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class KimiLinearDecoderLayer(nn.Module):
    """Single Kimi Linear decoder layer with MLA attention and optional MoE.

    For non-KDA layers, uses Multi-head Latent Attention (MLA). The
    feed-forward block is either a dense MLP or an MoE block depending
    on the layer index and MoE configuration.

    Attributes:
        is_linear: Whether this layer should use linear attention (KDA).
            Currently, KDA layers fall back to MLA in this port.
        self_attn: MLA attention sub-layer.
        mlp: Feed-forward sub-layer (MLP or MoE).
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MLP.

    Example:
        >>> config = KimiLinearConfig(hidden_size=256, num_attention_heads=8,
        ...     kv_lora_rank=64)
        >>> layer = KimiLinearDecoderLayer(config, layer_idx=0)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        """Initialize KimiLinearDecoderLayer.

        Args:
            config: Kimi Linear model configuration.
            layer_idx: Layer index for MoE and KDA scheduling.
        """
        super().__init__()
        kda_layers = config.linear_attn_config.get("kda_layers", [])
        self.is_linear = (layer_idx + 1) in kda_layers

        self.self_attn = KimiMLAAttention(config)

        if (
            config.num_experts > 0
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = KimiLinearMoE(config)
        else:
            self.mlp = KimiLinearMLP(config)

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


@register_module(task_type=TaskType.BASE_MODULE, config=KimiLinearConfig, model_type="kimi_linear")
class KimiLinearModel(EasyMLXBaseModule):
    """Base Kimi Linear transformer model with MLA and MoE (no LM head).

    Combines Multi-head Latent Attention (MLA) layers with optional
    MoE feed-forward blocks. The architecture mirrors the upstream
    kimi_linear model.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding layer.
        layers: List of ``KimiLinearDecoderLayer`` instances.
        norm: Final RMSNorm.

    Example:
        >>> config = KimiLinearConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, kv_lora_rank=64)
        >>> model = KimiLinearModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
        >>> out.shape
        [1, 3, 256]
    """

    config_class = KimiLinearConfig

    def __init__(self, config: KimiLinearConfig):
        """Initialize KimiLinearModel.

        Args:
            config: Kimi Linear model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [KimiLinearDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
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
        """Run the Kimi Linear transformer forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
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
        """Sanitize upstream Kimi Linear checkpoint weights.

        Performs the following transformations:
        1. Strips MTP (multi-token prediction) weights.
        2. Drops tied LM head weights if applicable.
        3. Remaps ``block_sparse_moe`` expert weights to SwitchGLU format
           and stacks per-expert tensors.
        4. Splits ``kv_b_proj`` into absorbed ``embed_q`` (key) and
           ``unembed_out`` (value) per-head projections, handling both
           quantized and full-precision weights.
        5. Removes rotary inv_freq buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary ready for model loading.
        """
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if getattr(self.config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"

            if self.config.num_experts > 0:
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                    key = f"{src_prefix}.experts.0.{src}.weight"
                    if key in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight") for i in range(self.config.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(stacked)

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = weights.pop(src_key)

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(bias_key)

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.config.qk_nope_head_dim or self.config.head_dim
                v_head = self.config.v_head_dim or self.config.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.config.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.config.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=KimiLinearConfig, model_type="kimi_linear")
class KimiLinearForCausalLM(BaseCausalLMModule[KimiLinearModel, KimiLinearConfig]):
    """Kimi Linear transformer with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = KimiLinearConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8, kv_lora_rank=64)
        >>> model = KimiLinearForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = KimiLinearConfig

    def __init__(self, config: KimiLinearConfig):
        """Initialize KimiLinearForCausalLM.

        Args:
            config: Kimi Linear model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=KimiLinearModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("KimiLinearForCausalLM", "KimiLinearModel")
