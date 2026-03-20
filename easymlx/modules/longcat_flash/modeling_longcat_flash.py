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

"""LongcatFlash MLX model implementation for serving and inference.

LongcatFlash uses Multi-head Latent Attention (MLA) with dual attention
sub-layers per decoder block, MoE feed-forward with identity zero experts,
and SwiGLU dense MLPs. Each block runs two attention+MLP passes with an
asynchronous MoE shortcut on the first pass.
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

from .longcat_flash_configuration import LongcatFlashConfig

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


class MultiLinear(nn.Module):
    """Per-head independent linear projection for MLA absorbed attention.

    Each head has its own weight matrix, enabling per-head transformations
    without sharing parameters across heads.

    Attributes:
        weight: Weight tensor of shape ``(num_heads, output_dims, input_dims)``.

    Example:
        >>> proj = MultiLinear(input_dims=32, output_dims=64, num_heads=8)
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        """Initialize the per-head linear projection.

        Args:
            input_dims: Input feature dimension per head.
            output_dims: Output feature dimension per head.
            num_heads: Number of independent heads.
        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_heads, output_dims, input_dims))

    def __call__(self, x, transpose=True):
        """Apply per-head linear projection.

        Args:
            x: Input tensor with a head dimension.
            transpose: If ``True``, multiply by ``W^T`` (standard projection).
                If ``False``, multiply by ``W`` (inverse direction).

        Returns:
            Projected tensor.
        """
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight


class LongcatFlashMLA(nn.Module):
    """Multi-head Latent Attention (MLA) for LongcatFlash.

    Compresses keys and values into a low-rank latent space via LoRA-style
    projections, applies RoPE to a separate positional embedding component,
    and uses absorbed attention for efficient computation.

    Attributes:
        num_attention_heads: Number of attention heads.
        qk_rope_head_dim: Dimension of the RoPE portion per head.
        qk_nope_head_dim: Dimension of the non-RoPE portion per head.
        kv_lora_rank: KV compression rank.
        q_lora_rank: Query compression rank (``None`` for direct projection).
        v_head_dim: Value head dimension.
        scale: Attention scaling factor.

    Example:
        >>> config = LongcatFlashConfig(hidden_size=64, num_attention_heads=4)
        >>> mla = LongcatFlashMLA(config)
    """

    def __init__(self, config: LongcatFlashConfig):
        """Initialize LongcatFlash MLA.

        Args:
            config: Model configuration with MLA hyperparameters.
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.v_head_dim = config.v_head_dim

        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        self.mla_scale_q_lora = None
        self.mla_scale_kv_lora = None

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_attention_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_attention_heads)

        self.o_proj = nn.Linear(
            self.num_attention_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias
        )

        if config.mla_scale_q_lora and self.q_lora_rank is not None:
            self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        if config.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5

        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = config.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.rope = get_rope(
            dims=self.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=True,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute MLA attention with latent KV compression and RoPE.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_view: KV cache view storing compressed latent and PE keys.
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_attention_heads, self.qk_head_dim).transpose(0, 2, 1, 3)

        if self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora

        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        offset = cache_view.offset if cache_view is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache_view is not None:
            kv_latent, k_pe, _ = cache_view.concatenate_to_cache(kv_latent, k_pe)

        # Compute PE scores
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        # MLA absorbed attention
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


class LongcatFlashMLP(nn.Module):
    """SwiGLU feed-forward MLP for LongcatFlash.

    Computes ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection.
        up_proj: Up projection.
        down_proj: Down projection.

    Example:
        >>> config = LongcatFlashConfig(hidden_size=64, ffn_hidden_size=128)
        >>> mlp = LongcatFlashMLP(config)
    """

    def __init__(self, config: LongcatFlashConfig, is_expert: bool = False):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            is_expert: If ``True``, uses ``expert_ffn_hidden_size`` instead
                of ``ffn_hidden_size``.
        """
        super().__init__()
        hidden_size = config.expert_ffn_hidden_size if is_expert else config.ffn_hidden_size
        self.gate_proj = nn.Linear(config.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LongcatFlashRouter(nn.Module):
    """Softmax router for LongcatFlash MoE with score correction bias.

    Routes tokens to top-k experts using softmax scores, with an additive
    score correction bias and optional probability normalization.

    Attributes:
        top_k: Number of experts activated per token.
        n_routed_experts: Total number of experts (including identity).
        routed_scaling_factor: Multiplicative scaling for routing weights.
        norm_topk_prob: Whether to normalize top-k probabilities.
        classifier: Linear projection for computing router logits.
        e_score_correction_bias: Learned additive bias for score correction.

    Example:
        >>> config = LongcatFlashConfig(hidden_size=64, n_routed_experts=8, moe_topk=2)
        >>> router = LongcatFlashRouter(config)
    """

    def __init__(self, config: LongcatFlashConfig):
        """Initialize the router.

        Args:
            config: Model configuration with routing hyperparameters.
        """
        super().__init__()
        self.top_k = config.moe_topk
        self.n_routed_experts = config.n_routed_experts + config.zero_expert_num
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        self.classifier = nn.Linear(config.hidden_size, self.n_routed_experts, bias=config.router_bias)
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route input tokens to top-k experts.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Tuple of ``(topk_indices, topk_weights)`` where indices select
            experts and weights are the corresponding routing scores.
        """
        router_logits = self.classifier(x)
        scores = mx.softmax(router_logits, axis=-1)

        corrected = scores + self.e_score_correction_bias
        topk_indices = mx.argpartition(corrected, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

        if self.norm_topk_prob:
            denominator = mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights.astype(x.dtype)


class LongcatFlashMoE(nn.Module):
    """MoE block for LongcatFlash with identity (zero) experts.

    Routes tokens through routed SwitchGLU experts and adds identity
    expert contributions for experts beyond ``n_routed_experts``.

    Attributes:
        n_routed_experts: Number of standard routed experts.
        zero_expert_num: Number of identity experts.
        switch_mlp: SwitchGLU module containing all routed expert weights.
        router: Softmax router with score correction.

    Example:
        >>> config = LongcatFlashConfig(
        ...     hidden_size=64, n_routed_experts=8,
        ...     expert_ffn_hidden_size=32, moe_topk=2,
        ... )
        >>> moe = LongcatFlashMoE(config)
    """

    def __init__(self, config: LongcatFlashConfig):
        """Initialize the MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num

        self.switch_mlp = SwitchGLU(config.hidden_size, config.expert_ffn_hidden_size, config.n_routed_experts)
        self.router = LongcatFlashRouter(config)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts with identity expert fallback.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Combined expert output of shape ``(..., hidden_size)``.
        """
        topk_indices, topk_weights = self.router(hidden_states)

        # Mask out identity experts
        mask = topk_indices >= self.n_routed_experts
        topk_indices_clamped = mx.where(mask, 0, topk_indices)
        regular_weights = mx.where(mask, 0.0, topk_weights)

        regular_outputs = self.switch_mlp(hidden_states, topk_indices_clamped)
        weighted_outputs = regular_outputs * regular_weights[..., None]
        final_output = mx.sum(weighted_outputs, axis=-2)

        # Add identity expert contribution
        identity_weights_sum = mx.sum(mx.where(mask, topk_weights, 0.0), axis=-1, keepdims=True)
        final_output = final_output + hidden_states * identity_weights_sum

        return final_output


class LongcatFlashDecoderLayer(nn.Module):
    """Decoder layer with dual MLA attention sub-layers and asynchronous MoE shortcut.

    Each block runs two attention + dense MLP passes. The MoE feed-forward
    is computed on the first pass's post-attention output and its result
    is added as a shortcut after the second pass completes.

    Attributes:
        mlp: MoE feed-forward block (applied as shortcut).
        self_attn: List of two MLA attention sub-layers.
        mlps: List of two dense SwiGLU MLP sub-layers.
        input_layernorm: List of two pre-attention RMSNorm layers.
        post_attention_layernorm: List of two pre-MLP RMSNorm layers.

    Example:
        >>> config = LongcatFlashConfig(hidden_size=64, num_attention_heads=4)
        >>> layer = LongcatFlashDecoderLayer(config)
    """

    def __init__(self, config: LongcatFlashConfig):
        """Initialize the decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.mlp = LongcatFlashMoE(config)
        self.self_attn = [LongcatFlashMLA(config) for _ in range(2)]
        self.mlps = [LongcatFlashMLP(config, False) for _ in range(2)]
        self.input_layernorm = [nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)]
        self.post_attention_layernorm = [nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)]

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run dual attention passes with MoE shortcut.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Attention mask.
            cache_views: List of two cache views (one per attention sub-layer).
            cache_metadata: Paged attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        hidden_states = x
        shortcut_mlp_output = None

        if cache_views is None:
            cache_views = [None, None]

        for i in range(2):
            residual = hidden_states
            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](
                hidden_states,
                mask=mask,
                cache_view=cache_views[i],
                cache_metadata=cache_metadata,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)

            if i == 0:
                shortcut_mlp_output = self.mlp(hidden_states)

            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states

            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=LongcatFlashConfig, model_type="longcat_flash")
class LongcatFlashModel(EasyMLXBaseModule):
    """Base LongcatFlash transformer model with MLA and MoE.

    Each decoder layer has two MLA attention sub-layers (requiring 2
    cache views per layer) and an asynchronous MoE shortcut.

    Attributes:
        config_class: Associated configuration class.
        embed_tokens: Token embedding table.
        layers: List of ``LongcatFlashDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = LongcatFlashConfig(vocab_size=1000, hidden_size=64, num_layers=2)
        >>> model = LongcatFlashModel(config)
    """

    config_class = LongcatFlashConfig

    def __init__(self, config: LongcatFlashConfig):
        """Initialize the base LongcatFlash model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LongcatFlashDecoderLayer(config) for _ in range(config.num_layers)]
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
        """Run the forward pass through the LongcatFlash backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Cache views for all attention sub-layers. Must have
                length ``2 * num_layers`` (2 per decoder layer).
            cache_metadata: Paged attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length is not ``2 * num_layers``.
        """
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

        # Each layer has 2 attention sub-layers, so cache_views has 2 per layer
        if cache_views is not None:
            if len(cache_views) != len(self.layers) * 2:
                raise ValueError(
                    f"cache_views length must be {len(self.layers) * 2} (2 per layer), got {len(cache_views)}."
                )
            layer_caches = [[cache_views[i * 2], cache_views[i * 2 + 1]] for i in range(len(self.layers))]
        else:
            layer_caches = [None] * len(self.layers)

        for layer, lc in zip(self.layers, layer_caches, strict=False):
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_views=lc,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream weights for LongcatFlash.

        Performs three transformations:
        1. Stacks per-expert weights into ``switch_mlp`` format.
        2. Splits ``kv_b_proj`` into separate ``embed_q`` and ``unembed_out``
           weight matrices for absorbed MLA attention.
        3. Removes rotary embedding inverse frequency keys and MTP layers.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        for layer_idx in range(self.config.num_layers):
            prefix = f"model.layers.{layer_idx}"
            for _n, _m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{_m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{_m}.{k}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{_m}.{k}"] = mx.stack(to_join)

            # Handle kv_b_proj splitting for each of the 2 attention sub-layers
            for i in range(2):
                attn_prefix = f"{prefix}.self_attn.{i}"
                kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
                if kv_b_key in weights:
                    num_heads = self.config.num_attention_heads
                    head_dim = self.config.qk_nope_head_dim + self.config.v_head_dim
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
                    wk = mx.contiguous(v[:, : self.config.qk_nope_head_dim, :].swapaxes(-1, -2))
                    wv = mx.contiguous(v[:, self.config.qk_nope_head_dim :, :])

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
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key and not key.startswith("model.mtp")
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=LongcatFlashConfig, model_type="longcat_flash")
class LongcatFlashForCausalLM(BaseCausalLMModule[LongcatFlashModel, LongcatFlashConfig]):
    """LongcatFlash causal language model with an LM head.

    Wraps ``LongcatFlashModel`` with a language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = LongcatFlashConfig(vocab_size=1000, hidden_size=64, num_layers=2)
        >>> model = LongcatFlashForCausalLM(config)
    """

    config_class = LongcatFlashConfig

    def __init__(self, config: LongcatFlashConfig):
        """Initialize the LongcatFlash causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=LongcatFlashModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("LongcatFlashForCausalLM", "LongcatFlashModel")
