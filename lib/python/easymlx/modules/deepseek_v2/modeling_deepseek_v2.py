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

"""DeepSeek V2 MLX model implementation for serving and inference.

This module provides the DeepSeek V2 architecture on MLX, featuring
Multi-head Latent Attention (MLA) with compressed KV projections,
YarnRoPE for extended context, MoE with group expert selection, and
a causal language model wrapper.
"""

from __future__ import annotations

import math

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

from .deepseek_v2_configuration import DeepseekV2Config

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


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Find the correction dimension for YaRN interpolation.

    Computes the RoPE dimension index at which the given number of rotations
    is achieved within ``max_position_embeddings``.

    Args:
        num_rotations: Target number of rotations.
        dim: RoPE dimensionality.
        base: RoPE base frequency. Defaults to ``10000``.
        max_position_embeddings: Maximum position for the original model.
            Defaults to ``2048``.

    Returns:
        The floating-point correction dimension index.
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find the range of RoPE dimensions that need YaRN correction.

    Args:
        low_rot: Lower rotation bound (``beta_fast``).
        high_rot: Upper rotation bound (``beta_slow``).
        dim: RoPE dimensionality.
        base: RoPE base frequency. Defaults to ``10000``.
        max_position_embeddings: Original maximum position. Defaults to ``2048``.

    Returns:
        A tuple ``(low, high)`` of clamped integer dimension indices.
    """
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale=1, mscale=1):
    """Compute the YaRN magnitude scaling factor.

    Args:
        scale: Context extension scaling factor. Defaults to ``1``.
        mscale: Magnitude scaling coefficient. Defaults to ``1``.

    Returns:
        The computed mscale value (``1.0`` when ``scale <= 1``).
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_val, max_val, dim):
    """Create a linear ramp mask for blending interpolated and extrapolated frequencies.

    Args:
        min_val: Start of the ramp (dimension index, inclusive).
        max_val: End of the ramp (dimension index, inclusive).
        dim: Total number of frequency dimensions.

    Returns:
        An ``mx.array`` of shape ``(dim,)`` with values linearly ramped
        from 0 to 1 between ``min_val`` and ``max_val``, clamped to ``[0, 1]``.
    """
    if min_val == max_val:
        max_val += 0.001
    linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
    return mx.clip(linear_func, 0, 1)


class DeepseekV2YarnRotaryEmbedding(nn.Module):
    """YaRN-based rotary positional embedding for DeepSeek V2 MLA.

    Implements Yet another RoPE extensioN (YaRN) which blends interpolated and
    extrapolated rotary frequencies based on a linear ramp, then applies
    magnitude scaling to compensate for the longer context.

    Attributes:
        mscale: Combined magnitude scaling factor applied to inputs.

    Example::

        >>> yarn_rope = DeepseekV2YarnRotaryEmbedding(dim=64, scaling_factor=4.0)
        >>> rotated = yarn_rope(queries, offset=0)
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        """Initialize YaRN rotary embedding.

        Args:
            dim: Dimensionality of the rotary embedding (RoPE head dim).
            max_position_embeddings: Extended maximum position. Defaults to ``2048``.
            base: RoPE base frequency. Defaults to ``10000``.
            scaling_factor: Context extension factor. Defaults to ``1.0``.
            original_max_position_embeddings: Original model's max positions.
                Defaults to ``4096``.
            beta_fast: Fast rotation bound for correction range. Defaults to ``32``.
            beta_slow: Slow rotation bound for correction range. Defaults to ``1``.
            mscale: Magnitude scaling coefficient. Defaults to ``1``.
            mscale_all_dim: Magnitude scaling coefficient for attention scale
                adjustment. Defaults to ``0``.
        """
        super().__init__()
        self.mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
        freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        freq_inter = scaling_factor * base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        self._freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))

    def __call__(self, x, offset=0):
        """Apply YaRN rotary positional embedding.

        Args:
            x: Input tensor to apply rotary embedding to.
            offset: Position offset for autoregressive generation.
                Defaults to ``0``.

        Returns:
            Tensor with YaRN-scaled rotary positional encoding applied.
        """
        if self.mscale != 1.0:
            x = self.mscale * x
        return mx.fast.rope(x, x.shape[-1], traditional=True, base=None, scale=1.0, offset=offset, freqs=self._freqs)


class DeepseekV2Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek V2.

    MLA compresses key-value projections using a LoRA-like decomposition to
    reduce KV cache memory. Hidden states are projected down to a low-rank
    ``kv_lora_rank`` latent via ``kv_a_proj_with_mqa``, normalized, then
    projected back up via ``kv_b_proj`` to produce per-head key and value
    tensors. RoPE is applied only to a dedicated ``qk_rope_head_dim`` slice
    of the query and key, while the remaining ``qk_nope_head_dim`` dimensions
    carry the non-positional signal. Queries optionally use a similar LoRA
    decomposition (``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj``).

    When YaRN rope scaling is configured, the attention scale is adjusted by
    the YaRN mscale factor to preserve attention entropy at extended contexts.

    Attributes:
        config: The model configuration.
        hidden_size: Hidden dimension size.
        num_heads: Number of query attention heads.
        q_lora_rank: Query LoRA rank (``None`` for direct projection).
        qk_rope_head_dim: RoPE portion dimension per head.
        kv_lora_rank: KV LoRA compression rank.
        v_head_dim: Value head dimension.
        qk_nope_head_dim: Non-RoPE portion dimension per head.
        q_head_dim: Total query head dimension (nope + rope).
        scale: Attention scaling factor.
        rope: Rotary positional embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> attn = DeepseekV2Attention(config)
        >>> out = attn(hidden_states, mask=mask, cache_view=cache)
    """

    def __init__(self, config: DeepseekV2Config):
        """Initialize MLA attention module.

        Args:
            config: DeepSeek V2 model configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        _has_yarn = (
            config.rope_scaling is not None and isinstance(config.rope_scaling, dict) and "factor" in config.rope_scaling
        )
        if _has_yarn:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scale = self.scale * mscale * mscale

            rope_kwargs = {
                key: config.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in config.rope_scaling
            }
            self.rope = DeepseekV2YarnRotaryEmbedding(
                dim=self.qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=config.rope_theta,
                **rope_kwargs,
            )
        else:
            self.rope = get_rope(
                dims=self.qk_rope_head_dim,
                base=config.rope_theta,
                traditional=True,
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
        """Run the MLA forward pass.

        Computes queries (optionally via LoRA), compressed KV latents, applies
        RoPE to the positional slices, and performs manual scaled dot-product
        attention with separate nope/pe key dimensions.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional boolean attention mask of shape
                ``(batch, 1, seq_len, kv_len)``, or ``None``.
            cache_view: Per-layer KV cache view for autoregressive generation.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _D = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache_view is not None:
            offset = cache_view.offset
            q_pe = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            if isinstance(mask, mx.array):
                scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
            else:
                pass
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DeepseekV2MLP(nn.Module):
    """SiLU-gated feed-forward MLP for DeepSeek V2.

    Implements the standard SwiGLU MLP: ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``.

    Attributes:
        gate_proj: Gate projection linear layer.
        up_proj: Up projection linear layer.
        down_proj: Down projection linear layer.

    Example::

        >>> mlp = DeepseekV2MLP(config)
        >>> output = mlp(hidden_states)
    """

    def __init__(self, config: DeepseekV2Config, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for the intermediate dimension. If ``None``,
                uses ``config.intermediate_size``.
        """
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run the SwiGLU MLP forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DeepseekV2MoE(nn.Module):
    """Mixture-of-Experts block for DeepSeek V2 with group expert selection.

    Routes tokens through the top-k routed experts via ``TopKRouter`` and a
    ``SwitchGLU`` expert bank. If shared experts are configured, their output
    is added to the routed result.

    Attributes:
        config: The model configuration.
        router: Top-k expert router with group selection.
        switch_mlp: Batched SwiGLU expert bank.
        shared_experts: Optional shared MLP active for all tokens.

    Example::

        >>> moe = DeepseekV2MoE(config)
        >>> output = moe(hidden_states)
    """

    def __init__(self, config: DeepseekV2Config):
        """Initialize the MoE block.

        Args:
            config: Model configuration with MoE hyperparameters.
        """
        super().__init__()
        self.config = config
        self.router = TopKRouter(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            scoring_func="softmax",
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group or 1,
            topk_group=config.topk_group or 1,
        )
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)

        if config.n_shared_experts is not None:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through selected experts and aggregate outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of shape ``(..., hidden_size)``.
        """
        inds, scores = self.router(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class DeepseekV2DecoderLayer(nn.Module):
    """Single DeepSeek V2 decoder layer.

    Contains MLA attention, pre/post-attention RMSNorm, and either a dense MLP
    or an MoE block depending on layer index, ``first_k_dense_replace``, and
    ``moe_layer_freq``.

    Attributes:
        self_attn: Multi-head Latent Attention module.
        mlp: Dense MLP or MoE block.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.

    Example::

        >>> layer = DeepseekV2DecoderLayer(config, layer_idx=0)
        >>> output = layer(hidden_states, mask=mask)
    """

    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based index of this layer, used to determine
                whether to use MoE or dense MLP.
        """
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
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
        """Run the decoder layer forward pass (pre-norm residual).

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

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


@register_module(task_type=TaskType.BASE_MODULE, config=DeepseekV2Config, model_type="deepseek_v2")
class DeepseekV2Model(EasyMLXBaseModule):
    """Base DeepSeek V2 transformer model with Multi-head Latent Attention.

    Implements a decoder-only transformer with MLA attention (LoRA-compressed
    KV projections), YaRN RoPE for extended context, and MoE with group
    expert selection for applicable layers.

    Attributes:
        config_class: The configuration class (``DeepseekV2Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``DeepseekV2DecoderLayer`` instances.
        norm: Final RMS normalization layer.

    Example::

        >>> config = DeepseekV2Config(vocab_size=102400, hidden_size=4096)
        >>> model = DeepseekV2Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV2Config

    def __init__(self, config: DeepseekV2Config):
        """Initialize the DeepSeek V2 base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
            input_ids: Integer token IDs of shape ``(batch, seq_len)``
                or ``(seq_len,)`` (auto-batched).
            attention_mask: Optional mask of shape ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings of shape
                ``(batch, seq_len, hidden_size)`` instead of ``input_ids``.
            cache_views: Per-layer KV cache views for autoregressive generation.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the layer count.
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
        """Transform checkpoint weights for DeepSeek V2.

        Performs the following transformations:

        1. **Expert stacking**: Individual per-expert ``gate_proj``, ``up_proj``,
           and ``down_proj`` weights are stacked into a single
           ``switch_mlp.{proj}.weight`` tensor for batched expert evaluation.
        2. **Router remapping**: Renames ``mlp.gate.weight`` to ``mlp.router.weight``.
        3. **Rotary filter**: Removes ``rotary_emb.inv_freq`` and ``rope.inv_freq``
           keys since frequencies are computed at init time.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with stacked experts and remapped keys.
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

                gate_key = f"{prefix}.mlp.gate.weight"
                router_key = f"{prefix}.mlp.router.weight"
                if gate_key in weights and router_key not in weights:
                    weights[router_key] = weights.pop(gate_key)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=DeepseekV2Config, model_type="deepseek_v2")
class DeepseekV2ForCausalLM(BaseCausalLMModule[DeepseekV2Model, DeepseekV2Config]):
    """DeepSeek V2 causal language model.

    Wraps ``DeepseekV2Model`` with a language modeling head for next-token
    prediction. Supports tied and untied word embeddings.

    Attributes:
        config_class: The configuration class (``DeepseekV2Config``).

    Example::

        >>> config = DeepseekV2Config(vocab_size=102400, hidden_size=4096)
        >>> model = DeepseekV2ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV2Config

    def __init__(self, config: DeepseekV2Config):
        """Initialize the DeepSeek V2 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekV2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("DeepseekV2ForCausalLM", "DeepseekV2Model")
