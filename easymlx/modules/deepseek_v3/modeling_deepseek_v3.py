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

"""DeepSeek V3 MLX model implementation for serving and inference.

This module provides the DeepSeek V3 architecture on MLX, featuring
Multi-head Latent Attention (MLA) with MultiLinear layers for
embed_q/unembed_out, FP8 dequantization support, compiled group expert
selection with noaux_tc routing, and a causal language model wrapper.
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
from easymlx.layers.moe import TopKRouter
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .deepseek_v3_configuration import DeepseekV3Config

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


class MultiLinear(nn.Module):
    """Per-head linear projection for MLA absorbed attention.

    Stores a ``(num_heads, output_dims, input_dims)`` weight tensor and
    applies per-head matrix multiplication.
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        """Initialize MultiLinear.

        Args:
            input_dims: Input feature dimension per head.
            output_dims: Output feature dimension per head.
            num_heads: Number of attention heads.
        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_heads, output_dims, input_dims))

    def __call__(self, x, transpose=True):
        """Apply per-head matrix multiplication.

        Args:
            x: Input tensor with shape ``(..., num_heads, seq_len, dim)``.
            transpose: If ``True``, multiplies by transposed weight
                (input_dims -> output_dims). If ``False``, multiplies by
                weight directly (output_dims -> input_dims). Defaults to ``True``.

        Returns:
            Transformed tensor with the last dimension changed according
            to the weight matrix orientation.
        """
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight


class DeepseekV3Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek V3.

    Uses MultiLinear layers to absorb the kv_b_proj into per-head
    embed_q and unembed_out projections, enabling efficient inference
    by caching only the compressed KV latent and the RoPE key.
    """

    def __init__(self, config: DeepseekV3Config):
        """Initialize DeepSeek V3 MLA attention.

        Unlike V2 which stores the full ``kv_b_proj``, V3 absorbs it into
        per-head ``embed_q`` and ``unembed_out`` MultiLinear projections.
        This allows caching only the low-rank KV latent and the RoPE key,
        dramatically reducing KV cache memory.

        Args:
            config: DeepSeek V3 model configuration.
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
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        if config.rope_scaling is not None and isinstance(config.rope_scaling, dict) and "factor" in config.rope_scaling:
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
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the absorbed MLA forward pass.

        During decode (``L == 1``), the ``embed_q`` projection is absorbed into
        the query nope path, and ``unembed_out`` is applied after attention.
        During prefill (``L > 1``), ``embed_q`` is applied to the KV latent to
        produce keys, and ``unembed_out`` produces values.

        Attention scores combine PE-based scores (using RoPE key) and nope-based
        scores (using absorbed projections), enabling efficient latent caching.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional boolean attention mask.
            cache_view: Per-layer KV cache view. Caches the compressed KV latent
                and RoPE key rather than full K/V tensors.
            cache_metadata: Paged-attention metadata.

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
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache_view.offset if cache_view is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache_view is not None:
            kv_latent, k_pe, _ = cache_view.concatenate_to_cache(kv_latent, k_pe)

        # Compute PE-based attention scores
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None and isinstance(mask, mx.array):
            pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))

        # MLA absorbed attention: branch on prefill vs decode
        if L == 1:
            # Decode: absorb embed_q into q_nope for cheaper computation
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            # Prefill: project kv_latent through embed_q and unembed_out
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        # Compute nope attention scores
        nope_scores = (q_nope * self.scale) @ k.swapaxes(-1, -2)
        scores = nope_scores + pe_scores
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = weights @ v

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DeepseekV3MLP(nn.Module):
    """SiLU-gated feed-forward MLP for DeepSeek V3.

    Implements ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``.

    Example::

        >>> mlp = DeepseekV3MLP(config)
        >>> output = mlp(hidden_states)
    """

    def __init__(self, config: DeepseekV3Config, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimension. If ``None``,
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


class DeepseekV3MoE(nn.Module):
    """Mixture-of-Experts block for DeepSeek V3 with noaux_tc routing.

    Uses sigmoid scoring with score correction bias and group-based expert
    selection. Shared experts are always active and added to the routed output.

    Attributes:
        config: Model configuration.
        router: Top-k expert router with noaux_tc method and score bias.
        switch_mlp: Batched SwiGLU expert bank.
        shared_experts: Optional shared MLP.

    Example::

        >>> moe = DeepseekV3MoE(config)
        >>> output = moe(hidden_states)
    """

    def __init__(self, config: DeepseekV3Config):
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
            scoring_func=config.scoring_func,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.n_group,
            topk_group=config.topk_group,
            use_score_bias=True,
        )
        self.switch_mlp = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts)

        if config.n_shared_experts is not None:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through selected experts and aggregate outputs.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of shape ``(..., hidden_size)``.
        """
        inds, scores = self.router(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class DeepseekV3DecoderLayer(nn.Module):
    """Single DeepSeek V3 decoder layer.

    Pre-norm residual architecture with MLA attention and dense MLP or MoE
    block, selected by layer index relative to ``first_k_dense_replace``
    and ``moe_layer_freq``.

    Attributes:
        self_attn: Absorbed MLA attention module.
        mlp: Dense MLP or MoE block.
        input_layernorm: Pre-attention RMSNorm.
        post_attention_layernorm: Pre-MLP RMSNorm.
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index.
        """
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
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
        """Run the decoder layer forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
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


@register_module(task_type=TaskType.BASE_MODULE, config=DeepseekV3Config, model_type="deepseek_v3")
class DeepseekV3Model(EasyMLXBaseModule):
    """Base DeepSeek V3 transformer model with absorbed MLA.

    Extends V2 by absorbing the ``kv_b_proj`` into per-head ``embed_q`` and
    ``unembed_out`` MultiLinear projections, enabling the KV cache to store
    only the compressed latent and RoPE key instead of full K/V tensors.
    Uses noaux_tc routing with sigmoid scoring and score correction bias.

    Attributes:
        config_class: The configuration class (``DeepseekV3Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``DeepseekV3DecoderLayer`` instances.
        norm: Final RMS normalization layer.

    Example::

        >>> config = DeepseekV3Config(vocab_size=102400, hidden_size=4096)
        >>> model = DeepseekV3Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV3Config

    def __init__(self, config: DeepseekV3Config):
        """Initialize the DeepSeek V3 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV3DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
            cache_views: Per-layer KV cache views for generation.
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
        """Transform checkpoint weights for DeepSeek V3.

        Performs the following transformations:

        1. **Expert stacking**: Individual per-expert weights are stacked into
           ``switch_mlp.{proj}.weight`` tensors.
        2. **Router remapping**: Renames ``mlp.gate.weight`` to ``mlp.router.weight``
           and ``mlp.gate.e_score_correction_bias`` to
           ``mlp.router.score_correction_bias``.
        3. **kv_b_proj splitting**: Splits the fused ``kv_b_proj.weight`` into
           separate ``embed_q.weight`` (transposed key nope portion) and
           ``unembed_out.weight`` (value portion) per head. Handles both
           quantized and unquantized weights with re-quantization if needed.
        4. **Rotary filter**: Removes ``rotary_emb.inv_freq`` keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        n_routed = getattr(self.config, "n_routed_experts", None)

        # Stack experts
        if n_routed is not None:
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}") for e in range(n_routed)]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
                # Remap router gate weight and score correction bias
                gate_key = f"{prefix}.mlp.gate.weight"
                router_key = f"{prefix}.mlp.router.weight"
                if gate_key in weights and router_key not in weights:
                    weights[router_key] = weights.pop(gate_key)
                bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
                router_bias_key = f"{prefix}.mlp.router.score_correction_bias"
                if bias_key in weights and router_bias_key not in weights:
                    weights[router_bias_key] = weights.pop(bias_key)

        # Handle kv_b_proj splitting into embed_q and unembed_out
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            if f"{prefix}.kv_b_proj.weight" in weights:
                v = weights.pop(f"{prefix}.kv_b_proj.weight")
                head_dim = self.config.qk_nope_head_dim + self.config.v_head_dim
                quantized = f"{prefix}.kv_b_proj.scales" in weights

                if quantized:
                    dims = self.config.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

                num_heads = self.config.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, : self.config.qk_nope_head_dim, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, self.config.qk_nope_head_dim :, :])

                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{prefix}.embed_q.scales"] = wk_scales
                    weights[f"{prefix}.unembed_out.scales"] = wv_scales
                    weights[f"{prefix}.embed_q.biases"] = wk_biases
                    weights[f"{prefix}.unembed_out.biases"] = wv_biases
                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=DeepseekV3Config, model_type="deepseek_v3")
class DeepseekV3ForCausalLM(BaseCausalLMModule[DeepseekV3Model, DeepseekV3Config]):
    """DeepSeek V3 causal language model.

    Wraps ``DeepseekV3Model`` with a language modeling head for next-token
    prediction.

    Example::

        >>> config = DeepseekV3Config()
        >>> model = DeepseekV3ForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV3Config

    def __init__(self, config: DeepseekV3Config):
        """Initialize the DeepSeek V3 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekV3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("DeepseekV3ForCausalLM", "DeepseekV3Model")
