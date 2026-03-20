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

"""DeepSeek V32 MLX model implementation for serving and inference.

This module provides the DeepSeek V32 architecture on MLX, the most
advanced variant featuring full MLA with absorbed projections,
Indexer-based sparse attention for long-context efficiency, and
advanced MoE with noaux_tc routing.
"""

from __future__ import annotations

import math
from typing import Any

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

from .deepseek_v32_configuration import DeepseekV32Config

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
            transpose: If ``True``, multiplies by transposed weight. Defaults to ``True``.

        Returns:
            Transformed tensor.
        """
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight


class Indexer(nn.Module):
    """Sparse attention indexer for DeepSeek V32.

    Computes top-k attention indices to enable sparse attention patterns,
    reducing computational cost for long sequences. When the sequence
    length is shorter than index_topk, returns None (full attention).
    """

    def __init__(self, config: DeepseekV32Config):
        """Initialize the Indexer.

        Args:
            config: Model configuration with indexer hyperparameters
                (``index_head_dim``, ``index_n_heads``, ``index_topk``).
        """
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.rope = get_rope(
            dims=config.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=True,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: mx.array | None,
        cache_view: Any | None = None,
    ):
        """Compute top-k KV position indices for sparse attention.

        Computes lightweight attention scores between a low-cost query
        projection and a single-head key, weighted by a learned importance
        projection. Returns the top-k indices, or ``None`` if the KV length
        is within ``index_topk`` (meaning full attention is cheaper).

        Args:
            x: Hidden states of shape ``(batch, seq_len, hidden_size)``.
            qr: Query LoRA intermediate of shape ``(batch, seq_len, q_lora_rank)``.
            mask: Optional boolean mask for the indexer scores.
            cache_view: Optional cache view for the indexer keys.

        Returns:
            Top-k indices of shape ``(batch, 1, 1, topk)`` for sparse KV
            selection, or ``None`` if full attention should be used.
        """
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        q_pe, q_nope = mx.split(q, [self.rope_head_dim], axis=-1)

        offset = cache_view.offset if cache_view is not None else 0

        q_pe = self.rope(q_pe, offset=offset)
        q = mx.concatenate([q_pe, q_nope], axis=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))
        k_pe, k_nope = mx.split(k, [self.rope_head_dim], axis=-1)
        k_pe = self.rope(k_pe, offset=offset)
        k = mx.concatenate([k_pe, k_nope], axis=-1)

        if cache_view is not None:
            k, _, _ = cache_view.concatenate_to_cache(k, mx.zeros([b, 1, s, 0]))

        if k.shape[2] <= self.index_topk:
            return None

        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        weights = weights.swapaxes(-1, -2)[..., None]
        scores = scores * weights
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[..., -self.index_topk :]


class DeepseekV32Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek V32 with sparse indexing.

    Extends the V3 MLA with an Indexer module that selects the most
    relevant KV positions for sparse attention on long sequences.
    """

    def __init__(self, config: DeepseekV32Config):
        """Initialize DeepSeek V32 MLA with sparse indexing.

        Combines absorbed MLA from V3 with an Indexer module that identifies
        the most relevant KV positions for long sequences, enabling sparse
        attention that reduces computation from O(n^2) to O(n * topk).

        Args:
            config: Model configuration.
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

        self.indexer = Indexer(config)
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
        indexer_cache_view: CacheView | None = None,
    ) -> mx.array:
        """Run the sparse MLA forward pass.

        First runs the Indexer to select top-k KV positions. During decode,
        the sparse indices are used to gather a subset of KV latents. During
        prefill, a sparse boolean mask is constructed from the indices.
        Then performs the absorbed MLA attention on the (possibly sparse) KV.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for MLA latent and RoPE key.
            cache_metadata: Paged-attention metadata.
            indexer_cache_view: Separate cache view for the Indexer keys.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        B, L, _D = hidden_states.shape

        qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(qr)

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

        # Run the indexer for sparse attention
        topk_indices = self.indexer(
            hidden_states, qr, mask if isinstance(mask, mx.array) else None, cache_view=indexer_cache_view
        )

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, (*idx.shape[:-1], kv_latent.shape[-1])),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, (*idx.shape[:-1], k_pe.shape[-1])),
                    axis=2,
                )
                mask = None
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(sparse_mask, topk_indices, mx.array(True), axis=-1)
                if mask is not None and isinstance(mask, mx.array):
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

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


class DeepseekV32MLP(nn.Module):
    """SiLU-gated feed-forward MLP for DeepSeek V32.

    Implements ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``.
    """

    def __init__(self, config: DeepseekV32Config, intermediate_size: int | None = None):
        """Initialize the MLP.

        Args:
            config: Model configuration.
            intermediate_size: Override for intermediate dimension, or ``None``.
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


class DeepseekV32MoE(nn.Module):
    """Mixture-of-Experts block for DeepSeek V32 with noaux_tc routing.

    Identical routing strategy to V3 with sigmoid scoring and score correction bias.
    """

    def __init__(self, config: DeepseekV32Config):
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
            self.shared_experts = DeepseekV32MLP(config, intermediate_size=shared_intermediate)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens through experts and aggregate.

        Args:
            hidden_states: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output of shape ``(..., hidden_size)``.
        """
        inds, scores = self.router(hidden_states)
        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y


class DeepseekV32DecoderLayer(nn.Module):
    """Single DeepSeek V32 decoder layer.

    Pre-norm residual with sparse MLA attention and MoE/dense MLP.
    Passes an additional ``indexer_cache_view`` for the Indexer module.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        """Initialize a decoder layer.

        Args:
            config: Model configuration.
            layer_idx: Zero-based layer index.
        """
        super().__init__()
        self.self_attn = DeepseekV32Attention(config)
        self.mlp = (
            DeepseekV32MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV32MLP(config)
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
        indexer_cache_view: CacheView | None = None,
    ) -> mx.array:
        """Run the decoder layer forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.
            indexer_cache_view: Cache view for the Indexer keys.

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
            indexer_cache_view=indexer_cache_view,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=DeepseekV32Config, model_type="deepseek_v32")
class DeepseekV32Model(EasyMLXBaseModule):
    """Base DeepSeek V32 transformer model with sparse MLA.

    The most advanced DeepSeek variant, combining absorbed MLA projections
    with an Indexer-based sparse attention mechanism for long-context
    efficiency. For sequences longer than ``index_topk``, only the most
    relevant KV positions are attended to.

    Attributes:
        config_class: The configuration class (``DeepseekV32Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``DeepseekV32DecoderLayer`` instances.
        norm: Final RMS normalization.

    Example::

        >>> config = DeepseekV32Config()
        >>> model = DeepseekV32Model(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV32Config

    def __init__(self, config: DeepseekV32Config):
        """Initialize the DeepSeek V32 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV32DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
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
        """Transform checkpoint weights for DeepSeek V32.

        Performs the following transformations:

        1. **MTP removal**: Filters out multi-token prediction layers
           (layers beyond ``num_hidden_layers``).
        2. **Expert stacking**: Stacks per-expert weights into ``switch_mlp``.
        3. **Router remapping**: Renames gate weights and score correction bias.
        4. **kv_b_proj splitting**: Splits into ``embed_q`` and ``unembed_out``
           MultiLinear weights, with optional re-quantization.
        5. **Rotary filter**: Removes ``inv_freq`` keys.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        # Remove multi-token prediction layers
        mpt_layer = self.config.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit() and int(parts[2]) >= mpt_layer:
                continue
            new_weights[k] = v
        weights = new_weights

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


@register_module(task_type=TaskType.CAUSAL_LM, config=DeepseekV32Config, model_type="deepseek_v32")
class DeepseekV32ForCausalLM(BaseCausalLMModule[DeepseekV32Model, DeepseekV32Config]):
    """DeepSeek V32 causal language model.

    Wraps ``DeepseekV32Model`` with a language modeling head.

    Example::

        >>> model = DeepseekV32ForCausalLM(DeepseekV32Config())
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = DeepseekV32Config

    def __init__(self, config: DeepseekV32Config):
        """Initialize the causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekV32Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("DeepseekV32ForCausalLM", "DeepseekV32Model")
