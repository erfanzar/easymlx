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

"""DBRX MLX implementation (serving/inference only).

Structure mirrors the upstream DBRX architecture:
  DBRXConfig -> DBRXAttention -> DBRXMLP -> DBRXSparseMoeBlock
  -> DBRXNormAttnNorm -> DBRXDecoderLayer -> DBRXModel -> DBRXForCausalLM

DBRX uses a NormAttnNorm structure with two LayerNorms per layer,
SwiGLU MoE experts, and QKV value clipping.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .dbrx_configuration import DBRXConfig

CacheView = TransformerCacheView | PageCacheView


class DBRXAttention(nn.Module):
    """Multi-head attention with fused QKV and QKV value clipping for DBRX.

    Uses a single fused ``Wqkv`` projection for Q, K, V, with optional
    value clipping to stabilize training/inference. Supports GQA via
    ``attn_config["kv_n_heads"]``.

    Attributes:
        num_heads: Number of query attention heads.
        d_model: Model hidden dimensionality.
        head_dim: Per-head dimensionality.
        num_kv_heads: Number of KV heads for GQA.
        clip_qkv: Optional QKV clipping value.
        scale: Attention scaling factor.
        Wqkv: Fused QKV linear projection.
        out_proj: Output linear projection.
        rope: Rotary position embedding module.
        attention_performer: Attention computation backend.

    Example::

        >>> config = DBRXConfig(d_model=4096, n_heads=32)
        >>> attn = DBRXAttention(config)
    """

    def __init__(self, config: DBRXConfig):
        """Initialize DBRX attention.

        Args:
            config: Model configuration with ``d_model``, ``n_heads``,
                and ``attn_config`` containing ``kv_n_heads``, ``clip_qkv``,
                and ``rope_theta``.
        """
        super().__init__()
        self.num_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.num_kv_heads = config.attn_config["kv_n_heads"]
        self.clip_qkv = config.attn_config.get("clip_qkv", None)
        rope_theta = config.attn_config.get("rope_theta", 500000.0)
        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            config.d_model,
            (self.num_kv_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = get_rope(
            dims=self.head_dim,
            base=rope_theta,
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
        """Run the fused QKV attention forward pass with optional clipping.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        lead = hidden_states.shape[:-1]
        qkv = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv = mx.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)

        splits = [
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim + self.num_kv_heads * self.head_dim,
        ]
        queries, keys, values = mx.split(qkv, splits, axis=-1)

        q = queries.reshape(*lead, self.num_heads, self.head_dim)
        k = keys.reshape(*lead, self.num_kv_heads, self.head_dim)
        v = values.reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.out_proj(attn.reshape(*lead, -1))


class DBRXNormAttnNorm(nn.Module):
    """NormAttnNorm block: two LayerNorms per layer (pre-attention, pre-MLP).

    First normalizes the input, applies attention with residual, then
    normalizes the result for the MLP stage. Returns both the residual
    and the pre-MLP normalized output.

    Attributes:
        norm_1: Pre-attention LayerNorm.
        norm_2: Pre-MLP LayerNorm.
        attn: DBRX attention module.
    """

    def __init__(self, config: DBRXConfig):
        """Initialize the NormAttnNorm block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DBRXAttention(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run NormAttnNorm forward pass.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Tuple of ``(residual, normalized_for_mlp)`` where both have
            shape ``(batch, seq_len, d_model)``.
        """
        h = self.attn(
            self.norm_1(hidden_states),
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        x = h + hidden_states
        return x, self.norm_2(x)


class DBRXMLP(nn.Module):
    """SwiGLU feed-forward MLP for a single DBRX expert.

    Uses ``w2(silu(w1(x)) * v1(x))`` where ``v1`` is the gate projection
    and ``w1`` is the up projection.

    Attributes:
        v1: Linear gate projection.
        w1: Linear up projection.
        w2: Linear down projection.
    """

    def __init__(self, d_model: int, ffn_dim: int):
        """Initialize a single DBRX expert MLP.

        Args:
            d_model: Model hidden dimensionality.
            ffn_dim: Expert intermediate dimensionality.
        """
        super().__init__()
        self.v1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SwiGLU expert MLP.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        return self.w2(nn.silu(self.w1(x)) * self.v1(x))


class DBRXRouter(nn.Module):
    """Expert routing for DBRX MoE.

    Projects hidden states to expert logits using a bias-free linear layer.

    Attributes:
        layer: Linear projection from ``d_model`` to ``num_experts``.
    """

    def __init__(self, d_model: int, num_experts: int):
        """Initialize the DBRX router.

        Args:
            d_model: Model hidden dimensionality.
            num_experts: Total number of experts.
        """
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute expert routing logits.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            Routing logits of shape ``(..., num_experts)``.
        """
        return self.layer(x)


class DBRXSparseMoeBlock(nn.Module):
    """Sparse MoE block with top-k softmax routing for DBRX.

    Routes each token to the top-k experts, normalizes routing
    weights by L1 norm, and computes a weighted sum of expert outputs.

    Attributes:
        d_model: Model hidden dimensionality.
        ffn_dim: Expert intermediate dimensionality.
        num_experts: Total number of experts.
        num_experts_per_tok: Number of experts activated per token.
        router: Expert routing module.
        experts: List of individual ``DBRXMLP`` expert modules.

    Example::

        >>> config = DBRXConfig(d_model=4096)
        >>> moe = DBRXSparseMoeBlock(config)
    """

    def __init__(self, config: DBRXConfig):
        """Initialize the DBRX sparse MoE block.

        Args:
            config: Model configuration with ``ffn_config`` containing
                ``ffn_hidden_size``, ``moe_num_experts``, and ``moe_top_k``.
        """
        super().__init__()
        self.d_model = config.d_model
        self.ffn_dim = config.ffn_config["ffn_hidden_size"]
        self.num_experts = config.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = config.ffn_config["moe_top_k"]

        self.router = DBRXRouter(self.d_model, self.num_experts)
        self.experts = [DBRXMLP(self.d_model, self.ffn_dim) for _ in range(self.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to top-k experts and compute MoE output.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            MoE output tensor of the same shape.
        """
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist(), strict=False):
            yt = mx.stack([self.experts[e](xt) for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)
        y = mx.stack(y, axis=0)

        return y.reshape(orig_shape)


class DBRXDecoderLayer(nn.Module):
    """Single DBRX decoder layer with NormAttnNorm + MoE FFN.

    Combines the NormAttnNorm block (two norms + attention + residual)
    with the sparse MoE feed-forward block.

    Attributes:
        ffn: Sparse MoE feed-forward block.
        norm_attn_norm: NormAttnNorm block.
    """

    def __init__(self, config: DBRXConfig):
        """Initialize the DBRX decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.ffn = DBRXSparseMoeBlock(config)
        self.norm_attn_norm = DBRXNormAttnNorm(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the NormAttnNorm + MoE decoder layer forward pass.

        Args:
            hidden_states: Input tensor of shape ``(..., d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        r, h = self.norm_attn_norm(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.ffn(h) + r


@register_module(task_type=TaskType.BASE_MODULE, config=DBRXConfig, model_type="dbrx")
class DBRXModel(EasyMLXBaseModule):
    """Base DBRX transformer model for inference.

    Implements a decoder-only MoE transformer with NormAttnNorm
    structure, fused QKV projection with value clipping, SwiGLU
    MoE experts with top-k softmax routing, and bias-free LayerNorm.

    Attributes:
        config_class: The configuration class (``DBRXConfig``).
        embed_tokens: Token embedding layer.
        layers: List of ``DBRXDecoderLayer`` decoder blocks.
        norm: Final LayerNorm normalization (bias-free).

    Example::

        >>> config = DBRXConfig(vocab_size=100352, d_model=4096)
        >>> model = DBRXModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = DBRXConfig

    def __init__(self, config: DBRXConfig):
        """Initialize the DBRX base model.

        Args:
            config: Model configuration with architecture hyperparameters.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [DBRXDecoderLayer(config) for _ in range(config.n_layers)]
        self.norm = nn.LayerNorm(config.d_model, bias=False)

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
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, d_model)``.

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
        """Split fused expert weights into separate per-expert projections.

        The upstream DBRX checkpoint stores all expert weights fused
        into a single tensor per projection. This method splits them
        into individual expert weight tensors and transposes ``w2``
        (down projection) weights.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with split expert weights and
            rotary buffers removed.
        """
        num_experts = self.config.ffn_config["moe_num_experts"]
        pattern = "experts.mlp"
        new_weights = {k: v for k, v in weights.items() if pattern not in k}
        for k, v in weights.items():
            if pattern in k:
                experts = [
                    (k.replace(".mlp", f".{e}") + ".weight", sv) for e, sv in enumerate(mx.split(v, num_experts, axis=0))
                ]
                if k.endswith("w2"):
                    experts = [(s, sv.T) for s, sv in experts]
                new_weights.update(experts)

        new_weights = {k: v for k, v in new_weights.items() if "rotary_emb.inv_freq" not in k}
        return new_weights


@register_module(task_type=TaskType.CAUSAL_LM, config=DBRXConfig, model_type="dbrx")
class DBRXForCausalLM(BaseCausalLMModule[DBRXModel, DBRXConfig]):
    """DBRX model with a causal language modeling head.

    Wraps ``DBRXModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``DBRXConfig``).

    Example::

        >>> config = DBRXConfig(vocab_size=100352, d_model=4096)
        >>> model = DBRXForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = DBRXConfig

    def __init__(self, config: DBRXConfig):
        """Initialize the DBRX causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=DBRXModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("DBRXForCausalLM", "DBRXModel")
