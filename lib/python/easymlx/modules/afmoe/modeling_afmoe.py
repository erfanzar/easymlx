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

"""AFMoE MLX model implementation for serving and inference.

AFMoE features sliding and full attention patterns with gated attention,
QK-norm, pre/post MLP layer norms, MuP scaling, and a mixture-of-experts
feed-forward with grouped expert routing and shared experts.
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
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .afmoe_configuration import AfmoeConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an int32 mx.array, or return None.

    Args:
        values: Input values to convert. If ``None``, returns ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class AfmoeAttention(nn.Module):
    """Multi-head attention for AFMoE with gated output, QK-norm, and optional RoPE.

    Uses separate Q, K, V projections with RMSNorm on Q and K. Applies
    a sigmoid gate on the attention output before the output projection.
    RoPE is only applied for local (sliding window) attention layers;
    full attention layers omit RoPE.

    Attributes:
        hidden_size: Model hidden dimensionality.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA.
        head_dim: Per-head dimensionality.
        is_local_attention: Whether this layer uses sliding window attention.
        scale: Attention scaling factor (``head_dim ** -0.5``).
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
        o_proj: Linear output projection.
        q_norm: RMSNorm applied to query heads.
        k_norm: RMSNorm applied to key heads.
        gate_proj: Sigmoid gating projection applied to attention output.
        rope: Rotary position embedding (``None`` for full attention layers).
        attention_performer: Attention computation backend.

    Example::

        >>> config = AfmoeConfig(hidden_size=2048, num_attention_heads=32)
        >>> attn = AfmoeAttention(config, is_local_attention=True)
    """

    def __init__(self, config: AfmoeConfig, is_local_attention: bool = False):
        """Initialize AFMoE attention.

        Args:
            config: Model configuration containing attention hyperparameters.
            is_local_attention: If ``True``, uses RoPE for local sliding
                window attention. If ``False``, omits RoPE for full attention.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.is_local_attention = is_local_attention
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        if is_local_attention:
            self.rope = get_rope(
                dims=self.head_dim,
                base=config.rope_theta,
                traditional=False,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.rope = None

        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run gated attention forward pass with QK-norm.

        Projects input to Q, K, V, applies QK-norm and optional RoPE,
        computes attention, gates the output with sigmoid, then projects
        back to hidden size.

        Args:
            x: Input hidden states of shape ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        lead = x.shape[:-1]

        q = self.q_proj(x).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        output = attn.reshape(*lead, -1)

        gate = mx.sigmoid(self.gate_proj(x))
        output = output * gate

        return self.o_proj(output)


class AfmoeMLP(nn.Module):
    """SwiGLU feed-forward network for AFMoE.

    Uses SiLU-gated linear unit: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
    Used both as the dense MLP in early layers and as the shared expert MLP.

    Attributes:
        gate_proj: Linear gate projection.
        down_proj: Linear down projection.
        up_proj: Linear up projection.
    """

    def __init__(self, config: AfmoeConfig, intermediate_size: int | None = None):
        """Initialize the AFMoE SwiGLU MLP.

        Args:
            config: Model configuration containing ``hidden_size`` and
                ``intermediate_size``.
            intermediate_size: Override for the intermediate dimensionality.
                If ``None``, uses ``config.intermediate_size``.
        """
        super().__init__()
        dim = config.hidden_size
        hidden_dim = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SwiGLU MLP to the input.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape as input.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class AfmoeMoE(nn.Module):
    """Mixture-of-Experts block for AFMoE with grouped expert routing.

    Routes tokens to top-k experts using sigmoid or softmax scoring
    with optional grouped expert selection. Supports shared (always-active)
    experts that are added to the routed expert output.

    Attributes:
        config: Model configuration.
        num_experts_per_tok: Number of experts activated per token.
        route_norm: Whether to normalize routing probabilities.
        route_scale: Scaling factor applied to routing scores.
        score_func: Scoring function (``"sigmoid"`` or ``"softmax"``).
        n_group: Number of expert groups for grouped routing.
        topk_group: Number of top groups to keep during routing.
        router: Expert routing module.
        expert_bias: Learnable bias added to expert selection scores.
        experts: SwitchGLU fused expert module.
        shared_experts: Optional shared expert MLP (always active).

    Example::

        >>> config = AfmoeConfig(hidden_size=2048, num_experts=16)
        >>> moe = AfmoeMoE(config)
    """

    def __init__(self, config: AfmoeConfig):
        """Initialize the AFMoE MoE block.

        Args:
            config: Model configuration containing MoE hyperparameters
                such as ``num_experts``, ``num_experts_per_tok``,
                ``num_shared_experts``, and routing parameters.
        """
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.score_func = config.score_func
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.router = _AfmoeRouter(config)
        self.expert_bias = mx.zeros((config.num_experts,))
        self.experts = SwitchGLU(config.hidden_size, config.moe_intermediate_size, config.num_experts)

        if config.num_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = AfmoeMLP(config, intermediate_size=shared_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Computes routing scores, selects top-k experts per token
        (with optional grouped selection), applies expert MLPs,
        and combines outputs with routing weights.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            MoE output tensor of the same shape as input.
        """
        gates = self.router(x)

        if self.score_func == "sigmoid":
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(gates.astype(mx.float32), axis=-1)

        selection_scores = scores + self.expert_bias

        if self.n_group > 1:
            selection_scores = mx.unflatten(selection_scores, axis=-1, shape=(self.n_group, -1))
            group_scores = mx.topk(selection_scores, 2, axis=-1).sum(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            selection_scores = mx.put_along_axis(selection_scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
            selection_scores = mx.flatten(selection_scores, -2, -1)

        k = self.num_experts_per_tok
        inds = mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k]
        selected_scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.route_norm and self.num_experts_per_tok > 1:
            denominator = selected_scores.sum(axis=-1, keepdims=True)
            selected_scores = selected_scores / denominator

        selected_scores = selected_scores * self.route_scale

        y = self.experts(x, inds)
        y = (y * selected_scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.config.num_shared_experts > 0:
            y = y + self.shared_experts(x)

        return y


class _AfmoeRouter(nn.Module):
    """Router for AFMoE expert selection.

    Projects hidden states to expert logits using a single linear layer.

    Attributes:
        gate: Linear projection from hidden size to number of experts.
    """

    def __init__(self, config: AfmoeConfig):
        """Initialize the AFMoE router.

        Args:
            config: Model configuration with ``hidden_size`` and
                ``num_experts``.
        """
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute expert routing logits.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Routing logits of shape ``(..., num_experts)``.
        """
        return self.gate(x)


class AfmoeDecoderLayer(nn.Module):
    """Single AFMoE decoder layer with pre/post MLP norms.

    Uses pre-attention RMSNorm, post-attention RMSNorm on the residual,
    pre-MLP RMSNorm, and post-MLP RMSNorm. The MLP is either a dense
    SwiGLU (for early layers) or a MoE block (for later layers).

    Attributes:
        use_sliding: Whether this layer uses sliding window attention.
        self_attn: Gated attention module with QK-norm.
        mlp: Dense MLP or MoE block depending on layer index.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm after attention residual.
        pre_mlp_layernorm: RMSNorm before MLP.
        post_mlp_layernorm: RMSNorm after MLP residual.

    Example::

        >>> config = AfmoeConfig(hidden_size=2048)
        >>> layer = AfmoeDecoderLayer(config, layer_idx=0)
    """

    def __init__(self, config: AfmoeConfig, layer_idx: int, use_sliding: bool = False):
        """Initialize the AFMoE decoder layer.

        Args:
            config: Model configuration with architecture hyperparameters.
            layer_idx: Index of this layer in the stack. Layers before
                ``num_dense_layers`` use dense MLP; others use MoE.
            use_sliding: Whether to use sliding window attention.
        """
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = AfmoeAttention(config, is_local_attention=use_sliding)

        if layer_idx < config.num_dense_layers:
            self.mlp = AfmoeMLP(config)
        else:
            self.mlp = AfmoeMoE(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            hidden_states: Input hidden states of shape
                ``(batch, seq_len, hidden_size)``.
            mask: Optional attention mask.
            cache_view: Per-layer KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Output hidden states of the same shape as input.
        """
        r = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        r = self.post_attention_layernorm(r)
        h = hidden_states + r

        r = self.mlp(self.pre_mlp_layernorm(h))
        r = self.post_mlp_layernorm(r)
        return h + r


@register_module(task_type=TaskType.BASE_MODULE, config=AfmoeConfig, model_type="afmoe")
class AfmoeModel(EasyMLXBaseModule):
    """Base AFMoE transformer model with MoE feed-forward.

    Implements a decoder-only transformer with sliding/full attention
    patterns, gated attention with QK-norm, MuP input scaling, and
    mixture-of-experts feed-forward with grouped expert routing.

    Attributes:
        config_class: The configuration class (``AfmoeConfig``).
        mup_enabled: Whether MuP input embedding scaling is active.
        hidden_size: Model hidden dimensionality.
        embed_tokens: Token embedding layer.
        layers: List of ``AfmoeDecoderLayer`` decoder blocks.
        norm: Final RMS normalization applied to the last hidden state.

    Example::

        >>> config = AfmoeConfig(vocab_size=200192, hidden_size=2048)
        >>> model = AfmoeModel(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = AfmoeConfig

    def __init__(self, config: AfmoeConfig):
        """Initialize the AFMoE base model.

        Args:
            config: Model configuration containing architecture
                hyperparameters including MoE and attention settings.
        """
        super().__init__(config)
        self.mup_enabled = config.mup_enabled
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            AfmoeDecoderLayer(
                config,
                layer_idx=idx,
                use_sliding=(config.layer_types[idx] == "sliding_attention" if idx < len(config.layer_types) else False),
            )
            for idx in range(config.num_hidden_layers)
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
        """Run the transformer forward pass.

        Embeds input tokens (with optional MuP scaling), applies all
        decoder layers with optional KV caching, and returns normalized
        hidden states.

        Args:
            input_ids: Integer token IDs of shape ``(batch, seq_len)``
                or ``(seq_len,)`` (auto-batched when no paged metadata).
            attention_mask: Optional boolean/int mask of shape
                ``(batch, seq_len)``.
            input_embeddings: Pre-computed embeddings to use instead of
                ``input_ids``. Shape ``(batch, seq_len, hidden_size)``.
            cache_views: Per-layer KV cache views for autoregressive
                generation. Length must match ``num_hidden_layers``.
            cache_metadata: Paged-attention metadata for batched serving.

        Returns:
            Hidden states tensor of shape ``(batch, seq_len, hidden_size)``
            after final layer normalization.

        Raises:
            ValueError: If ``cache_views`` length does not match the
                number of layers.
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

        if self.mup_enabled:
            hidden_states = hidden_states * math.sqrt(self.hidden_size)

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
        """Stack per-expert weights into fused SwitchGLU format.

        Converts individual expert weight tensors (``experts.0.gate_proj``,
        ``experts.1.gate_proj``, etc.) into stacked tensors compatible
        with the ``SwitchGLU`` module. Also removes rotary embedding
        buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with expert weights stacked and
            rotary buffers removed.
        """
        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx < self.config.num_dense_layers:
                continue
            prefix = f"model.layers.{layer_idx}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}") for e in range(self.config.num_experts)
                        ]
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(to_join)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=AfmoeConfig, model_type="afmoe")
class AfmoeForCausalLM(BaseCausalLMModule[AfmoeModel, AfmoeConfig]):
    """AFMoE model with a causal language modeling head.

    Wraps ``AfmoeModel`` with a linear projection to vocabulary logits.

    Attributes:
        config_class: The configuration class (``AfmoeConfig``).

    Example::

        >>> config = AfmoeConfig(vocab_size=200192, hidden_size=2048)
        >>> model = AfmoeForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
        >>> output.logits.shape
        (1, 3, 200192)
    """

    config_class = AfmoeConfig

    def __init__(self, config: AfmoeConfig):
        """Initialize the AFMoE causal language model.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=AfmoeModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Apply model-level then base-level weight sanitization.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        weights = self.model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("AfmoeForCausalLM", "AfmoeModel")
