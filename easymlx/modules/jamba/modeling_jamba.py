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

"""Jamba MLX model implementation for serving and inference.

Jamba is a hybrid architecture interleaving transformer attention layers
with Mamba SSM layers. Some feed-forward layers use Mixture-of-Experts.
Attention layers use standard EasyMLX attention; Mamba layers use SSM.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchGLU
from easymlx.modules._base import BaseCausalLMModule

from .jamba_configuration import JambaConfig

CacheView = TransformerCacheView | PageCacheView


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """Compute SwiGLU activation: ``silu(gate) * x``.

    Args:
        gate: Gate tensor.
        x: Input tensor.

    Returns:
        Element-wise product of ``silu(gate)`` and ``x``.
    """
    return nn.silu(gate) * x


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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class JambaMLP(nn.Module):
    """Dense feed-forward block with SwiGLU activation.

    Attributes:
        gate_proj: Gate projection (no bias).
        up_proj: Up projection (no bias).
        down_proj: Down projection (no bias).

    Example:
        >>> config = JambaConfig(hidden_size=256, intermediate_size=512)
        >>> mlp = JambaMLP(config)
        >>> out = mlp(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: JambaConfig):
        """Initialize JambaMLP.

        Args:
            config: Jamba model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute SwiGLU forward pass.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class JambaSparseMoeBlock(nn.Module):
    """Sparse Mixture-of-Experts feed-forward block.

    Routes tokens to the top-k experts via softmax gating, then combines
    the expert outputs weighted by their routing scores.

    Attributes:
        num_experts_per_tok: Number of experts activated per token.
        router: Router linear projection.
        switch_mlp: SwitchGLU expert bank.

    Example:
        >>> config = JambaConfig(hidden_size=256, intermediate_size=512,
        ...     num_experts=4, num_experts_per_tok=2)
        >>> moe = JambaSparseMoeBlock(config)
        >>> out = moe(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: JambaConfig):
        """Initialize JambaSparseMoeBlock.

        Args:
            config: Jamba model configuration.
        """
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(config.hidden_size, config.intermediate_size, config.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to experts and compute MoE output.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of shape ``(..., hidden_size)``.
        """
        gates = self.router(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class JambaAttention(nn.Module):
    """Standard grouped-query attention for Jamba attention layers.

    Used only in layers designated as ``"attention"`` in the
    ``layers_block_type`` schedule. No RoPE is applied (positions are
    handled via absolute/relative embeddings depending on config).

    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        scale: Attention scaling factor.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        attention_performer: Attention computation backend.

    Example:
        >>> config = JambaConfig(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> attn = JambaAttention(config)
        >>> out = attn(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: JambaConfig):
        """Initialize JambaAttention.

        Args:
            config: Jamba model configuration.
        """
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.attention_performer = AttentionPerformer(
            scale=self.scale,
            attn_mechanism=getattr(config, "attn_mechanism", None),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute attention forward pass.

        Args:
            hidden_states: Input tensor of shape ``(*lead, hidden_size)``.
            mask: Attention mask or ``None``.
            cache_view: Optional KV cache view.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, hidden_size)``.
        """
        lead = hidden_states.shape[:-1]
        queries = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.o_proj(attn.reshape(*lead, -1))


# ---------------------------------------------------------------------------
# Mamba mixer (for non-attention layers)
# ---------------------------------------------------------------------------


class JambaMambaMixer(nn.Module):
    """Mamba SSM mixer used in non-attention Jamba layers.

    Implements a selective state-space model (SSM) with gated input
    projection, depthwise Conv1d, discretized A/B/C matrices, and
    SwiGLU output gating. Uses RMSNorm on delta, B, and C projections.

    Attributes:
        hidden_size: Model hidden size.
        ssm_state_size: SSM state dimensionality (``d_state``).
        conv_kernel_size: 1D convolution kernel size.
        intermediate_size: Expanded SSM dimensionality.
        time_step_rank: Rank for the delta-time projection.
        in_proj: Input projection (2x intermediate for gating).
        conv1d: Depthwise 1D convolution.
        x_proj: Projects to ``(dt_rank + 2 * d_state)``.
        dt_proj: Delta-time projection.
        A_log: Log-space A matrix.
        D: Skip connection parameter.
        out_proj: Output projection.

    Example:
        >>> config = JambaConfig(hidden_size=256, mamba_d_state=16,
        ...     mamba_d_conv=4, mamba_expand=2)
        >>> mixer = JambaMambaMixer(config)
        >>> out = mixer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: JambaConfig):
        """Initialize JambaMambaMixer.

        Args:
            config: Jamba model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias

        self.in_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=self.use_bias,
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            bias=self.use_conv_bias,
            padding=0,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = nn.RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = nn.RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

    def ssm_step(
        self,
        x: mx.array,
        A: mx.array,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Run SSM recurrence with layernorm on delta, B, and C.

        Args:
            x: Input tensor of shape ``(B, T, intermediate_size)``.
            A: Discrete-time A matrix.
            state: Previous SSM state, or ``None`` for the first step.

        Returns:
            Tuple of ``(output, final_state)`` where output has shape
            ``(B, T, intermediate_size)`` and final_state has shape
            ``(B, intermediate_size, d_state)``.
        """
        T = x.shape[1]
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            [
                self.time_step_rank,
                self.time_step_rank + self.ssm_state_size,
            ],
            axis=-1,
        )
        delta = self.dt_layernorm(delta)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        delta = nn.softplus(self.dt_proj(delta))

        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, -2)
        dtA = mx.exp(mx.expand_dims(delta, -1) * A)

        for t in range(T):
            if state is not None:
                new_state_t = state * dtA[:, t] + new_state[:, t]
                new_state = new_state.at[:, t].add(new_state_t - new_state[:, t])
                state = new_state_t
            else:
                state = new_state[:, t]

        y = (new_state @ mx.expand_dims(C, -1)).squeeze(-1)
        y = y + D * x
        return y, new_state[:, -1]

    def _process_sequence(
        self,
        x: mx.array,
    ) -> mx.array:
        """Process a full sequence through the Mamba mixer.

        Args:
            x: Input tensor of shape ``(B, L, hidden_size)``.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
        """
        xz = self.in_proj(x)
        x_branch, z = xz.split(indices_or_sections=2, axis=-1)

        K = self.conv_kernel_size
        x_full = mx.pad(x_branch, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        x_branch = nn.silu(conv_out)

        A = -mx.exp(self.A_log)
        y, _state = self.ssm_step(x_branch, A)
        output = self.out_proj(_swiglu(z, y))
        return output

    def __call__(self, x: mx.array) -> mx.array:
        """Run the Mamba mixer.

        Args:
            x: Input tensor of shape ``(B, L, hidden_size)``.

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
        """
        return self._process_sequence(x)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class JambaDecoderLayer(nn.Module):
    """Single Jamba decoder layer -- either attention or Mamba.

    The layer type is determined at construction time. Attention layers
    use standard GQA attention with KV caching. Mamba layers use the
    SSM mixer. The feed-forward block is either dense MLP or sparse MoE
    based on the layer index and config periods.

    Attributes:
        is_attn: Whether this layer uses attention (vs. Mamba SSM).
        self_attn: Attention module (present only for attention layers).
        mamba: Mamba mixer (present only for SSM layers).
        feed_forward: MLP or MoE feed-forward block.
        input_layernorm: RMSNorm before the mixing block.
        pre_ff_layernorm: RMSNorm before the feed-forward block.

    Example:
        >>> config = JambaConfig(hidden_size=256, num_attention_heads=8,
        ...     num_key_value_heads=4)
        >>> layer = JambaDecoderLayer(config, layer_type="attention", layer_idx=0)
        >>> out = layer(mx.zeros((1, 10, 256)))
        >>> out.shape
        [1, 10, 256]
    """

    def __init__(self, config: JambaConfig, layer_type: str, layer_idx: int):
        """Initialize JambaDecoderLayer.

        Args:
            config: Jamba model configuration.
            layer_type: Either ``"attention"`` or ``"mamba"``.
            layer_idx: Layer index for MoE scheduling.
        """
        super().__init__()
        self.is_attn = layer_type == "attention"

        if self.is_attn:
            self.self_attn = JambaAttention(config)
        else:
            self.mamba = JambaMambaMixer(config)

        # MoE or dense feed-forward
        use_moe = config.num_experts > 1 and (layer_idx + config.expert_layer_offset) % config.expert_layer_period == 0
        if use_moe:
            self.feed_forward = JambaSparseMoeBlock(config)
        else:
            self.feed_forward = JambaMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Execute the decoder layer.

        Args:
            x: Input tensor of shape ``(B, L, hidden_size)``.
            mask: Attention mask (ignored for Mamba layers).
            cache_view: KV cache view (ignored for Mamba layers).
            cache_metadata: Paged-attention metadata (ignored for Mamba layers).

        Returns:
            Output tensor of shape ``(B, L, hidden_size)``.
        """
        normed = self.input_layernorm(x)
        if self.is_attn:
            h = self.self_attn(
                normed,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        else:
            h = self.mamba(normed)
        r = x + h
        out = r + self.feed_forward(self.pre_ff_layernorm(r))
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.BASE_MODULE, config=JambaConfig, model_type="jamba")
class JambaModel(EasyMLXBaseModule):
    """Base Jamba hybrid attention/SSM model (no LM head).

    Attention layers use standard KV caches via ``cache_views``.
    Mamba layers use internal SSM state (no cache needed for inference
    without incremental generation).
    """

    config_class = JambaConfig

    def __init__(self, config: JambaConfig):
        """Initialize JambaModel.

        Args:
            config: Jamba model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [JambaDecoderLayer(config, t, idx) for idx, t in enumerate(config.layers_block_type)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Track which layers are attention for cache routing
        self._attn_layer_indices = [i for i, layer in enumerate(self.layers) if layer.is_attn]

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Jamba hybrid forward pass.

        Cache views are routed only to attention layers; Mamba layers
        use internal SSM state and do not need external caching.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: KV cache views (only for attention layers).
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        # Build attention mask for attention layers
        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(
                    attention_mask_arr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )

        # 1:1 cache mapping: every layer gets its own cache view.
        # Non-attention layers accept but ignore the cache_view.
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask if layer.is_attn else None,
                cache_view=layer_cache,
                cache_metadata=cache_metadata if layer.is_attn else None,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream Jamba checkpoint weights.

        Performs three transformations:
        1. Transposes Mamba ``conv1d.weight`` from ``(C, 1, K)`` to ``(C, K, 1)``.
        2. Drops tied ``lm_head.weight`` if ``tie_word_embeddings`` is True.
        3. Stacks per-expert MLP weights into SwitchGLU format.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary ready for model loading.
        """
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                sanitized[k] = v.moveaxis(2, 1)
            else:
                sanitized[k] = v

        if getattr(self.config, "tie_word_embeddings", True):
            sanitized.pop("lm_head.weight", None)

        # Stack individual expert weights into SwitchGLU format
        for layer_idx in range(self.config.num_hidden_layers):
            base = f"model.layers.{layer_idx}.feed_forward"
            for proj in ["gate_proj", "down_proj", "up_proj"]:
                for name in ["weight", "bias", "scales", "biases"]:
                    expert_tensors = [
                        sanitized.pop(f"{base}.experts.{e}.{proj}.{name}")
                        for e in range(self.config.num_experts)
                        if f"{base}.experts.{e}.{proj}.{name}" in sanitized
                    ]
                    if expert_tensors:
                        sanitized[f"{base}.switch_mlp.{proj}.{name}"] = mx.stack(expert_tensors)

        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=JambaConfig, model_type="jamba")
class JambaForCausalLM(BaseCausalLMModule[JambaModel, JambaConfig]):
    """Jamba hybrid attention/SSM model with a causal language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = JambaConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)
        >>> model = JambaForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 1000]
    """

    config_class = JambaConfig

    def __init__(self, config: JambaConfig):
        """Initialize JambaForCausalLM.

        Args:
            config: Jamba model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=JambaModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Delegate sanitization to the inner JambaModel.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """
        return self.base_model.sanitize(weights)


__all__ = ("JambaForCausalLM", "JambaModel")
