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

"""NemotronNAS MLX implementation (serving/inference only).

Structure mirrors upstream Nemotron-NAS architecture:
  NemotronNASConfig -> NemotronNASAttention -> NemotronNASMLP
  -> NemotronNASLinearReplacement -> NemotronNASDecoderLayer
  -> NemotronNASModel -> NemotronNASForCausalLM

NemotronNAS uses heterogeneous per-layer architecture where each layer
can have different attention and FFN configurations. Some layers can be
no-op or replaced with simple linear projections.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .nemotron_nas_configuration import (
    AttentionBlockConfig,
    FFNBlockConfig,
    NemotronNASConfig,
    _ffn_mult_to_intermediate_size,
)

CacheView = TransformerCacheView | PageCacheView

_ACT2FN: dict[str, tp.Callable[[mx.array], mx.array]] = {
    "silu": nn.silu,
    "relu": nn.relu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
    "gelu_fast": nn.gelu_approx,
}


class NemotronNASAttention(nn.Module):
    """GQA attention for NemotronNAS with per-layer GQA group size.

    Each layer can have a different number of KV heads determined by
    ``n_heads_in_group`` from ``AttentionBlockConfig``.

    Attributes:
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimensionality per attention head.
        scale: Attention scaling factor.

    Example:
        >>> config = NemotronNASConfig()
        >>> attn_cfg = AttentionBlockConfig(n_heads_in_group=4)
        >>> attn = NemotronNASAttention(config, attn_cfg)
        >>> out = attn(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronNASConfig, attention_config: AttentionBlockConfig):
        """Initialize NemotronNAS attention.

        Args:
            config (NemotronNASConfig): Global model configuration.
            attention_config (AttentionBlockConfig): Per-layer attention config
                with ``n_heads_in_group`` for GQA group size.
        """
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = self.num_heads // attention_config.n_heads_in_group
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=config.attention_bias)

        self.rope = get_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            traditional=False,
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
        """Compute GQA attention forward pass.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class NemotronNASMLP(nn.Module):
    """Standard gated MLP with per-layer FFN multiplier for NemotronNAS.

    The intermediate size is computed from ``ffn_mult`` using the formula
    ``round_up(2 * ffn_mult * hidden_size / 3, 256)``.

    Attributes:
        gate_proj: Gating projection.
        down_proj: Down-projection.
        up_proj: Up-projection.

    Example:
        >>> config = NemotronNASConfig()
        >>> ffn_cfg = FFNBlockConfig(ffn_mult=4.0)
        >>> mlp = NemotronNASMLP(config, ffn_cfg)
        >>> out = mlp(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronNASConfig, ffn_config: FFNBlockConfig):
        """Initialize NemotronNAS gated MLP.

        Args:
            config (NemotronNASConfig): Global model configuration.
            ffn_config (FFNBlockConfig): Per-layer FFN config with ``ffn_mult``.

        Raises:
            ValueError: If the activation function is not recognized.
        """
        super().__init__()
        dim = config.hidden_size
        hidden_dim = _ffn_mult_to_intermediate_size(ffn_config.ffn_mult, dim)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=config.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=config.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=config.mlp_bias)

        act_name = config.hidden_act
        if act_name not in _ACT2FN:
            raise ValueError(f"Unknown activation function: {act_name}")
        self._act_name = act_name

    def __call__(self, x: mx.array) -> mx.array:
        """Compute gated MLP forward pass.

        Args:
            x (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        act_fn = _ACT2FN[self._act_name]
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class NemotronNASLinearReplacement(nn.Module):
    """Simple linear layer to replace Attention or MLP blocks.

    Used in NAS-discovered architectures where certain layers are
    replaced with simple linear projections for efficiency.

    Attributes:
        linear: Linear projection (hidden_size -> hidden_size).

    Example:
        >>> replacement = NemotronNASLinearReplacement(4096, bias=False)
        >>> out = replacement(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, hidden_size: int, bias: bool):
        """Initialize linear replacement.

        Args:
            hidden_size (int): Input and output dimensionality.
            bias (bool): Whether to include bias.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        """Apply linear projection, ignoring extra keyword arguments.

        Args:
            x (mx.array): Input tensor of shape ``(..., hidden_size)``.
            **kwargs: Ignored (allows uniform interface with attention/MLP).

        Returns:
            mx.array: Output tensor of shape ``(..., hidden_size)``.
        """
        return self.linear(x)


class NemotronNASDecoderLayer(nn.Module):
    """Single NemotronNAS decoder layer with heterogeneous attention/FFN.

    Each layer independently decides whether to use full attention, linear
    replacement, or no-op for both the attention and FFN sub-layers, based
    on the layer's ``BlockConfig``.

    Attributes:
        attention_config: Per-layer attention configuration.
        ffn_config: Per-layer FFN configuration.
        self_attn: Attention module (or None if no-op).
        mlp: MLP module (or None if no-op).
        input_layernorm: Pre-attention norm (or None if no-op).
        post_attention_layernorm: Pre-MLP norm (or None if no-op).

    Example:
        >>> config = NemotronNASConfig()
        >>> layer = NemotronNASDecoderLayer(config, layer_idx=0)
        >>> out = layer(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronNASConfig, layer_idx: int):
        """Initialize NemotronNAS decoder layer.

        Args:
            config (NemotronNASConfig): Global model configuration.
            layer_idx (int): Layer index for looking up ``block_configs``.
        """
        super().__init__()
        block_config = config.block_configs[layer_idx]
        self.attention_config = block_config.attention
        self.ffn_config = block_config.ffn

        # Attention block
        if self.attention_config.no_op:
            self.self_attn = None
            self.input_layernorm = None
            self._has_attn = False
        elif self.attention_config.replace_with_linear:
            self.self_attn = NemotronNASLinearReplacement(config.hidden_size, config.attention_bias)
            self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._has_attn = True
        else:
            self.self_attn = NemotronNASAttention(config, self.attention_config)
            self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._has_attn = True

        # FFN block
        if self.ffn_config.no_op:
            self.mlp = None
            self.post_attention_layernorm = None
            self._has_mlp = False
        elif self.ffn_config.replace_with_linear:
            self.mlp = NemotronNASLinearReplacement(config.hidden_size, config.mlp_bias)
            self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._has_mlp = True
        else:
            self.mlp = NemotronNASMLP(config, self.ffn_config)
            self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._has_mlp = True

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the heterogeneous decoder layer.

        Skips attention and/or MLP sub-layers when configured as no-op.
        Linear replacement layers ignore mask/cache arguments.

        Args:
            hidden_states (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        # Attention part
        if self._has_attn:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            if self.attention_config.replace_with_linear:
                attn_out = self.self_attn(hidden_states)
            else:
                attn_out = self.self_attn(
                    hidden_states,
                    mask=mask,
                    cache_view=cache_view,
                    cache_metadata=cache_metadata,
                )
            hidden_states = residual + attn_out

        # MLP part
        if self._has_mlp:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(hidden_states)

        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=NemotronNASConfig, model_type="nemotron-nas")
class NemotronNASModel(EasyMLXBaseModule):
    """Base NemotronNAS transformer model with heterogeneous layers.

    Each layer can have different attention (full/linear/no-op) and FFN
    (full/linear/no-op) configurations. Cache views are mapped only to
    layers that have active attention sub-layers.

    Attributes:
        config_class: Associated configuration class (``NemotronNASConfig``).
        embed_tokens: Token embedding layer.
        layers: List of heterogeneous decoder layers.
        norm: Final RMSNorm.

    Example:
        >>> config = NemotronNASConfig()
        >>> model = NemotronNASModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronNASConfig

    def __init__(self, config: NemotronNASConfig):
        """Initialize NemotronNAS base model.

        Args:
            config (NemotronNASConfig): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronNASDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
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
        """Forward pass through the NemotronNAS base model.

        Cache views are mapped only to layers with active attention.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings.
            cache_views (list[CacheView] | None): KV cache views for
                attention layers only.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Normalized hidden states of shape ``(B, L, D)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of attention layers.
        """
        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        # Count attention layers for cache mapping
        attn_layer_indices = [i for i, layer in enumerate(self.layers) if layer._has_attn]

        if cache_views is not None and len(cache_views) != len(attn_layer_indices):
            raise ValueError(
                f"cache_views length ({len(cache_views)}) must match number of "
                f"attention layers ({len(attn_layer_indices)})."
            )

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        cache_idx = 0
        for layer in self.layers:
            if layer._has_attn and cache_views is not None:
                layer_cache = cache_views[cache_idx]
                cache_idx += 1
            else:
                layer_cache = None
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Handle tie_word_embeddings and remove rotary inv_freq buffers.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Filtered weight dictionary.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}


@register_module(task_type=TaskType.CAUSAL_LM, config=NemotronNASConfig, model_type="nemotron-nas")
class NemotronNASForCausalLM(BaseCausalLMModule[NemotronNASModel, NemotronNASConfig]):
    """NemotronNAS causal language model with LM head.

    Attributes:
        config_class: Associated configuration class (``NemotronNASConfig``).

    Example:
        >>> config = NemotronNASConfig()
        >>> model = NemotronNASForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronNASConfig

    def __init__(self, config: NemotronNASConfig):
        """Initialize NemotronNAS causal LM.

        Args:
            config (NemotronNASConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=NemotronNASModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )


__all__ = ("NemotronNASForCausalLM", "NemotronNASModel")
