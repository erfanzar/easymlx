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

"""Nemotron-H MLX model implementation for serving and inference.

Nemotron-H is a hybrid architecture interleaving Mamba2 SSM, attention,
dense MLP, and MoE blocks. Block types are controlled by the
``hybrid_override_pattern`` config field: ``"M"`` for Mamba, ``"*"`` for
attention, ``"-"`` for MLP, and ``"E"`` for MoE.

Only attention blocks use the EasyMLX cache infrastructure. Mamba state
is not externally cached in this implementation (stateless per call).
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

from .nemotron_h_configuration import NemotronHConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert values to int32 mx.array, or return None.

    Args:
        values (mx.ArrayLike | None): Input values to convert.

    Returns:
        mx.array | None: Int32 array or None if input is None.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class NemotronHAttention(nn.Module):
    """Standard multi-head attention block for Nemotron-H.

    Used in layers where the hybrid pattern character is ``"*"``.

    Attributes:
        hidden_size: Model hidden dimensionality.
        num_heads: Number of query attention heads.
        head_dim: Per-head dimensionality.
        num_kv_heads: Number of key/value heads for GQA.
        scale: Attention scaling factor.

    Example:
        >>> config = NemotronHConfig()
        >>> attn = NemotronHAttention(config)
        >>> out = attn(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronHConfig):
        """Initialize Nemotron-H attention.

        Args:
            config (NemotronHConfig): Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.attention_performer = AttentionPerformer(scale=self.scale)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute attention forward pass.

        Args:
            x (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask.
            cache_view (CacheView | None): KV cache view.
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)``.
        """
        lead = x.shape[:-1]

        q = self.q_proj(x).reshape(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(*lead, self.num_kv_heads, self.head_dim)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.o_proj(attn.reshape(*lead, -1))


class NemotronHMLP(nn.Module):
    """Dense MLP block with ReLU^2 activation for Nemotron-H.

    Used in layers where the hybrid pattern character is ``"-"``, and
    also as the shared expert within MoE blocks.

    Attributes:
        up_proj: Up-projection linear layer.
        down_proj: Down-projection linear layer.

    Example:
        >>> config = NemotronHConfig()
        >>> mlp = NemotronHMLP(config)
        >>> out = mlp(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronHConfig, intermediate_size: int | None = None):
        """Initialize Nemotron-H MLP.

        Args:
            config (NemotronHConfig): Model configuration.
            intermediate_size (int | None): Override intermediate dimensionality.
                Defaults to ``config.intermediate_size``.
        """
        super().__init__()
        intermediate = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Compute ReLU^2 MLP forward pass.

        Args:
            x (mx.array): Input of shape ``(..., hidden_size)``.

        Returns:
            mx.array: Output of shape ``(..., hidden_size)``.
        """
        return self.down_proj(nn.relu(self.up_proj(x)) ** 2)


class NemotronHMoE(nn.Module):
    """Mixture-of-Experts block for Nemotron-H with sigmoid gating.

    Used in layers where the hybrid pattern character is ``"E"``. Supports
    optional shared experts and routed scaling factor.

    Attributes:
        config: Model configuration.
        num_experts_per_tok: Number of experts activated per token.
        switch_mlp: SwitchGLU expert bank.
        gate: Router linear projection.
        shared_experts: Optional shared expert MLP.

    Example:
        >>> config = NemotronHConfig(n_routed_experts=8, num_experts_per_tok=2)
        >>> moe = NemotronHMoE(config)
        >>> out = moe(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronHConfig):
        """Initialize Nemotron-H MoE block.

        Args:
            config (NemotronHConfig): Model configuration with expert settings.
        """
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        n_routed = config.n_routed_experts or 4

        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size or config.intermediate_size,
            n_routed,
        )
        self.gate = nn.Linear(config.hidden_size, n_routed, bias=False)

        if config.n_shared_experts is not None:
            intermediate = config.moe_shared_expert_intermediate_size or config.intermediate_size
            self.shared_experts = NemotronHMLP(config, intermediate_size=intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to experts with sigmoid gating.

        Args:
            x (mx.array): Input of shape ``(B, L, D)``.

        Returns:
            mx.array: Expert-weighted output of shape ``(B, L, D)``.
        """
        gates = self.gate(x)
        scores = mx.sigmoid(gates.astype(mx.float32))
        k = self.num_experts_per_tok or 2
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        selected_scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.config.routed_scaling_factor is not None:
            selected_scores = selected_scores * self.config.routed_scaling_factor

        y = self.switch_mlp(x, inds)
        y = (y * selected_scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class NemotronHBlock(nn.Module):
    """Single Nemotron-H block with configurable block type.

    The block type determines which mixer module is used:
    ``"*"`` for attention, ``"-"`` for MLP, ``"E"`` for MoE,
    ``"M"`` for Mamba (falls back to MLP in this implementation).

    Attributes:
        block_type: Character identifying the block type.
        norm: Pre-norm RMSNorm.
        mixer: The block's mixer module (attention, MLP, or MoE).

    Example:
        >>> config = NemotronHConfig()
        >>> block = NemotronHBlock(config, "*")
        >>> out = block(mx.zeros((1, 128, 4096)))
    """

    def __init__(self, config: NemotronHConfig, block_type: str):
        """Initialize Nemotron-H block.

        Args:
            config (NemotronHConfig): Model configuration.
            block_type (str): One of ``"*"``, ``"-"``, ``"E"``, or ``"M"``.
        """
        super().__init__()
        self.block_type = block_type
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if block_type == "*":
            self.mixer = NemotronHAttention(config)
        elif block_type == "-":
            self.mixer = NemotronHMLP(config)
        elif block_type == "E":
            self.mixer = NemotronHMoE(config)
        else:
            self.mixer = NemotronHMLP(config)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the hybrid block.

        Only attention blocks (``"*"``) receive mask and cache arguments.
        Other block types ignore them.

        Args:
            x (mx.array): Input of shape ``(B, L, D)``.
            mask (mx.array | str | None): Attention mask (attention blocks only).
            cache_view (CacheView | None): KV cache view (attention blocks only).
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Output of shape ``(B, L, D)`` with residual connection.
        """
        hidden_states = self.norm(x)
        if self.block_type == "*":
            hidden_states = self.mixer(
                hidden_states,
                mask=mask,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
        else:
            hidden_states = self.mixer(hidden_states)
        return x + hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=NemotronHConfig, model_type="nemotron_h")
class NemotronHModel(EasyMLXBaseModule):
    """Base Nemotron-H hybrid transformer model.

    Interleaves attention, MLP, MoE, and Mamba blocks according to
    ``hybrid_override_pattern``. Only attention blocks use the KV cache.

    Attributes:
        config_class: Associated configuration class (``NemotronHConfig``).
        embed_tokens: Token embedding layer.
        layers: List of hybrid blocks.
        norm: Final RMSNorm.

    Example:
        >>> config = NemotronHConfig()
        >>> model = NemotronHModel(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronHConfig

    def __init__(self, config: NemotronHConfig):
        """Initialize Nemotron-H base model.

        Args:
            config (NemotronHConfig): Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronHBlock(config, bt) for bt in config.hybrid_override_pattern]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self._attn_layer_indices = [i for i, bt in enumerate(config.hybrid_override_pattern) if bt == "*"]

    @property
    def num_attn_layers(self) -> int:
        """Return the number of attention layers in the hybrid model.

        Returns:
            int: Count of attention blocks (``"*"`` pattern entries).
        """
        return len(self._attn_layer_indices)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through the Nemotron-H base model.

        Cache views are mapped only to attention blocks; non-attention
        blocks receive no cache.

        Args:
            input_ids (mx.ArrayLike): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (mx.ArrayLike | None): Optional attention mask.
            input_embeddings (mx.array | None): Pre-computed embeddings.
            cache_views (list[CacheView] | None): KV cache views for
                attention blocks only (length must match ``num_attn_layers``).
            cache_metadata (PageMetadata | None): Paged attention metadata.

        Returns:
            mx.array: Normalized hidden states of shape ``(B, L, D)``.
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
        """Handle conv1d weight transposition and expert stacking.

        Transposes conv1d weights from ``(out, in, kernel)`` to
        ``(out, kernel, in)`` format, stacks per-expert MoE weights,
        and removes rotary embedding buffers.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Sanitized weight dictionary.
        """
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        n_routed = self.config.n_routed_experts
        if n_routed is not None:
            for layer_idx in range(len(self.layers)):
                prefix = f"backbone.layers.{layer_idx}.mixer"
                for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                    if f"{prefix}.experts.0.{m}.weight" in weights:
                        to_join = [weights.pop(f"{prefix}.experts.{e}.{m}.weight") for e in range(n_routed)]
                        weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=NemotronHConfig, model_type="nemotron_h")
class NemotronHForCausalLM(BaseCausalLMModule[NemotronHModel, NemotronHConfig]):
    """Nemotron-H causal language model with LM head.

    Attributes:
        config_class: Associated configuration class (``NemotronHConfig``).

    Example:
        >>> config = NemotronHConfig()
        >>> model = NemotronHForCausalLM(config)
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = NemotronHConfig

    def __init__(self, config: NemotronHConfig):
        """Initialize Nemotron-H causal LM.

        Args:
            config (NemotronHConfig): Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=NemotronHModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap ``backbone.*`` keys to ``model.*`` and delegate sanitization.

        Args:
            weights (dict[str, mx.array]): Raw checkpoint weight dictionary.

        Returns:
            dict[str, mx.array]: Sanitized weight dictionary.
        """

        remapped = {}
        for key, value in weights.items():
            if key.startswith("backbone."):
                new_key = "model." + key[len("backbone.") :]
                remapped[new_key] = value
            else:
                remapped[key] = value
        remapped = self.model.sanitize(remapped)
        return super().sanitize(remapped)


__all__ = ("NemotronHForCausalLM", "NemotronHModel")
