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

"""Phixtral MLX model implementation for serving and inference.

Structure mirrors EasyDeL's phixtral:
  PhixtralConfig -> PhixtralRoPEAttention -> PhixtralMOE
  -> PhixtralParallelBlock -> PhixtralModel -> PhixtralForCausalLM

Parallel attention+MoE architecture with partial RoPE and SwitchMLP.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.linears import SwitchMLP
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .phixtral_configuration import PhixtralConfig

CacheView = TransformerCacheView | PageCacheView


class PhixtralRoPEAttention(nn.Module):
    """Attention with partial rotary positional embeddings for Phixtral.

    Uses a fused QKV projection and partial RoPE.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimensionality per attention head.
        scale: Scaling factor for attention logits.
    """

    def __init__(self, config: PhixtralConfig):
        """Initialize Phixtral attention layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        dims = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = dims // self.num_heads
        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

        self.rope = get_rope(
            dims=config.rotary_dim,
            base=10000.0,
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
        """Compute multi-head attention with partial RoPE.

        Args:
            hidden_states: Input tensor of shape ``(*lead, model_dim)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(*lead, model_dim)``.
        """
        lead = hidden_states.shape[:-1]
        qkv = self.Wqkv(hidden_states)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(*lead, self.num_heads, self.head_dim)
        k = k.reshape(*lead, self.num_heads, self.head_dim)
        v = v.reshape(*lead, self.num_heads, self.head_dim)

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


class PhixtralMOE(nn.Module):
    """Mixture-of-Experts block with SwitchMLP for Phixtral.

    Uses SwitchMLP (fc1/fc2) with GELU activation instead of SwitchGLU.

    Attributes:
        num_experts: Total number of experts.
        num_experts_per_tok: Number of experts activated per token.
        gate: Router linear projection.
        switch_mlp: SwitchMLP expert bank.
    """

    def __init__(self, config: PhixtralConfig):
        """Initialize Phixtral MoE block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.switch_mlp = SwitchMLP(dims, mlp_dims, self.num_experts, bias=True)
        self.gate = nn.Linear(dims, self.num_experts, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Route tokens to top-k experts and aggregate results.

        Args:
            hidden_states: Input tensor of shape
                ``(batch, seq_len, model_dim)``.

        Returns:
            Output tensor of the same shape as input.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, orig_shape[-1])

        gate_logits = self.gate(hidden_states)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gate_logits, kth=k - 1, axis=-1)[:, :k])
        scores = mx.take_along_axis(gate_logits, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)

        y = self.switch_mlp(hidden_states, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(orig_shape)


class PhixtralParallelBlock(nn.Module):
    """Parallel attention+MoE block for Phixtral.

    Both attention and MoE receive the same normalized input and their
    outputs are summed with the residual.

    Attributes:
        mixer: RoPE attention module.
        moe: Mixture-of-experts module.
        ln: Layer normalization.
    """

    def __init__(self, config: PhixtralConfig):
        """Initialize Phixtral parallel block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        dims = config.model_dim
        self.mixer = PhixtralRoPEAttention(config)
        self.ln = nn.LayerNorm(dims, eps=config.layer_norm_eps)
        self.moe = PhixtralMOE(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run parallel attention + MoE with residual connection.

        Both branches receive the same LayerNorm-ed input, and their
        outputs are summed with the residual.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, model_dim)``.
            mask: Attention mask or None.
            cache_view: KV cache view for autoregressive decoding.
            cache_metadata: Paged-attention metadata.

        Returns:
            Output tensor of shape ``(batch, seq_len, model_dim)``.
        """
        h = self.ln(hidden_states)
        attn_h = self.mixer(
            h,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        ff_h = self.moe(h)
        return attn_h + ff_h + hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=PhixtralConfig, model_type="phixtral")
class PhixtralModel(EasyMLXBaseModule):
    """Base Phixtral transformer model.

    Attributes:
        config_class: The associated configuration class (``PhixtralConfig``).
        wte: Token embedding layer.
        h: List of parallel blocks.
    """

    config_class = PhixtralConfig

    def __init__(self, config: PhixtralConfig):
        """Initialize the base Phixtral model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.h = [PhixtralParallelBlock(config) for _ in range(config.num_layers)]

    @property
    def embed_tokens(self):
        """Compatibility alias for the token embedding layer."""
        return self.wte

    @property
    def layers(self):
        """Compatibility alias for the list of decoder layers."""
        return self.h

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through all parallel blocks.

        Note: Phixtral does not apply a final LayerNorm in the base
        model; the LM head wrapper applies its own norm.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, model_dim)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the
                number of layers.
        """
        if cache_views is not None and len(cache_views) != len(self.h):
            raise ValueError("cache_views length must match number of layers.")

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.wte(input_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        for layer_idx, layer in enumerate(self.h):
            layer_cache = None if cache_views is None else cache_views[layer_idx]
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_view=layer_cache,
                cache_metadata=cache_metadata,
            )

        return hidden_states


@register_module(task_type=TaskType.CAUSAL_LM, config=PhixtralConfig, model_type="phixtral")
class PhixtralForCausalLM(BaseCausalLMModule[PhixtralModel, PhixtralConfig]):
    """Phixtral model with a causal language modeling head.

    The LM head includes a LayerNorm followed by a linear projection,
    matching the upstream architecture.

    Attributes:
        config_class: The associated configuration class (``PhixtralConfig``).
    """

    config_class = PhixtralConfig

    def __init__(self, config: PhixtralConfig):
        """Initialize the Phixtral causal LM.

        Overrides the default LM head with a LayerNorm + linear
        projection matching the upstream architecture.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=PhixtralModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )
        # Override lm_head with norm+linear as in upstream phixtral.
        self.lm_head_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        return_dict: bool = True,
    ) -> mx.array:
        """Run the full model and return logits.

        Applies LayerNorm before the LM head projection.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.
            return_dict: Unused, kept for API compatibility.

        Returns:
            Logits tensor of shape ``(batch, seq_len, num_vocab)``.
        """
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )
        return self.lm_head(self.lm_head_norm(hidden_states))

    def sanitize(self, weights: dict) -> dict:
        """Sanitize upstream weights by stacking per-expert parameters.

        Transforms individual expert weights from
        ``transformer.h.N.moe.mlp.E.{fc1,fc2}.weight``
        into stacked ``transformer.h.N.moe.switch_mlp.{fc1,fc2}.weight``.
        Also remaps ``transformer.`` prefix to ``model.`` prefix.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        # Remap upstream transformer prefix to model prefix if needed.
        remapped = {}
        for key, val in weights.items():
            new_key = key
            if key.startswith("transformer."):
                new_key = "model." + key[len("transformer.") :]
                # Remap embd.wte -> wte
                new_key = new_key.replace("model.embd.wte", "model.wte")
            if key.startswith("lm_head."):
                # lm_head.ln -> lm_head_norm, lm_head.linear -> lm_head
                if key.startswith("lm_head.ln."):
                    new_key = "lm_head_norm." + key[len("lm_head.ln.") :]
                elif key.startswith("lm_head.linear."):
                    new_key = "lm_head." + key[len("lm_head.linear.") :]
            remapped[new_key] = val
        weights = remapped

        if "model.h.0.moe.mlp.0.fc1.weight" not in weights:
            return weights

        for layer_idx in range(self.config.num_layers):
            prefix = f"model.h.{layer_idx}"
            for n in ["fc1", "fc2"]:
                for k in ["weight", "scales", "biases", "bias"]:
                    if f"{prefix}.moe.mlp.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.moe.mlp.{e}.{n}.{k}") for e in range(self.config.num_local_experts)
                        ]
                        weights[f"{prefix}.moe.switch_mlp.{n}.{k}"] = mx.stack(to_join)

        return weights


__all__ = ("PhixtralForCausalLM", "PhixtralModel")
