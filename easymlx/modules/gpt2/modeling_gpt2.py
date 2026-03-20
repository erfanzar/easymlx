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

"""GPT-2 MLX model implementation for serving and inference.

This module provides the GPT-2 architecture on MLX, featuring absolute
position embeddings, LayerNorm, GELU approximate activation, bias in
attention and MLP, and Conv1D weight sanitization for HuggingFace
checkpoint compatibility.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from .gpt2_configuration import GPT2Config

CacheView = TransformerCacheView | PageCacheView


class GPT2Attention(nn.Module):
    """GPT-2 multi-head attention with absolute position embeddings.

    Uses a fused QKV projection (``c_attn``) and does not apply
    rotary embeddings -- position information comes from absolute
    position embeddings added at the model level.

    Attributes:
        n_head: Number of attention heads.
        n_embd: Hidden dimensionality.
        head_dim: Per-head dimensionality (``n_embd // n_head``).
        scale: Attention scaling factor (``head_dim ** -0.5``).
        c_attn: Fused Q/K/V linear projection with bias.
        c_proj: Output projection with bias.
        attention_performer: Attention computation backend.

    Example::

        >>> attn = GPT2Attention(GPT2Config(n_embd=64, n_head=4))
        >>> out = attn(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 attention.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

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
        """Compute multi-head attention (no RoPE).

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.
            mask: Attention mask.
            cache_view: Optional KV cache for decoding.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Output of shape ``[batch, seq_len, n_embd]``.
        """
        lead = hidden_states.shape[:-1]

        qkv = self.c_attn(hidden_states)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.reshape(*lead, self.n_head, self.head_dim)
        keys = keys.reshape(*lead, self.n_head, self.head_dim)
        values = values.reshape(*lead, self.n_head, self.head_dim)

        attn = self.attention_performer(
            queries,
            keys,
            values,
            rope=None,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        return self.c_proj(attn.reshape(*lead, -1))


class GPT2MLP(nn.Module):
    """GPT-2 feed-forward network with GELU approximate activation.

    A simple two-layer MLP with 4x expansion and bias:
    ``c_proj(gelu_approx(c_fc(x)))``.

    Attributes:
        c_fc: Up-projection to ``4 * n_embd``.
        c_proj: Down-projection back to ``n_embd``.

    Example::

        >>> mlp = GPT2MLP(GPT2Config(n_embd=64))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 MLP.

        Args:
            config: Model configuration with ``n_embd``.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Apply MLP with GELU approximate activation.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.

        Returns:
            Output of the same shape.
        """
        return self.c_proj(nn.gelu_approx(self.c_fc(hidden_states)))


class GPT2DecoderLayer(nn.Module):
    """Single GPT-2 decoder layer with pre-norm and residuals.

    Uses LayerNorm (not RMSNorm) before attention and MLP sub-layers
    with additive residual connections.

    Attributes:
        attn: GPT-2 multi-head attention.
        mlp: GPT-2 GELU MLP.
        ln_1: LayerNorm before attention.
        ln_2: LayerNorm before MLP.

    Example::

        >>> layer = GPT2DecoderLayer(GPT2Config(n_embd=64))
        >>> out = layer(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 decoder layer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attn = GPT2Attention(config)
        self.mlp = GPT2MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass through one GPT-2 decoder layer.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, n_embd]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.

        Returns:
            Hidden states after attention and MLP.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@register_module(task_type=TaskType.BASE_MODULE, config=GPT2Config, model_type="gpt2")
class GPT2Model(EasyMLXBaseModule):
    """Base GPT-2 transformer with learned absolute position embeddings.

    Unlike RoPE-based models, GPT-2 adds learned position embeddings
    (``wpe``) to the token embeddings before passing through the
    transformer layers. The position offset is derived from the KV
    cache for autoregressive decoding.

    Attributes:
        wte: Token embedding layer.
        wpe: Absolute position embedding layer.
        h: Stack of ``GPT2DecoderLayer`` instances.
        ln_f: Final LayerNorm.

    Example::

        >>> model = GPT2Model(GPT2Config(vocab_size=256, n_embd=64, n_layer=2))
        >>> h = model(mx.array([[1, 2, 3]]))
        >>> h.shape
        [1, 3, 64]
    """

    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        """Initialize GPT-2 base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = [GPT2DecoderLayer(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    @property
    def layers(self):
        """Return the decoder layer stack (alias ``h``)."""
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
        """Forward pass with absolute position embeddings.

        Adds learned position embeddings to token embeddings, then
        passes through the decoder layers and final LayerNorm.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Normalized hidden states of shape
            ``[batch, seq_len, n_embd]``.

        Raises:
            ValueError: If ``cache_views`` length does not match layers.
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

        # Compute position embeddings.
        if hidden_states.ndim == 3:
            seq_len = hidden_states.shape[1]
        else:
            seq_len = hidden_states.shape[0]

        offset = 0
        if cache_views is not None and cache_views[0] is not None:
            offset = cache_views[0].offset
        offset = mx.array(offset)
        position_ids = mx.arange(seq_len) + offset[..., None]
        hidden_states = hidden_states + self.wpe(position_ids)

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

        return self.ln_f(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize HuggingFace GPT-2 weights for MLX.

        Performs three transformations:

        1. Removes ``attn.bias`` and ``attn.masked_bias`` buffer keys
           (causal mask buffers, not real learned biases).
        2. Transposes Conv1D weights from HuggingFace's ``[out, in]``
           layout to MLX's ``[in, out]`` layout. HF GPT-2 uses Conv1D
           layers which store weights transposed relative to nn.Linear.
        3. Adds ``model.`` prefix to keys that lack it.

        Args:
            weights: Raw HuggingFace checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary compatible with this module.
        """
        new_weights: dict[str, mx.array] = {}
        for key, value in weights.items():
            # Skip attention bias buffers (but not real parameter biases like c_attn.bias).
            if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
                continue

            # Transpose Conv1D weights from HF (transposed) to MLX layout.
            if key.endswith((".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")):
                value = value.transpose(1, 0)

            # Ensure model. prefix.
            if not key.startswith("model."):
                new_weights[f"model.{key}"] = value
            else:
                new_weights[key] = value

        return new_weights


@register_module(task_type=TaskType.CAUSAL_LM, config=GPT2Config, model_type="gpt2")
class GPT2ForCausalLM(BaseCausalLMModule[GPT2Model, GPT2Config]):
    """GPT-2 causal language model with tied embeddings.

    Wraps ``GPT2Model`` and adds a linear LM head. Supports tied
    word embeddings by default. The ``sanitize`` method handles
    Conv1D weight transposition and tied weight removal.

    Attributes:
        config_class: ``GPT2Config``.

    Example::

        >>> model = GPT2ForCausalLM(GPT2Config(vocab_size=256, n_embd=64))
        >>> logits = model(mx.array([[1, 2, 3]]))
        >>> logits.shape
        [1, 3, 256]
    """

    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(
            config=config,
            base_model_class=GPT2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize HuggingFace GPT-2 weights for the causal LM.

        Performs the same Conv1D transposition and prefix normalization
        as ``GPT2Model.sanitize``, and additionally removes the
        ``lm_head.weight`` key when word embeddings are tied.

        Args:
            weights: Raw HuggingFace checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary compatible with this module.
        """
        new_weights: dict[str, mx.array] = {}
        for key, value in weights.items():
            # Skip attention bias buffers (but not real parameter biases like c_attn.bias).
            if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
                continue

            # Transpose Conv1D weights from HF (transposed) to MLX layout.
            if key.endswith((".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")):
                value = value.transpose(1, 0)

            # Ensure model. prefix.
            if not key.startswith("model."):
                new_weights[f"model.{key}"] = value
            else:
                new_weights[key] = value

        # Remove tied lm_head if applicable.
        if self._tie_word_embeddings:
            new_weights.pop("lm_head.weight", None)
            new_weights.pop("model.lm_head.weight", None)

        return new_weights


__all__ = ("GPT2ForCausalLM", "GPT2Model")
