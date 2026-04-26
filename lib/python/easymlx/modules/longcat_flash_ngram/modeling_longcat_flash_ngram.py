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

"""LongcatFlashNgram MLX model implementation for serving and inference.

Extends the LongcatFlash architecture with ngram-based token embeddings.
Instead of a simple embedding lookup, input tokens are combined with
hash-based ngram features from neighboring tokens, providing enhanced
local context modeling.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.modules._base import BaseCausalLMModule

from ..longcat_flash.modeling_longcat_flash import LongcatFlashDecoderLayer
from .longcat_flash_ngram_configuration import LongcatFlashNgramConfig

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like value to an int32 mx.array.

    Args:
        values: Input values to convert, or ``None``.

    Returns:
        An ``mx.array`` with dtype ``int32``, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


class NgramEmbedding(nn.Module):
    """Ngram embedding combining word embeddings with hash-based ngram features.

    Augments standard token embeddings with multiple ngram-level embeddings
    computed via polynomial hash functions over neighboring tokens. Each
    ngram level and split produces a separate embedding that is projected
    to the full hidden dimension and summed.

    Attributes:
        vocab_size: Token vocabulary size.
        hidden_size: Model hidden dimension.
        m: Base hash modulus (``ngram_vocab_size_ratio * vocab_size``).
        k: Number of embedding splits per ngram level.
        n: Number of neighboring tokens to consider.
        word_embeddings: Standard token embedding table.
        embedders: List of per-split ngram embedding tables.
        post_projs: List of per-split projection layers.

    Example:
        >>> config = LongcatFlashNgramConfig(
        ...     vocab_size=1000, hidden_size=64,
        ...     ngram_vocab_size_ratio=10, emb_neighbor_num=3, emb_split_num=2,
        ... )
        >>> emb = NgramEmbedding(config)
    """

    def __init__(self, config: LongcatFlashNgramConfig):
        """Initialize ngram embedding.

        Args:
            config: Model configuration with ngram parameters.
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = config.hidden_size // num_embedders

        self.embedders = []
        self.post_projs = []
        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
            self.post_projs.append(nn.Linear(emb_dim, config.hidden_size, bias=False))
        self._compute_vocab_mods()

    def _compute_vocab_mods(self):
        """Precompute polynomial hash modular powers for each ngram level."""
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right(self, x: mx.array, n: int) -> mx.array:
        """Shift token IDs right by n positions, zero-filling the left.

        Args:
            x: Token ID tensor of shape ``(batch, seq_len)``.
            n: Number of positions to shift.

        Returns:
            Shifted tensor of the same shape.
        """
        if n <= 0:
            return x
        batch_size, seq_len = x.shape
        if seq_len <= n:
            return mx.zeros_like(x)
        return mx.concatenate([mx.zeros((batch_size, n), dtype=x.dtype), x[..., :-n]], axis=-1)

    def _get_ngram_ids(self, input_ids, shifted_ids, vocab_mods, ngram):
        """Compute ngram hash IDs from input and shifted token IDs.

        Args:
            input_ids: Original token IDs.
            shifted_ids: Dict mapping shift amount to shifted token IDs.
            vocab_mods: Precomputed modular powers for the current split.
            ngram: Ngram order (e.g., 2 for bigram).

        Returns:
            Hash-based ngram IDs.
        """
        ngram_ids = input_ids
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(self, input_ids: mx.array, *, ngram_context: mx.array | None = None) -> mx.array:
        """Compute ngram-augmented embeddings.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.
            ngram_context: Optional context tokens to prepend for ngram
                computation (e.g., from a previous segment).

        Returns:
            Embedding tensor of shape ``(batch, seq_len, hidden_size)``,
            averaged over all ngram contributions.
        """
        seq_len = input_ids.shape[-1]
        input_ids = input_ids.astype(mx.int64)

        if ngram_context is not None:
            context = mx.concatenate([ngram_context, input_ids], axis=-1)
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right(context, i - 1)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self._get_ngram_ids(context, shifted_ids, self._vocab_mods[(i, j)], ngram=i)
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

        return x / (1 + self.k * (self.n - 1))


@register_module(task_type=TaskType.BASE_MODULE, config=LongcatFlashNgramConfig, model_type="longcat_flash_ngram")
class LongcatFlashNgramModel(EasyMLXBaseModule):
    """Base LongcatFlashNgram transformer model with ngram embeddings.

    Extends LongcatFlash by replacing the standard embedding layer with
    ``NgramEmbedding`` for hash-based ngram feature augmentation.

    Attributes:
        config_class: Associated configuration class.
        ngram_embeddings: Ngram-augmented embedding module.
        layers: List of ``LongcatFlashDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = LongcatFlashNgramConfig(
        ...     vocab_size=1000, hidden_size=64, num_layers=2,
        ... )
        >>> model = LongcatFlashNgramModel(config)
    """

    config_class = LongcatFlashNgramConfig

    def __init__(self, config: LongcatFlashNgramConfig):
        """Initialize the base LongcatFlashNgram model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.ngram_embeddings = NgramEmbedding(config)
        self.layers = [LongcatFlashDecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def embed_tokens(self):
        """Compatibility accessor for the word embedding layer."""
        return self.ngram_embeddings.word_embeddings

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass with ngram-augmented embeddings.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Cache views (``2 * num_layers`` entries).
            cache_metadata: Paged attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length is not ``2 * num_layers``.
        """
        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.ngram_embeddings(input_ids)

        mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = _as_int_array(attention_mask)
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)

        if cache_views is not None:
            if len(cache_views) != len(self.layers) * 2:
                raise ValueError(f"cache_views length must be {len(self.layers) * 2}, got {len(cache_views)}.")
            layer_caches = [[cache_views[i * 2], cache_views[i * 2 + 1]] for i in range(len(self.layers))]
        else:
            layer_caches = [None] * len(self.layers)

        for layer, lc in zip(self.layers, layer_caches, strict=False):
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache_views=lc,
                cache_metadata=cache_metadata,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize upstream weights for LongcatFlashNgram.

        Remaps ``embed_tokens`` to ``ngram_embeddings.word_embeddings``,
        stacks per-expert weights, and removes rotary embedding and MTP keys.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        if "model.embed_tokens.weight" in weights:
            weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop("model.embed_tokens.weight")

        for layer_idx in range(self.config.num_layers):
            prefix = f"model.layers.{layer_idx}"
            for _n, _m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{_m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{_m}.{k}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{_m}.{k}"] = mx.stack(to_join)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key and not key.startswith("model.mtp")
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=LongcatFlashNgramConfig, model_type="longcat_flash_ngram")
class LongcatFlashNgramForCausalLM(BaseCausalLMModule[LongcatFlashNgramModel, LongcatFlashNgramConfig]):
    """LongcatFlashNgram causal language model with an LM head.

    Wraps ``LongcatFlashNgramModel`` with a language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = LongcatFlashNgramConfig(vocab_size=1000, hidden_size=64, num_layers=2)
        >>> model = LongcatFlashNgramForCausalLM(config)
    """

    config_class = LongcatFlashNgramConfig

    def __init__(self, config: LongcatFlashNgramConfig):
        """Initialize the LongcatFlashNgram causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=LongcatFlashNgramModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("LongcatFlashNgramForCausalLM", "LongcatFlashNgramModel")
