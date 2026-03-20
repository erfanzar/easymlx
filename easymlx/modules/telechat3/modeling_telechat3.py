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

"""Telechat3 MLX model implementation for serving and inference."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import CausalLMOutput, EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import build_attention_mask
from easymlx.modules.llama.modeling_llama import LlamaDecoderLayer

from .telechat3_configuration import Telechat3Config

CacheView = TransformerCacheView | PageCacheView


def _as_int_array(values: mx.ArrayLike | None) -> mx.array | None:
    """Convert an array-like to an ``mx.array`` of dtype ``int32``.

    Args:
        values: Input values. Accepts ``mx.array``, sequences, or ``None``.

    Returns:
        An ``mx.array`` with ``int32`` dtype, or ``None``.
    """
    if values is None:
        return None
    if isinstance(values, mx.array):
        return values.astype(mx.int32)
    return mx.array(list(values), dtype=mx.int32)


@register_module(task_type=TaskType.BASE_MODULE, config=Telechat3Config, model_type="telechat3")
class Telechat3Model(EasyMLXBaseModule):
    """Base Telechat3 transformer model.

    A standard dense decoder-only transformer that reuses
    ``LlamaDecoderLayer`` for its layer stack. Supports both batched
    3-D input and flat 2-D paged-attention input.

    Attributes:
        config_class: Associated configuration class (``Telechat3Config``).
        embed_tokens: Token embedding layer.
        layers: List of ``LlamaDecoderLayer`` modules.
        norm: Final RMSNorm.

    Example:
        >>> config = Telechat3Config(hidden_size=4096, num_hidden_layers=4)
        >>> model = Telechat3Model(config)
        >>> hidden = model(mx.array([[1, 2, 3]]))
    """

    config_class = Telechat3Config

    def __init__(self, config: Telechat3Config):
        """Initialize the base Telechat3 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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
        """Run the forward pass through all decoder layers.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``cache_views`` length does not match the number
                of layers.
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
        """Remove non-persistent rotary embedding buffers from checkpoint weights.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=Telechat3Config, model_type="telechat3")
class Telechat3ForCausalLM(EasyMLXBaseModule):
    """Telechat3 causal language model with an LM head.

    Unlike most other models in EasyMLX, Telechat3ForCausalLM directly
    extends ``EasyMLXBaseModule`` (not ``BaseCausalLMModule``) to provide
    custom paged-cache output extraction and explicit embedding-tying
    logic.

    Attributes:
        config_class: Associated configuration class (``Telechat3Config``).
        model: The underlying ``Telechat3Model``.

    Example:
        >>> config = Telechat3Config(hidden_size=4096, num_hidden_layers=4)
        >>> model = Telechat3ForCausalLM(config)
    """

    config_class = Telechat3Config

    def __init__(self, config: Telechat3Config):
        """Initialize the Telechat3 causal LM.

        Args:
            config: Model configuration. ``tie_word_embeddings`` controls
                whether the LM head reuses the embedding weights.
        """
        super().__init__(config)
        self.model = Telechat3Model(config)
        self._tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        if not self._tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        return_dict: bool = True,
    ) -> mx.array | CausalLMOutput:
        """Run the forward pass and compute logits.

        For paged-attention mode (``cache_metadata`` provided), extracts
        the last token hidden state per sequence using
        ``cache_metadata.query_start_loc`` before projecting to logits.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.
            return_dict: If ``True``, returns ``CausalLMOutput``; otherwise
                returns raw logits tensor.

        Returns:
            ``CausalLMOutput`` with logits or raw logits tensor depending
            on ``return_dict``.
        """
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

        if hidden_states.ndim == 2 and cache_metadata is not None:
            qsl = cache_metadata.query_start_loc
            if not isinstance(qsl, mx.array):
                qsl = mx.array(list(qsl), dtype=mx.int32)
            last_indices = (qsl[1:].astype(mx.int32) - 1).astype(mx.int32)
            hidden_states = mx.take(hidden_states, last_indices, axis=0)

        if self._tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if return_dict:
            return CausalLMOutput(logits=logits)
        return logits

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            The ``nn.Embedding`` module from the underlying model.
        """
        return self.model.embed_tokens

    def get_task_head(self):
        """Return the LM head projection, or ``None`` if embeddings are tied.

        Returns:
            The ``nn.Linear`` LM head or ``None``.
        """
        if self._tie_word_embeddings:
            return None
        return self.lm_head

    def get_lm_head(self):
        """Return the LM head projection (alias for ``get_task_head``).

        Returns:
            The ``nn.Linear`` LM head or ``None``.
        """
        return self.get_task_head()

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights: remove rotary buffers and LM head if tied.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict.
        """
        weights = self.model.sanitize(weights)
        if self._tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights


__all__ = ("Telechat3ForCausalLM", "Telechat3Model")
