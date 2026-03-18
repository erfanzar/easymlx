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

"""Generic base class for Causal Language Modeling tasks on MLX.

This module provides ``BaseCausalLMModule``, the foundation for all autoregressive
language models. It wraps a base transformer model with a language modeling head
that projects hidden states to vocabulary logits, supporting weight tying, logit
capping, and both standard and paged KV cache strategies.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCache,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import CausalLMOutput

CacheView = TransformerCacheView | PageCache


class BaseCausalLMModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for causal language models on MLX.

    Wraps a base transformer model with a language modeling head that projects
    hidden states to vocabulary logits. Supports weight tying, logit capping,
    paged cache initialization, and standard cache initialization.

    Subclasses only need to provide ``__init__`` — ``__call__``, ``init_paged_cache``,
    ``init_cache``, and ``sanitize`` are all inherited.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
    ):
        """Initialize the causal language model module.

        Args:
            config: Model configuration containing hyperparameters such as
                ``hidden_size``, ``vocab_size``, etc.
            base_model: Pre-instantiated base model instance. If provided,
                ``base_model_class`` is ignored.
            base_model_class: Base model class to instantiate. Required if
                ``base_model`` is not provided.
            base_model_name: Attribute name under which to store the base
                model. Defaults to ``"model"``.
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the LM head. Defaults to False.
            logit_cap: Maximum absolute value for output logits via tanh
                scaling. If None, no capping is applied. Defaults to None.
            lm_head_name: Attribute name for the LM head linear layer.
                Defaults to ``"lm_head"``.
            lm_head_bias: Whether to include bias in the LM head.
                Defaults to False.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
        """
        super().__init__(config=config)
        self._base_model_name = base_model_name
        self._lm_head_name = lm_head_name
        self._tie_word_embeddings = tie_word_embeddings
        self._logit_cap = logit_cap

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        if not tie_word_embeddings:
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=lm_head_bias)
            setattr(self, lm_head_name, lm_head)

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            The underlying transformer base model instance.
        """
        return getattr(self, self._base_model_name)

    def get_embedding(self) -> nn.Embedding:
        """Return the input embedding layer from the base model.

        Searches for the embedding layer using common attribute names
        and method conventions across different model architectures.

        Returns:
            The input embedding module.

        Raises:
            AttributeError: If no embedding layer can be found on the
                base model.
        """
        model = self.base_model
        for attr in ("get_embedding", "get_input_embeddings"):
            fn = getattr(model, attr, None)
            if callable(fn):
                return fn()
        for attr in ("embed_tokens", "wte", "word_embeddings", "embeddings"):
            if hasattr(model, attr):
                return getattr(model, attr)
        raise AttributeError("Cannot find embedding layer on base model.")

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits.

        Uses the LM head linear layer, or the embedding layer's ``as_linear``
        method if weight tying is enabled. Applies logit capping if configured.

        Args:
            hidden_states: Hidden state tensor of shape
                ``(batch_size, seq_len, hidden_size)`` or
                ``(num_tokens, hidden_size)``.

        Returns:
            Logits tensor of shape ``(..., vocab_size)``.
        """
        if self._tie_word_embeddings:
            embedding = self.get_embedding()
            logits = embedding.as_linear(hidden_states)
        else:
            lm_head = getattr(self, self._lm_head_name)
            logits = lm_head(hidden_states)
        return self.apply_logit_cap(logits)

    def apply_logit_cap(self, logits: mx.array) -> mx.array:
        """Apply logit capping via tanh scaling if configured.

        When a logit cap value is set, applies ``cap * tanh(logits / cap)``
        to smoothly constrain logits to ``(-cap, cap)``.

        Args:
            logits: Input logits tensor of any shape.

        Returns:
            Capped logits if logit capping is configured, otherwise
            the input unchanged.
        """
        if self._logit_cap is not None:
            cap = self._logit_cap
            logits = cap * mx.tanh(logits / cap)
        return logits

    def _extract_last_tokens(
        self,
        hidden_states: mx.array,
        cache_metadata: PageMetadata | None,
    ) -> mx.array:
        """Extract the last token hidden state per sequence for paged serving.

        When using paged attention with flattened (2D) hidden states, this
        method uses ``cache_metadata.query_start_loc`` to identify the last
        token position for each sequence in the batch.

        Args:
            hidden_states: Hidden states tensor. If 2D and cache_metadata is
                provided, assumed to be flattened across sequences.
            cache_metadata: Paged cache metadata containing query start
                locations. If None, hidden_states are returned as-is.

        Returns:
            Hidden states with only the last token per sequence, shaped
            ``(batch_size, hidden_size)``.
        """
        if hidden_states.ndim == 2 and cache_metadata is not None:
            qsl = cache_metadata.query_start_loc
            if not isinstance(qsl, mx.array):
                qsl = mx.array(list(qsl), dtype=mx.int32)
            last_indices = (qsl[1:].astype(mx.int32) - 1).astype(mx.int32)
            hidden_states = mx.take(hidden_states, last_indices, axis=0)
        return hidden_states

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
        """Forward pass through the causal language model.

        Runs the base model, extracts the last tokens (for paged serving),
        and projects to vocabulary logits.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``
                or ``(num_tokens,)`` for paged serving.
            attention_mask: Optional attention mask of shape
                ``(batch_size, seq_len)``.
            input_embeddings: Optional pre-computed input embeddings.
                Overrides ``input_ids`` if provided.
            cache_views: Per-layer KV cache views for incremental decoding.
            cache_metadata: Paged cache metadata for paged attention serving.
            return_dict: If True, return a ``CausalLMOutput`` dataclass.
                If False, return raw logits tensor. Defaults to True.

        Returns:
            A ``CausalLMOutput`` containing logits if ``return_dict`` is True,
            otherwise a raw logits ``mx.array``.
        """
        hidden_states = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

        hidden_states = self._extract_last_tokens(hidden_states, cache_metadata)
        logits = self.compute_lm_logits(hidden_states)

        if return_dict:
            return CausalLMOutput(logits=logits)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Filter out rotary embedding frequencies and tied LM head weights.

        Removes weight keys that should not be loaded, such as
        ``rotary_emb.inv_freq`` (computed at init) and the LM head weight
        when weight tying is enabled.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weights dictionary with irrelevant keys removed.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self._tie_word_embeddings:
            weights.pop(f"{self._lm_head_name}.weight", None)
        return weights

    def get_task_head(self) -> nn.Module | None:
        """Return the LM head module, or None if weight tying is enabled.

        Returns:
            The LM head ``nn.Linear`` module, or None when weights are tied
            with the input embedding layer.
        """
        if self._tie_word_embeddings:
            return None
        return getattr(self, self._lm_head_name, None)

    def get_lm_head(self) -> nn.Module | None:
        """Return the LM head module. Alias for ``get_task_head``.

        Returns:
            The LM head module, or None if weight tying is enabled.
        """
        return self.get_task_head()
