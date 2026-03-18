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

"""Generic base class for Embedding / Retrieval tasks on MLX.

This module provides ``BaseEmbeddingModule`` for producing dense vector
representations from input sequences. It supports multiple pooling strategies,
optional L2 normalization, and Matryoshka truncation for flexible embedding
dimensions.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import EmbeddingOutput

ModelT = tp.TypeVar("ModelT", bound=nn.Module)
ConfigT = tp.TypeVar("ConfigT", bound=EasyMLXBaseConfig)


def _pool_sequence(
    hidden_states: mx.array,
    strategy: str,
    input_ids: mx.array | None = None,
    attention_mask: mx.array | None = None,
    pad_token_id: int | None = None,
) -> mx.array:
    """Pool a sequence of hidden states to a single vector per batch element.

    Args:
        hidden_states: Tensor of shape ``(batch_size, seq_len, hidden_dim)``.
        strategy: Pooling strategy. One of ``"first"``, ``"mean"``, ``"max"``,
            ``"last"``, or ``"weighted_mean"``.
        input_ids: Optional input token IDs for determining sequence lengths
            in the ``"last"`` strategy.
        attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len)`` where 1 indicates valid tokens.
        pad_token_id: Padding token ID, used with ``"last"`` strategy when
            ``attention_mask`` is not provided.

    Returns:
        Pooled tensor of shape ``(batch_size, hidden_dim)``.

    Raises:
        ValueError: If ``strategy`` is not a recognized pooling strategy.
    """
    if strategy == "first":
        return hidden_states[:, 0]
    elif strategy == "mean":
        if attention_mask is not None:
            mask = attention_mask[:, :, None].astype(hidden_states.dtype)
            return (hidden_states * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), mx.array(1e-9))
        return hidden_states.mean(axis=1)
    elif strategy == "max":
        if attention_mask is not None:
            mask = attention_mask[:, :, None].astype(hidden_states.dtype)
            hidden_states = hidden_states * mask + (1 - mask) * (-1e9)
        return hidden_states.max(axis=1)
    elif strategy == "last":
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(axis=-1).astype(mx.int32) - 1
        elif input_ids is not None and pad_token_id is not None:
            seq_lengths = (input_ids != pad_token_id).sum(axis=-1).astype(mx.int32) - 1
        else:
            seq_lengths = mx.array([hidden_states.shape[1] - 1] * hidden_states.shape[0])
        batch_idx = mx.arange(hidden_states.shape[0])
        return hidden_states[batch_idx, seq_lengths]
    elif strategy == "weighted_mean":
        if attention_mask is not None:
            mask = attention_mask[:, :, None].astype(hidden_states.dtype)
            seq_len = hidden_states.shape[1]
            weights = mx.arange(1, seq_len + 1, dtype=hidden_states.dtype)[None, :, None]
            weights = weights * mask
            return (hidden_states * weights).sum(axis=1) / mx.maximum(weights.sum(axis=1), mx.array(1e-9))
        seq_len = hidden_states.shape[1]
        weights = mx.arange(1, seq_len + 1, dtype=hidden_states.dtype)[None, :, None]
        return (hidden_states * weights).sum(axis=1) / weights.sum(axis=1)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


class BaseEmbeddingModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for embedding models on MLX.

    Pools hidden states from a base transformer and optionally L2-normalizes
    the result. No task-specific head is required.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        pooling_strategy: str = "last",
        normalize_embeddings: bool = True,
        embedding_dim: int | None = None,
    ):
        """Initialize the embedding module.

        Args:
            config: Model configuration containing hyperparameters.
            base_model: Pre-instantiated base model instance.
            base_model_class: Base model class to instantiate if
                ``base_model`` is not provided.
            base_model_name: Attribute name for the base model.
                Defaults to ``"model"``.
            pooling_strategy: Strategy for reducing sequence to a single
                vector. One of ``"last"``, ``"first"``, ``"mean"``,
                ``"max"``, ``"weighted_mean"``. Defaults to ``"last"``.
            normalize_embeddings: Whether to L2-normalize the output
                embeddings. Defaults to True.
            embedding_dim: Optional Matryoshka truncation dimension. If set,
                embeddings are truncated to this size. Defaults to None.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
        """
        super().__init__(config=config)
        self._base_model_name = base_model_name
        self._pooling_strategy = pooling_strategy
        self._normalize_embeddings = normalize_embeddings
        self._embedding_dim = embedding_dim

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            The underlying transformer base model instance.
        """
        return getattr(self, self._base_model_name)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: tp.Any | None = None,
        inputs_embeds: mx.array | None = None,
        **kwargs,
    ) -> EmbeddingOutput:
        """Forward pass to produce embeddings.

        Runs the base model, pools hidden states, optionally truncates
        and L2-normalizes the result.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
            position_ids: Position indices for positional embeddings.
            cache: Cached key-value states (typically unused for embeddings).
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            ``EmbeddingOutput`` containing the embedding tensor of shape
            ``(batch_size, embedding_dim)``.
        """
        base_kwargs: dict[str, tp.Any] = {}
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds
        else:
            base_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            base_kwargs["position_ids"] = position_ids
        if cache is not None:
            base_kwargs["cache"] = cache
        base_kwargs.update(kwargs)

        outputs = self.base_model(**base_kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = getattr(outputs, "last_hidden_state", outputs)

        embeddings = _pool_sequence(
            hidden_states,
            self._pooling_strategy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=getattr(self.config, "pad_token_id", None),
        )

        # Optional Matryoshka truncation
        if self._embedding_dim is not None:
            embeddings = embeddings[:, : self._embedding_dim]

        # Optional L2 normalization
        if self._normalize_embeddings:
            norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / mx.maximum(norms, mx.array(1e-12))

        return EmbeddingOutput(embeddings=embeddings)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        *,
        normalize: bool | None = None,
        truncate_dim: int | None = None,
    ) -> mx.array:
        """Convenience method that returns raw embedding array.

        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask for padding.
            normalize: Override L2 normalization setting.
            truncate_dim: Override Matryoshka truncation dimension.

        Returns:
            Embedding array of shape (batch, dim).
        """
        orig_norm = self._normalize_embeddings
        orig_dim = self._embedding_dim
        try:
            if normalize is not None:
                self._normalize_embeddings = normalize
            if truncate_dim is not None:
                self._embedding_dim = truncate_dim
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.embeddings
        finally:
            self._normalize_embeddings = orig_norm
            self._embedding_dim = orig_dim

    @staticmethod
    def cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
        """Compute pairwise cosine similarity between two embedding matrices.

        Args:
            a: First embedding matrix of shape ``(n, dim)``.
            b: Second embedding matrix of shape ``(m, dim)``.

        Returns:
            Similarity matrix of shape ``(n, m)`` with cosine similarity
            scores in the range ``[-1, 1]``.
        """
        a_norm = a / mx.maximum(mx.linalg.norm(a, axis=-1, keepdims=True), mx.array(1e-12))
        b_norm = b / mx.maximum(mx.linalg.norm(b, axis=-1, keepdims=True), mx.array(1e-12))
        return a_norm @ b_norm.T

    @staticmethod
    def dot_product_similarity(a: mx.array, b: mx.array) -> mx.array:
        """Compute pairwise dot-product similarity between two embedding matrices.

        Args:
            a: First embedding matrix of shape ``(n, dim)``.
            b: Second embedding matrix of shape ``(m, dim)``.

        Returns:
            Similarity matrix of shape ``(n, m)`` with dot-product scores.
        """
        return a @ b.T

    def get_task_head(self) -> None:
        """Return the task-specific head. Embedding models have none.

        Returns:
            Always returns None since embedding models produce dense vectors
            directly from pooled hidden states.
        """
        return None

    def get_lm_head(self) -> None:
        """Return the LM head. Not applicable for embedding models.

        Raises:
            NotImplementedError: Always, as embedding models do not have
                a language modeling head.
        """
        raise NotImplementedError("Embedding models do not have a language modeling head.")
