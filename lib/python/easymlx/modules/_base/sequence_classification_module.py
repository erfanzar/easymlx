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

"""Generic base class for Sequence Classification tasks on MLX.

This module provides ``BaseSequenceClassificationModule`` for models that
classify entire input sequences (e.g., sentiment analysis, natural language
inference). It wraps a base transformer with a classification head that pools
the sequence and projects to ``num_labels`` logits.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import SequenceClassifierOutput

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
            or ``"last"``.
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
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


class BaseSequenceClassificationModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for sequence classification models on MLX.

    Wraps a base transformer and adds a classification head that pools
    the sequence and projects to num_labels logits.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        pooling_strategy: str = "last",
        score_head_name: str = "score",
        score_head_bias: bool = False,
    ):
        """Initialize the sequence classification module.

        Args:
            config: Model configuration. Must have ``num_labels`` and
                ``hidden_size`` attributes.
            base_model: Pre-instantiated base model instance.
            base_model_class: Base model class to instantiate if
                ``base_model`` is not provided.
            base_model_name: Attribute name for the base model.
                Defaults to ``"model"``.
            pooling_strategy: Strategy for pooling the sequence. One of
                ``"last"``, ``"first"``, ``"mean"``, or ``"max"``.
                Defaults to ``"last"``.
            score_head_name: Attribute name for the score head linear layer.
                Defaults to ``"score"``.
            score_head_bias: Whether to include bias in the score head.
                Defaults to False.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
            AssertionError: If ``config`` does not have ``num_labels``.
        """
        super().__init__(config=config)
        assert hasattr(config, "num_labels"), "config must have num_labels attribute"

        self._base_model_name = base_model_name
        self._score_head_name = score_head_name
        self._pooling_strategy = pooling_strategy

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        score_head = nn.Linear(config.hidden_size, config.num_labels, bias=score_head_bias)
        setattr(self, score_head_name, score_head)

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
    ) -> SequenceClassifierOutput:
        """Forward pass for sequence classification.

        Runs the base model, applies the score head to all tokens, then
        pools to a single logit vector per sequence.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
            position_ids: Position indices for positional embeddings.
            cache: Cached key-value states for incremental decoding.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            ``SequenceClassifierOutput`` containing pooled classification
            logits of shape ``(batch_size, num_labels)``.
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

        score_head = getattr(self, self._score_head_name)
        logits = score_head(hidden_states)

        pooled_logits = _pool_sequence(
            logits,
            self._pooling_strategy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=getattr(self.config, "pad_token_id", None),
        )

        return SequenceClassifierOutput(logits=pooled_logits)

    def get_task_head(self) -> nn.Module:
        """Return the classification score head.

        Returns:
            The ``nn.Linear`` module that produces classification logits.
        """
        return getattr(self, self._score_head_name)

    def get_lm_head(self) -> None:
        """Return the LM head. Not applicable for classification models.

        Raises:
            NotImplementedError: Always, as sequence classification models
                use a classification head, not a language modeling head.
        """
        raise NotImplementedError("SequenceClassification models use a classification head, not an lm_head.")
