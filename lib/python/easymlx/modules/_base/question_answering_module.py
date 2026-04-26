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

"""Generic base class for extractive Question Answering tasks on MLX.

This module provides ``BaseQuestionAnsweringModule`` for models that predict
start and end positions of answer spans within a context passage.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import QuestionAnsweringOutput

ModelT = tp.TypeVar("ModelT", bound=nn.Module)
ConfigT = tp.TypeVar("ConfigT", bound=EasyMLXBaseConfig)


class BaseQuestionAnsweringModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for extractive question answering on MLX.

    Predicts start and end positions of answer spans within context.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        qa_head_bias: bool = True,
    ):
        """Initialize the question answering module.

        Args:
            config: Model configuration. Must have a ``hidden_size`` attribute.
            base_model: Pre-instantiated base model instance.
            base_model_class: Base model class to instantiate if
                ``base_model`` is not provided.
            base_model_name: Attribute name for the base model.
                Defaults to ``"model"``.
            qa_head_bias: Whether to include bias in the QA output head.
                Defaults to True.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
        """
        super().__init__(config=config)
        self._base_model_name = base_model_name

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        self.qa_outputs = nn.Linear(config.hidden_size, 2, bias=qa_head_bias)

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
    ) -> QuestionAnsweringOutput:
        """Forward pass for extractive question answering.

        Runs the base model and applies the QA head to produce start and end
        logits for each token position.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask of shape ``(batch_size, seq_len)``.
            position_ids: Position indices for positional embeddings.
            cache: Cached key-value states for incremental decoding.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            ``QuestionAnsweringOutput`` containing ``start_logits`` and
            ``end_logits``, each of shape ``(batch_size, seq_len)``.
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

        logits = self.qa_outputs(hidden_states)
        start_logits = logits[..., 0]
        end_logits = logits[..., 1]

        return QuestionAnsweringOutput(start_logits=start_logits, end_logits=end_logits)

    def get_task_head(self) -> nn.Module:
        """Return the QA output head.

        Returns:
            The ``nn.Linear`` module that produces start/end logits.
        """
        return self.qa_outputs

    def get_lm_head(self) -> None:
        """Return the LM head. Not applicable for QA models.

        Raises:
            NotImplementedError: Always, as QA models do not have a
                language modeling head.
        """
        raise NotImplementedError("Question answering models don't have an lm_head.")
