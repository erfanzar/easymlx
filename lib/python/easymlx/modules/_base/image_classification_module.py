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

"""Generic base class for Image Classification tasks on MLX.

This module provides ``BaseImageClassificationModule`` for models that classify
images into predefined categories. It processes pixel values through a vision
encoder, pools the output, and projects to class logits.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import ImageClassifierOutput

ModelT = tp.TypeVar("ModelT", bound=nn.Module)
ConfigT = tp.TypeVar("ConfigT", bound=EasyMLXBaseConfig)


class BaseImageClassificationModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for image classification models on MLX.

    Processes pixel values through a vision encoder, pools the output, and
    projects to class logits.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "vision_model",
        *,
        pooling_strategy: str = "first",
        classifier_bias: bool = True,
    ):
        """Initialize the image classification module.

        Args:
            config: Model configuration. Must have a ``num_labels`` attribute
                specifying the number of classification classes.
            base_model: Pre-instantiated vision model instance.
            base_model_class: Vision model class to instantiate if
                ``base_model`` is not provided.
            base_model_name: Attribute name for the vision model.
                Defaults to ``"vision_model"``.
            pooling_strategy: Strategy for pooling patch embeddings. One of
                ``"first"`` (CLS token), ``"mean"``, or ``"max"``.
                Defaults to ``"first"``.
            classifier_bias: Whether to include bias in the classification
                head. Defaults to True.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
            AssertionError: If ``config`` does not have ``num_labels``.
            AttributeError: If vision hidden size cannot be inferred from
                the config.
        """
        super().__init__(config=config)
        assert hasattr(config, "num_labels"), "config must have num_labels attribute"

        self._base_model_name = base_model_name
        self._pooling_strategy = pooling_strategy

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        vision_config = getattr(config, "vision_config", config)
        hidden_size = getattr(vision_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Cannot infer vision hidden_size from config.")

        self.classifier = None
        if config.num_labels > 0:
            self.classifier = nn.Linear(hidden_size, config.num_labels, bias=classifier_bias)

    @property
    def base_model(self) -> ModelT:
        """Access the vision model via the configured attribute name.

        Returns:
            The underlying vision model instance.
        """
        return getattr(self, self._base_model_name)

    def _pool(self, hidden_states: mx.array) -> mx.array:
        """Pool vision encoder hidden states to a single vector per image.

        Args:
            hidden_states: Tensor of shape
                ``(batch_size, num_patches, hidden_dim)``.

        Returns:
            Pooled tensor of shape ``(batch_size, hidden_dim)``.

        Raises:
            ValueError: If the pooling strategy is not recognized.
        """
        if self._pooling_strategy == "first":
            return hidden_states[:, 0]
        elif self._pooling_strategy == "mean":
            return hidden_states.mean(axis=1)
        elif self._pooling_strategy == "max":
            return hidden_states.max(axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self._pooling_strategy}")

    def __call__(
        self,
        pixel_values: mx.array,
        **kwargs,
    ) -> ImageClassifierOutput:
        """Forward pass for image classification.

        Args:
            pixel_values: Input images of shape
                ``(batch_size, channels, height, width)``.
            **kwargs: Additional keyword arguments forwarded to the vision model.

        Returns:
            ``ImageClassifierOutput`` containing classification logits of
            shape ``(batch_size, num_labels)``.
        """
        outputs = self.base_model(pixel_values=pixel_values, **kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = getattr(outputs, "last_hidden_state", outputs)

        pooled = self._pool(hidden_states)
        logits = pooled if self.classifier is None else self.classifier(pooled)

        return ImageClassifierOutput(logits=logits)

    def get_task_head(self) -> nn.Module | None:
        """Return the classification head.

        Returns:
            The classifier ``nn.Linear`` module, or None if ``num_labels``
            is 0.
        """
        return self.classifier

    def get_lm_head(self) -> None:
        """Return the LM head. Not applicable for image classification models.

        Raises:
            NotImplementedError: Always, as image classification models do
                not have a language modeling head.
        """
        raise NotImplementedError("Image classification models don't have an lm_head.")
