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

"""Abstract base class for all task-specific modules.

This module provides the foundation for all task-specific model wrappers,
including auto-registration, feature management, and common interfaces.

The BaseTaskModule class serves as the abstract base for creating specialized
model wrappers for different NLP and ML tasks such as:
    - Causal Language Modeling
    - Sequence Classification
    - Token Classification
    - Question Answering
    - Image Classification
    - Vision-Language Modeling

Key Features:
    - Generic typing support with ModelT and ConfigT type parameters
    - Automatic registration with the factory system
    - Modular feature system for logit capping, embedding tying, etc.
    - Consistent interface across all task-specific modules
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.factory import TaskType, register_module

from ._features import (
    GradientCheckpointingFeature,
    LogitCapFeature,
    RouterAuxLossFeature,
    SequenceLengthPoolingFeature,
    TieEmbeddingsFeature,
)
from ._protocols import BaseModelProtocol


class BaseTaskModule[ModelT: BaseModelProtocol, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule, ABC):
    """Abstract base class for all task-specific modules.

    This class provides the foundation for creating task-specific model wrappers
    (e.g., ForCausalLM, ForSequenceClassification) with automatic registration,
    feature management, and consistent interfaces.

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol).
        ConfigT: The configuration type (must extend EasyMLXBaseConfig).

    Attributes:
        config (ConfigT): Model configuration containing hyperparameters.

    Class Attributes:
        _task_type (TaskType | None): The TaskType this module implements.
        _auto_register (bool): Whether to auto-register with the factory system.
        _model_type (str | None): Model type string for registration.
        _config_class (type | None): The configuration class for this model.
    """

    # Class variables for registration (to be set by subclasses)
    _task_type: TaskType | None = None
    _auto_register: bool = True
    _model_type: str | None = None
    _config_class: type | None = None

    def __init_subclass__(cls, **kwargs):
        """Handle automatic registration when subclassed.

        This method is called automatically by Python when a class inherits
        from BaseTaskModule. It registers the module with the factory system
        if auto-registration is enabled and all required class attributes are set.

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__.
        """
        super().__init_subclass__(**kwargs)

        # Auto-register if enabled and task type is specified
        if cls._auto_register and cls._task_type is not None:
            # Only register if this is a concrete subclass with a model type
            if cls._model_type is not None and cls._config_class is not None:
                # Apply registration decorator
                register_module(
                    task_type=cls._task_type,
                    config=cls._config_class,
                    model_type=cls._model_type,
                )(cls)

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        # Feature flags
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        router_aux_loss_coef: float | None = None,
        pooling_strategy: str = "last",
        # Head configuration
        head_bias: bool = False,
    ):
        """Initialize the base task module.

        Args:
            config: Model configuration containing hyperparameters such as
                hidden_size, vocab_size, num_layers, etc.
            base_model: Pre-instantiated base model instance. If provided,
                base_model_class is ignored.
            base_model_class: Base model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name under which to store the base model.
                Common values include "model", "transformer", "bert", etc.
                Defaults to "model".
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the output projection (LM head).
                Defaults to False.
            logit_cap: Maximum absolute value for output logits. If set,
                logits are clipped to [-logit_cap, logit_cap].
                Defaults to None (no capping).
            router_aux_loss_coef: Coefficient for Mixture-of-Experts router
                auxiliary loss. Defaults to None (no aux loss).
            pooling_strategy: Strategy for reducing sequence to single vector.
                Options: "last", "first", "mean", "max", "weighted_mean".
                Defaults to "last".
            head_bias: Whether to include bias in the task-specific head.
                Defaults to False.

        Raises:
            ValueError: If neither base_model nor base_model_class is provided.
        """
        super().__init__(config)

        # Create or store base model
        if base_model is None:
            if base_model_class is None:
                raise ValueError("Either base_model or base_model_class must be provided")
            base_model = base_model_class(config=config)

        # Store base model with custom attribute name
        setattr(self, base_model_name, base_model)
        self._base_model_name = base_model_name

        # Store head configuration
        self._head_bias = head_bias

        # Initialize features
        self._logit_cap_feature = LogitCapFeature(logit_cap) if logit_cap is not None else None
        self._tie_embeddings_feature = TieEmbeddingsFeature(tie_word_embeddings)
        self._router_aux_loss_feature = (
            RouterAuxLossFeature(router_aux_loss_coef) if router_aux_loss_coef is not None else None
        )

        pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = -1
        self._pooling_feature = SequenceLengthPoolingFeature(
            strategy=pooling_strategy,
            pad_token_id=pad_token_id,
        )
        self._gradient_checkpointing_feature = GradientCheckpointingFeature(
            policy=getattr(config, "gradient_checkpointing", None),
            save_names=getattr(config, "gradient_checkpointing_targets", None),
            exclude_names=getattr(config, "gradient_checkpointing_targets", None),
        )

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            ModelT: The base model instance.
        """
        return getattr(self, self._base_model_name)

    def apply_logit_cap(self, logits: Any) -> Any:
        """Apply logit capping if configured.

        Args:
            logits: Input logits tensor of any shape.

        Returns:
            Logits tensor with values clipped to [-cap, cap] if logit capping
            is enabled, otherwise returns the input unchanged.
        """
        if self._logit_cap_feature is not None:
            return self._logit_cap_feature.apply(logits)
        return logits

    def compute_router_aux_loss(self, outputs: Any) -> Any:
        """Compute router auxiliary loss for MoE models if configured.

        Args:
            outputs: Model outputs object that may contain ``all_router_losses``
                attribute with per-layer router loss values.

        Returns:
            The weighted sum of router losses, or None if MoE auxiliary loss
            is not configured or no router losses are available.
        """
        if self._router_aux_loss_feature is None:
            return None

        router_losses = getattr(outputs, "all_router_losses", None)
        if router_losses is None:
            return None

        return self._router_aux_loss_feature.compute_loss(router_losses)

    def pool_sequence(
        self,
        hidden_states: Any,
        input_ids: Any | None = None,
        attention_mask: Any | None = None,
    ) -> Any:
        """Pool sequence of hidden states to a single vector.

        Args:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_dim).
            input_ids: Optional tensor of shape (batch_size, sequence_length).
            attention_mask: Optional tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor of shape (batch_size, hidden_dim) with the pooled representation.
        """
        return self._pooling_feature.pool(hidden_states, input_ids, attention_mask=attention_mask)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the module.

        Must be implemented by subclasses to define the specific task logic.

        Returns:
            A task-specific output dataclass.
        """
        raise NotImplementedError

    def get_encoder(self) -> Any:
        """Return the encoder part of the model's architecture.

        Raises:
            NotImplementedError: For decoder-only models.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> Any:
        """Return the decoder part of the model's architecture."""
        return self.base_model.get_decoder()

    @abstractmethod
    def get_task_head(self) -> Any:
        """Return the task-specific output head.

        Returns:
            The task-specific head module (typically a Linear layer).
        """
        raise NotImplementedError

    def get_embedding(self) -> Any:
        """Return the input embedding layer of the model."""
        return self.base_model.get_embedding()
