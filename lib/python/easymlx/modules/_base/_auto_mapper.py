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

"""Auto-mapper for dynamically creating task-specific model classes (MLX).

Factory functions that generate ForCausalLM, ForSequenceClassification, etc.
classes from base models with minimal boilerplate.
"""

from __future__ import annotations

from typing import Any, TypeVar

from easymlx.infra.factory import TaskType

from .causal_lm_module import BaseCausalLMModule
from .conditional_generation_module import BaseConditionalGenerationModule
from .embedding_module import BaseEmbeddingModule
from .image_classification_module import BaseImageClassificationModule
from .question_answering_module import BaseQuestionAnsweringModule
from .sequence_classification_module import BaseSequenceClassificationModule
from .token_classification_module import BaseTokenClassificationModule

ModelT = TypeVar("ModelT")
ConfigT = TypeVar("ConfigT")


def create_causal_lm_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseCausalLMModule]:
    """Create a ForCausalLM class dynamically.

    Args:
        model_name: Name prefix (e.g. "Llama" -> "LlamaForCausalLM").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model.
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForCausalLM class.
    """
    class_name = f"{model_name}ForCausalLM"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseCausalLMModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.CAUSAL_LM,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_sequence_classification_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseSequenceClassificationModule]:
    """Create a ForSequenceClassification class dynamically.

    Args:
        model_name: Name prefix (e.g. "Llama" -> "LlamaForSequenceClassification").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to "model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForSequenceClassification class.
    """
    class_name = f"{model_name}ForSequenceClassification"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseSequenceClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.SEQUENCE_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_token_classification_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseTokenClassificationModule]:
    """Create a ForTokenClassification class dynamically.

    Args:
        model_name: Name prefix (e.g. "Llama" -> "LlamaForTokenClassification").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to "model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForTokenClassification class.
    """
    class_name = f"{model_name}ForTokenClassification"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseTokenClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.TOKEN_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_question_answering_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseQuestionAnsweringModule]:
    """Create a ForQuestionAnswering class dynamically.

    Args:
        model_name: Name prefix (e.g. "Llama" -> "LlamaForQuestionAnswering").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to "model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForQuestionAnswering class.
    """
    class_name = f"{model_name}ForQuestionAnswering"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseQuestionAnsweringModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.QUESTION_ANSWERING,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_conditional_generation_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseConditionalGenerationModule]:
    """Create a ForConditionalGeneration class dynamically.

    Args:
        model_name: Name prefix (e.g. "T5" -> "T5ForConditionalGeneration").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to "model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForConditionalGeneration class.
    """
    class_name = f"{model_name}ForConditionalGeneration"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseConditionalGenerationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.SEQUENCE_TO_SEQUENCE,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_image_classification_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "vision_model",
    **default_feature_kwargs: Any,
) -> type[BaseImageClassificationModule]:
    """Create a ForImageClassification class dynamically.

    Args:
        model_name: Name prefix (e.g. "ViT" -> "ViTForImageClassification").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to
            "vision_model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForImageClassification class.
    """
    class_name = f"{model_name}ForImageClassification"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseImageClassificationModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.IMAGE_CLASSIFICATION,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


def create_embedding_class[ModelT, ConfigT](
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
) -> type[BaseEmbeddingModule]:
    """Create a ForEmbedding class dynamically.

    Args:
        model_name: Name prefix (e.g. "BERT" -> "BERTForEmbedding").
        base_model_class: The base model class to wrap.
        config_class: Config class for registry.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model. Defaults to "model".
        **default_feature_kwargs: Defaults merged at instantiation.

    Returns:
        Dynamically created ForEmbedding class.
    """
    class_name = f"{model_name}ForEmbedding"

    def __init__(self, config, **kwargs):
        merged = {**default_feature_kwargs, **kwargs}
        super(type(self), self).__init__(
            config=config,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            **merged,
        )

    cls = type(
        class_name,
        (BaseEmbeddingModule,),
        {
            "__init__": __init__,
            "__module__": base_model_class.__module__,
            "__qualname__": class_name,
            "_task_type": TaskType.EMBEDDING,
            "_model_type": model_type,
            "_config_class": config_class,
        },
    )
    return cls


AUTO_MODEL_FACTORY_REGISTRY: dict[TaskType, Any] = {
    TaskType.CAUSAL_LM: create_causal_lm_class,
    TaskType.SEQUENCE_CLASSIFICATION: create_sequence_classification_class,
    TaskType.TOKEN_CLASSIFICATION: create_token_classification_class,
    TaskType.QUESTION_ANSWERING: create_question_answering_class,
    TaskType.SEQUENCE_TO_SEQUENCE: create_conditional_generation_class,
    TaskType.IMAGE_CLASSIFICATION: create_image_classification_class,
    TaskType.EMBEDDING: create_embedding_class,
}
"""Registry mapping TaskType to factory function."""


def create_task_model_class[ModelT, ConfigT](
    task_type: TaskType,
    model_name: str,
    base_model_class: type[ModelT],
    config_class: type[ConfigT],
    model_type: str,
    base_model_name: str = "model",
    **default_feature_kwargs: Any,
):
    """Create a task-specific model class using the appropriate factory.

    Args:
        task_type: Task type determining the factory function.
        model_name: Name prefix for the generated class.
        base_model_class: Base model class to wrap.
        config_class: Configuration class for this model.
        model_type: Model type string for registry.
        base_model_name: Attribute name for the base model.
        **default_feature_kwargs: Defaults for the factory function.

    Returns:
        A new task-specific model class.
    """
    if task_type not in AUTO_MODEL_FACTORY_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}. Supported: {list(AUTO_MODEL_FACTORY_REGISTRY.keys())}")
    factory_fn = AUTO_MODEL_FACTORY_REGISTRY[task_type]
    return factory_fn(
        model_name=model_name,
        base_model_class=base_model_class,
        config_class=config_class,
        model_type=model_type,
        base_model_name=base_model_name,
        **default_feature_kwargs,
    )
