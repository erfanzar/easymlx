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

"""Registry system for easymlx modules and configurations (serving-only).

Provides a central ``Registry`` for mapping model-type strings and task
types to their corresponding configuration and module classes. Decorators
:func:`register_config` and :func:`register_module` are the primary
public API for registering new model implementations.

Typical usage::

    @register_config("my_model")
    class MyModelConfig(EasyMLXBaseConfig):
        model_type = "my_model"
        ...

    @register_module(
        task_type=TaskType.CAUSAL_LM,
        config=MyModelConfig,
        model_type="my_model",
    )
    class MyModelForCausalLM(EasyMLXBaseModule):
        ...
"""

from __future__ import annotations

import inspect
import typing as tp
from dataclasses import dataclass
from enum import StrEnum

from .base_config import EasyMLXBaseConfig

if tp.TYPE_CHECKING:
    from .base_module import EasyMLXBaseModule

T = tp.TypeVar("T")


class ConfigType(StrEnum):
    """Enumeration of configuration registry namespaces.

    Attributes:
        MODULE_CONFIG: The default namespace for model configuration
            classes.
    """

    MODULE_CONFIG = "module-config"


class TaskType(StrEnum):
    """Enumeration of supported model task types.

    Each variant maps a human-readable task name to the string key
    used in the internal task registry.

    Attributes:
        CAUSAL_LM: Autoregressive causal language modelling.
        BASE_MODULE: Generic base module without a task-specific head.
        SEQUENCE_CLASSIFICATION: Sequence-level classification.
        TOKEN_CLASSIFICATION: Token-level classification (e.g. NER).
        QUESTION_ANSWERING: Extractive question answering.
        IMAGE_TEXT_TO_TEXT: Multimodal image+text to text generation.
        IMAGE_CLASSIFICATION: Image classification.
        EMBEDDING: Embedding / retrieval models.
        VISION_LM: Vision-language modelling.
        BASE_VISION: Base vision backbone module.
        SEQUENCE_TO_SEQUENCE: Encoder-decoder sequence to sequence.
    """

    CAUSAL_LM = "causal-language-model"
    BASE_MODULE = "base-module"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    IMAGE_CLASSIFICATION = "image-classification"
    EMBEDDING = "embedding"
    VISION_LM = "vision-language-model"
    BASE_VISION = "vision-module"
    SEQUENCE_TO_SEQUENCE = "sequence-to-sequence"


@dataclass(frozen=True, slots=True)
class ModuleRegistration:
    """Immutable record pairing a module class with its configuration class.

    Attributes:
        module: The ``EasyMLXBaseModule`` subclass.
        config: The ``EasyMLXBaseConfig`` subclass used by the module.
    """

    module: type[EasyMLXBaseModule]
    config: type[EasyMLXBaseConfig]


class Registry:
    """Central registry for easymlx configuration and module classes.

    Maintains two internal dictionaries:

    - ``_config_registry`` -- maps ``(ConfigType, model_type_str)`` to
      configuration classes.
    - ``_task_registry`` -- maps ``(TaskType, model_type_str)`` to
      :class:`ModuleRegistration` instances.
    """

    def __init__(self):
        """Initialize empty config and task registries."""
        self._config_registry: dict[ConfigType, dict[str, type[EasyMLXBaseConfig]]] = {ConfigType.MODULE_CONFIG: {}}
        self._task_registry: dict[TaskType, dict[str, ModuleRegistration]] = {task: {} for task in TaskType}

    def register_config(
        self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG
    ) -> tp.Callable[[T], T]:
        """Return a class decorator that registers a configuration class.

        Args:
            config_type: The model-type string key under which the
                configuration will be registered (e.g. ``"llama"``).
            config_field: The registry namespace. Defaults to
                ``ConfigType.MODULE_CONFIG``.

        Returns:
            A decorator that registers the decorated class and returns
            it unchanged.

        Raises:
            TypeError: If the decorated object is not a subclass of
                ``EasyMLXBaseConfig``.

        Example::

            @registry.register_config("my_model")
            class MyConfig(EasyMLXBaseConfig):
                model_type = "my_model"
        """

        def decorator(obj: T) -> T:
            if inspect.isclass(obj) and issubclass(tp.cast(type, obj), EasyMLXBaseConfig):
                self._config_registry[config_field][config_type] = tp.cast(type[EasyMLXBaseConfig], obj)
            else:
                raise TypeError("register_config can only be used with EasyMLXBaseConfig subclasses")
            return obj

        return decorator

    def register_module(
        self,
        *,
        task_type: TaskType,
        config: type[EasyMLXBaseConfig],
        model_type: str,
    ) -> tp.Callable[[T], T]:
        """Return a class decorator that registers a module class.

        The decorated class is also annotated with ``_model_task`` and
        ``_model_type`` class attributes.

        Args:
            task_type: The task under which this module is registered.
            config: The configuration class that accompanies this module.
            model_type: The model-type string key (e.g. ``"llama"``).

        Returns:
            A decorator that registers the decorated class and returns
            it unchanged.

        Raises:
            TypeError: If the decorated object is not a subclass of
                ``EasyMLXBaseModule``.

        Example::

            @registry.register_module(
                task_type=TaskType.CAUSAL_LM,
                config=MyConfig,
                model_type="my_model",
            )
            class MyModelForCausalLM(EasyMLXBaseModule):
                ...
        """

        def decorator(module: T) -> T:
            from .base_module import EasyMLXBaseModule

            if not (inspect.isclass(module) and issubclass(tp.cast(type, module), EasyMLXBaseModule)):
                raise TypeError("register_module can only be used with EasyMLXBaseModule subclasses")
            tp.cast(type, module)._model_task = task_type.value
            tp.cast(type, module)._model_type = model_type
            self._task_registry[task_type][model_type] = ModuleRegistration(
                module=tp.cast(type[EasyMLXBaseModule], module),
                config=config,
            )
            return module

        return decorator

    def get_config(
        self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG
    ) -> type[EasyMLXBaseConfig]:
        """Retrieve a registered configuration class.

        Args:
            config_type: The model-type string key.
            config_field: The registry namespace. Defaults to
                ``ConfigType.MODULE_CONFIG``.

        Returns:
            The registered ``EasyMLXBaseConfig`` subclass.

        Raises:
            KeyError: If no configuration is registered under the given
                ``config_type`` and ``config_field``.
        """
        try:
            return self._config_registry[config_field][config_type]
        except KeyError as exc:
            raise KeyError(f"No config registered for {config_type!r} under {config_field.value!r}") from exc

    def get_module_registration(self, task_type: TaskType, model_type: str) -> ModuleRegistration:
        """Retrieve a registered module and its associated configuration.

        Args:
            task_type: The task category to look up.
            model_type: The model-type string key.

        Returns:
            A :class:`ModuleRegistration` containing the module and
            config classes.

        Raises:
            KeyError: If no module is registered for the given
                ``task_type`` and ``model_type`` combination.
        """
        try:
            return self._task_registry[task_type][model_type]
        except KeyError as exc:
            raise KeyError(f"No module registered for task={task_type.value!r}, model_type={model_type!r}") from exc


registry = Registry()

register_config = registry.register_config
register_module = registry.register_module
