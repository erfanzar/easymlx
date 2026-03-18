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

"""Base operation and registry for MLX attention kernels.

Defines the abstract ``BaseOperation`` class that all attention kernels must
implement, and the ``OperationRegistry`` singleton for discovering and
instantiating operations by name.
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

from ._operation_meta import OperationMetadata
from .requirements import ExecutionMode, OperationRequirements

if tp.TYPE_CHECKING:
    pass


class BaseOperation(ABC):
    """Abstract base class for all attention operation implementations.

    Subclasses must implement ``get_impl_name`` and ``forward_native``.
    Operations are callable and delegate to ``forward_native``.

    Attributes:
        metadata: Optional metadata describing runtime preferences
            such as dtype and backend.
    """

    metadata: OperationMetadata | None = None

    def __init__(self, metadata: OperationMetadata | None = None):
        """Initialize the operation with optional metadata.

        Args:
            metadata: Optional operation metadata for runtime configuration.
        """
        self.metadata = metadata

    @classmethod
    @abstractmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the name(s) under which this operation is registered.

        Returns:
            A single name string or a tuple of alias strings.
        """
        ...

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Return the metadata and cache requirements for this operation.

        Args:
            mode: The execution mode (prefill, decode, or mixed).

        Returns:
            An ``OperationRequirements`` instance describing required and
            optional metadata fields and cache types.
        """
        name = cls.get_impl_name()
        if isinstance(name, tuple):
            name = name[0]
        return OperationRequirements.default(str(name))

    def get_instance_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Return requirements for this specific instance.

        Args:
            mode: The execution mode.

        Returns:
            An ``OperationRequirements`` instance.
        """
        return self.get_requirements(mode)

    def get_impl_metadata(self) -> OperationMetadata | None:
        """Return the metadata associated with this operation instance.

        Returns:
            The ``OperationMetadata`` if set, otherwise None.
        """
        return self.metadata

    @abstractmethod
    def forward_native(self, *args, **kwargs) -> tp.Any:
        """Execute the attention computation.

        Subclasses must implement this method with the actual kernel logic.

        Args:
            *args: Positional arguments (implementation-specific).
            **kwargs: Keyword arguments (implementation-specific).

        Returns:
            An ``AttentionOutput`` or similar result container.
        """
        ...

    def __call__(self, *args, **kwargs) -> tp.Any:
        """Invoke the operation by delegating to ``forward_native``.

        Args:
            *args: Positional arguments forwarded to ``forward_native``.
            **kwargs: Keyword arguments forwarded to ``forward_native``.

        Returns:
            The result of ``forward_native``.
        """
        return self.forward_native(*args, **kwargs)


class OperationRegistry:
    """Global registry for attention operation implementations.

    Operations register themselves via ``@OperationRegistry.register`` and
    can be looked up or instantiated by name.
    """

    _registry: tp.ClassVar[dict[str, type[BaseOperation]]] = {}

    @classmethod
    def register(cls, operation_cls: type[BaseOperation]) -> type[BaseOperation]:
        """Register an operation class under all of its implementation names.

        Intended to be used as a class decorator.

        Args:
            operation_cls: The operation class to register.

        Returns:
            The same class, unmodified.
        """
        names = operation_cls.get_impl_name()
        if isinstance(names, str):
            names = (names,)
        for name in names:
            cls._registry[str(name)] = operation_cls
        return operation_cls

    @classmethod
    def get(cls, name: str) -> type[BaseOperation]:
        """Look up a registered operation class by name.

        Args:
            name: The registered name of the operation.

        Returns:
            The operation class.

        Raises:
            KeyError: If no operation is registered under the given name.
        """
        try:
            return cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown operation '{name}'. Available: {available}") from exc

    @classmethod
    def create(cls, name: str, *, metadata: OperationMetadata | None = None) -> BaseOperation:
        """Create an operation instance by name.

        Args:
            name: The registered name of the operation.
            metadata: Optional metadata to pass to the constructor.

        Returns:
            A new instance of the requested operation.

        Raises:
            KeyError: If no operation is registered under the given name.
        """
        return cls.get(name)(metadata=metadata)

    @classmethod
    def available_impls(cls) -> list["str"]:
        """Return a sorted list of all registered operation names.

        Returns:
            A list of registered implementation name strings.
        """
        return sorted(cls._registry)


Operation = BaseOperation

__all__ = ("BaseOperation", "Operation", "OperationRegistry")
