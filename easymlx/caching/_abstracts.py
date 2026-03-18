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

"""Abstract cache hierarchy for easymlx.

Three-tier design mirroring EasyDeL:
  BaseCacheConfig  — static, serializable configuration
  BaseCacheView    — single-layer mutable state (key/value or recurrent)
  BaseCache        — multi-layer container (list of views)

Plus OperationsMetadata which wraps any cache-type metadata so that
operations can consume it uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


class BaseCacheConfig(ABC):
    """Static, serializable description of how to build a cache.

    Subclasses hold all parameters needed to allocate a cache (batch size,
    number of layers, head dimensions, dtype, etc.) and expose a :meth:`create`
    factory for keyword-based construction.
    """

    @classmethod
    @abstractmethod
    def create(cls, **kwargs: Any) -> BaseCacheConfig:
        """Create a new cache configuration from keyword arguments.

        Args:
            **kwargs: Configuration parameters specific to the cache type.

        Returns:
            A new ``BaseCacheConfig`` subclass instance.
        """
        ...


class BaseCacheView(ABC):
    """Mutable state for a single layer.

    Each view holds the cached key/value tensors (or recurrent states) for one
    layer of a model. Views are owned by a :class:`BaseCache` container.
    """

    @abstractmethod
    def concatenate_to_cache(
        self,
        key_states: mx.array,
        value_states: mx.array,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array, BaseCacheView]:
        """Append new K/V states and return the full cache contents.

        Subclasses may accept additional keyword arguments (e.g. ``mask``,
        ``cache_metadata``, ``query_states``).

        Args:
            key_states: New key tensor to append to the cache.
            value_states: New value tensor to append to the cache.
            **kwargs: Additional keyword arguments consumed by specific
                cache implementations.

        Returns:
            A three-tuple of ``(full_keys, full_values, updated_view)`` where
            the first two elements are the complete (possibly truncated) cached
            key and value tensors, and the third is the mutated view itself.
        """
        ...

    @property
    def offset(self) -> int:
        """Return the current write position (number of tokens already cached)."""
        return 0


class BaseCache(ABC):
    """Multi-layer container that owns a list of :class:`BaseCacheView`.

    Provides list-like access (indexing, iteration, ``len``) over per-layer
    cache views, plus factory and reset methods.
    """

    @property
    @abstractmethod
    def views(self) -> list[BaseCacheView | None]:
        """Return the ordered per-layer cache views."""
        ...

    def __getitem__(self, index: int) -> BaseCacheView | None:
        """Retrieve the cache view for a specific layer.

        Args:
            index: Zero-based layer index.

        Returns:
            The :class:`BaseCacheView` for the given layer, or ``None`` if
            the layer has no active cache.
        """
        return self.views[index]

    def __setitem__(self, index: int, view: BaseCacheView | None) -> None:
        """Replace the cache view for a specific layer.

        Args:
            index: Zero-based layer index.
            view: The new cache view, or ``None`` to clear the layer.
        """
        self.views[index] = view  # type: ignore[index]

    def __len__(self) -> int:
        """Return the number of layers in the cache."""
        return len(self.views)

    def __iter__(self):
        """Iterate over per-layer cache views."""
        return iter(self.views)

    @classmethod
    @abstractmethod
    def init_cache(cls, config: BaseCacheConfig, **kwargs: Any) -> BaseCache:
        """Allocate an empty cache from a configuration object.

        Args:
            config: Static cache configuration describing dimensions,
                dtype, and layer count.
            **kwargs: Additional keyword arguments for subclass-specific
                initialization.

        Returns:
            A freshly allocated :class:`BaseCache` with empty views.
        """
        ...

    def clear(self) -> None:
        """Reset all layer views to ``None``."""
        for i in range(len(self.views)):
            self.views[i] = None  # type: ignore[index]


@dataclass(slots=True)
class OperationsMetadata:
    """Thin wrapper that lets operations consume cache metadata uniformly.

    This mirrors EasyDeL's ``OperationsMetadata`` which can represent
    transformer, "paged", recurrent, or hybrid cache metadata under one type.
    """

    cache_type: str
    metadata: Any = None
    cache_view: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_transformer(
        cls,
        *,
        cache_view: Any,
        offset: int = 0,
        mask: mx.array | None = None,
    ) -> OperationsMetadata:
        """Create metadata for a standard transformer KV cache operation.

        Args:
            cache_view: The :class:`TransformerCacheView` for the current layer.
            offset: Number of tokens already cached before this step.
            mask: Optional attention mask array.

        Returns:
            An :class:`OperationsMetadata` with ``cache_type="transformer"``.
        """
        return cls(
            cache_type="transformer",
            metadata={"offset": offset, "mask": mask},
            cache_view=cache_view,
        )

    @classmethod
    def for_paged(
        cls,
        *,
        cache_view: Any,
        page_metadata: Any,
    ) -> OperationsMetadata:
        """Create metadata for a paged KV cache operation.

        Args:
            cache_view: The paged cache view for the current layer.
            page_metadata: A :class:`PageMetadata` instance with block tables
                and KV lengths.

        Returns:
            An :class:`OperationsMetadata` with ``cache_type="paged"``.
        """
        return cls(
            cache_type="paged",
            metadata=page_metadata,
            cache_view=cache_view,
        )

    @classmethod
    def for_recurrent(
        cls,
        *,
        cache_view: Any,
    ) -> OperationsMetadata:
        """Create metadata for a recurrent / SSM cache operation.

        Args:
            cache_view: The :class:`RecurrentCacheView` for the current layer.

        Returns:
            An :class:`OperationsMetadata` with ``cache_type="recurrent"``.
        """
        return cls(
            cache_type="recurrent",
            cache_view=cache_view,
        )

    @classmethod
    def for_hybrid(
        cls,
        *,
        cache_view: Any,
        layer_type: str,
    ) -> OperationsMetadata:
        """Create metadata for a hybrid (attention + recurrent) cache operation.

        Args:
            cache_view: The :class:`HybridCacheView` for the current layer.
            layer_type: One of ``"full_attention"``, ``"linear_attention"``,
                or ``"parallel_hybrid"``.

        Returns:
            An :class:`OperationsMetadata` with ``cache_type="hybrid"``.
        """
        return cls(
            cache_type="hybrid",
            metadata={"layer_type": layer_type},
            cache_view=cache_view,
        )


__all__ = ("BaseCache", "BaseCacheConfig", "BaseCacheView", "OperationsMetadata")
