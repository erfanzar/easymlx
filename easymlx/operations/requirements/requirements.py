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

"""Core requirement dataclasses.

Provides the frozen ``OperationRequirements`` dataclass that declares what
metadata fields and cache types an attention operation needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MetadataField


@dataclass(frozen=True)
class OperationRequirements:
    """Immutable declaration of metadata and cache requirements for an operation.

    Supports combining requirements from multiple operations via the ``|``
    operator (union of required/optional fields and cache needs).

    Attributes:
        name: Human-readable name or combined name of the operation(s).
        required_metadata: Metadata fields that must be present.
        optional_metadata: Metadata fields that may be present.
        requires_cache: Whether a KV cache is required.
        cache_view_cls: Specific cache view class required, or None for any.
    """

    name: str = ""
    required_metadata: MetadataField = MetadataField.NONE
    optional_metadata: MetadataField = MetadataField.NONE
    requires_cache: bool = False
    cache_view_cls: type | None = None

    def __or__(self, other: OperationRequirements) -> OperationRequirements:
        """Combine two requirement sets via union.

        Args:
            other: The other requirements to merge with.

        Returns:
            A new ``OperationRequirements`` with merged fields, names, and
            cache requirements.
        """
        cache_view_cls = self.cache_view_cls
        if cache_view_cls is None:
            cache_view_cls = other.cache_view_cls
        elif other.cache_view_cls is not None and cache_view_cls is not other.cache_view_cls:
            cache_view_cls = object
        return OperationRequirements(
            name="+".join(part for part in (self.name, other.name) if part),
            required_metadata=self.required_metadata | other.required_metadata,
            optional_metadata=self.optional_metadata | other.optional_metadata,
            requires_cache=self.requires_cache or other.requires_cache,
            cache_view_cls=cache_view_cls,
        )

    @classmethod
    def create(
        cls,
        *,
        name: str = "",
        required_metadata: MetadataField = MetadataField.NONE,
        optional_metadata: MetadataField = MetadataField.NONE,
        requires_cache: bool = False,
        cache_view_cls: type | None = None,
    ) -> OperationRequirements:
        """Create a new requirements instance with explicit field values.

        Args:
            name: Operation name.
            required_metadata: Required metadata field flags.
            optional_metadata: Optional metadata field flags.
            requires_cache: Whether a cache is required.
            cache_view_cls: Specific cache view class, or None.

        Returns:
            A new ``OperationRequirements`` instance.
        """
        return cls(
            name=name,
            required_metadata=required_metadata,
            optional_metadata=optional_metadata,
            requires_cache=requires_cache,
            cache_view_cls=cache_view_cls,
        )

    @classmethod
    def default(cls, name: str = "") -> OperationRequirements:
        """Create a default requirements instance with basic optional metadata.

        Args:
            name: Operation name.

        Returns:
            An ``OperationRequirements`` with mask and sinks as optional fields.
        """
        return cls.create(name=name, optional_metadata=MetadataField.basic())


__all__ = "OperationRequirements"
