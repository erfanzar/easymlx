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

"""Fluent builders for operation requirements.

Provides ``RequirementsBuilder``, a chainable API for constructing
``OperationRequirements`` instances in a readable way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .requirements import OperationRequirements
from .types import MetadataField


@dataclass(slots=True)
class RequirementsBuilder:
    """Fluent builder for constructing ``OperationRequirements``.

    Example:
        >>> reqs = (
        ...     RequirementsBuilder("my_op")
        ...     .require_metadata(MetadataField.MASK)
        ...     .optional_metadata(MetadataField.SINKS)
        ...     .build()
        ... )

    Attributes:
        name: Name of the operation these requirements describe.
    """

    name: str
    _required: MetadataField = MetadataField.NONE
    _optional: MetadataField = MetadataField.NONE
    _requires_cache: bool = False
    _cache_view_cls: type | None = None

    def require_metadata(self, *fields: MetadataField) -> RequirementsBuilder:
        """Mark metadata fields as required.

        Args:
            *fields: One or more ``MetadataField`` flags to require.

        Returns:
            Self for method chaining.
        """
        for field in fields:
            self._required |= field
        return self

    def optional_metadata(self, *fields: MetadataField) -> RequirementsBuilder:
        """Mark metadata fields as optional.

        Args:
            *fields: One or more ``MetadataField`` flags to mark optional.

        Returns:
            Self for method chaining.
        """
        for field in fields:
            self._optional |= field
        return self

    def require_cache(self, cache_view_cls: type | None = None) -> RequirementsBuilder:
        """Declare that this operation requires a KV cache.

        Args:
            cache_view_cls: Optional specific cache view class required.

        Returns:
            Self for method chaining.
        """
        self._requires_cache = True
        self._cache_view_cls = cache_view_cls
        return self

    def build(self) -> OperationRequirements:
        """Construct the finalized ``OperationRequirements``.

        Returns:
            A frozen ``OperationRequirements`` instance.
        """
        return OperationRequirements.create(
            name=self.name,
            required_metadata=self._required,
            optional_metadata=self._optional,
            requires_cache=self._requires_cache,
            cache_view_cls=self._cache_view_cls,
        )


__all__ = "RequirementsBuilder"
