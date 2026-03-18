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

"""Operation requirements exports.

This package provides the requirements system for attention operations,
including metadata field flags, execution mode enums, cache type declarations,
a fluent builder for constructing requirements, and validation utilities.
"""

from __future__ import annotations

from .builder import RequirementsBuilder
from .requirements import OperationRequirements
from .types import CacheType, ExecutionMode, MetadataField
from .validation import available_metadata_fields, iter_required_field_names, missing_required_fields, validate_metadata

__all__ = (
    "CacheType",
    "ExecutionMode",
    "MetadataField",
    "OperationRequirements",
    "RequirementsBuilder",
    "available_metadata_fields",
    "iter_required_field_names",
    "missing_required_fields",
    "validate_metadata",
)
