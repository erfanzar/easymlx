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

"""Validation helpers for operation requirements.

Provides functions to inspect metadata objects for available fields,
check for missing required fields, and validate metadata against
an operation's declared requirements.
"""

from __future__ import annotations

from collections.abc import Iterable

from .requirements import OperationRequirements
from .types import MetadataField

_FIELD_NAMES = {
    MetadataField.MASK: ("mask",),
    MetadataField.SINKS: ("sinks",),
    MetadataField.QUERY_START_LOC: ("query_start_loc",),
    MetadataField.BLOCK_TABLES: ("block_tables",),
    MetadataField.KV_LENS: ("kv_lens",),
    MetadataField.BLOCK_SIZE: ("block_size",),
    MetadataField.SLIDING_WINDOW: ("sliding_window",),
}


def available_metadata_fields(metadata: object | None) -> MetadataField:
    """Determine which metadata fields are present on an object.

    Inspects the object for known attribute names and returns a combined
    ``MetadataField`` flag for all fields found with non-None values.

    Args:
        metadata: An object to inspect, or None.

    Returns:
        A ``MetadataField`` flag representing all available fields.
    """
    if metadata is None:
        return MetadataField.NONE
    available = MetadataField.NONE
    for field, attr_names in _FIELD_NAMES.items():
        for attr_name in attr_names:
            if getattr(metadata, attr_name, None) is not None:
                available |= field
                break
    return available


def missing_required_fields(metadata: object | None, requirements: OperationRequirements) -> MetadataField:
    """Compute which required fields are missing from the metadata.

    Args:
        metadata: An object to check, or None.
        requirements: The operation requirements declaring required fields.

    Returns:
        A ``MetadataField`` flag representing the missing required fields.
        Returns ``MetadataField.NONE`` if all required fields are present.
    """
    return requirements.required_metadata & ~available_metadata_fields(metadata)


def validate_metadata(metadata: object | None, requirements: OperationRequirements) -> None:
    """Validate that metadata satisfies all required fields.

    Args:
        metadata: An object to validate, or None.
        requirements: The operation requirements to validate against.

    Raises:
        ValueError: If any required metadata fields are missing, listing
            the missing field names.
    """
    missing = missing_required_fields(metadata, requirements)
    if missing is MetadataField.NONE:
        return
    fields = [field.name.lower() for field in MetadataField if field not in {MetadataField.NONE} and field in missing]
    raise ValueError(f"Missing required metadata for {requirements.name}: {', '.join(fields)}")


def iter_required_field_names(requirements: OperationRequirements) -> Iterable["str"]:
    """Iterate over the attribute names of required metadata fields.

    Args:
        requirements: The operation requirements.

    Yields:
        The primary attribute name string for each required field.
    """
    for field, attr_names in _FIELD_NAMES.items():
        if field is MetadataField.NONE or field not in requirements.required_metadata:
            continue
        yield attr_names[0]


__all__ = ("available_metadata_fields", "iter_required_field_names", "missing_required_fields", "validate_metadata")
