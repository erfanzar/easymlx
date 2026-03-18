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

"""Low-level attention operations for easymlx.

This package provides a pluggable attention operation framework for MLX.
It includes base classes for defining attention operations, a registry for
discovering them by name, concrete kernel implementations (scaled dot-product,
vanilla, and paged/unified attention), a requirements system for declaring
metadata and cache dependencies, and an executor for mode-aware dispatch.
"""

from __future__ import annotations

from ._attention_outputs import AttentionOutput
from ._base_operation import BaseOperation, Operation, OperationRegistry
from ._operation_impl import OperationImpl, OperationOutput
from ._operation_meta import OperationMetadata
from .executor import OperationExecutor
from .kernels import (
    ScaledDotProductAttention,
    ScaledDotProductAttn,
    UnifiedAttention,
    UnifiedAttn,
    UnifiedAttnConfig,
    UnifiedAttnMetadata,
    Vanilla,
    VanillaAttention,
    VanillaAttn,
    paged_attention,
)
from .requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
    available_metadata_fields,
    iter_required_field_names,
    missing_required_fields,
    validate_metadata,
)

__all__ = (
    "AttentionOutput",
    "BaseOperation",
    "CacheType",
    "ExecutionMode",
    "MetadataField",
    "Operation",
    "OperationExecutor",
    "OperationImpl",
    "OperationMetadata",
    "OperationOutput",
    "OperationRegistry",
    "OperationRequirements",
    "RequirementsBuilder",
    "ScaledDotProductAttention",
    "ScaledDotProductAttn",
    "UnifiedAttention",
    "UnifiedAttn",
    "UnifiedAttnConfig",
    "UnifiedAttnMetadata",
    "Vanilla",
    "VanillaAttention",
    "VanillaAttn",
    "available_metadata_fields",
    "iter_required_field_names",
    "missing_required_fields",
    "paged_attention",
    "validate_metadata",
)
