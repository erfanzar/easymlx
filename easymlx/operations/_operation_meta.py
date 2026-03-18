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

"""Operation metadata containers.

Provides a lightweight dataclass for carrying runtime hints about an
operation, such as preferred dtype and backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(slots=True)
class OperationMetadata:
    """Metadata describing runtime preferences for an operation.

    Attributes:
        runtime_dtype: Preferred data type for computation (e.g., ``mx.float16``).
        preferred_backend: Preferred execution backend name (e.g., ``"metal"``).
        name: Optional human-readable name for the operation instance.
    """

    runtime_dtype: mx.Dtype | None = None
    preferred_backend: str | None = None
    name: str | None = None


__all__ = "OperationMetadata"
