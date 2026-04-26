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

"""Axis-name helpers mirroring the EasyDeL top-level layout.

Provides canonical string constants for tensor axis names used throughout the
easymlx codebase. These constants ensure consistent naming when referencing
batch, sequence, head, hidden, and expert dimensions.

Attributes:
    BATCH_AXIS: Axis name for the batch dimension (``"batch"``).
    SEQUENCE_AXIS: Axis name for the sequence/time dimension (``"sequence"``).
    HEAD_AXIS: Axis name for the attention head dimension (``"head"``).
    HIDDEN_AXIS: Axis name for the hidden/feature dimension (``"hidden"``).
    EXPERT_AXIS: Axis name for the mixture-of-experts dimension (``"expert"``).
"""

from __future__ import annotations

BATCH_AXIS = "batch"
SEQUENCE_AXIS = "sequence"
HEAD_AXIS = "head"
HIDDEN_AXIS = "hidden"
EXPERT_AXIS = "expert"

__all__ = ("BATCH_AXIS", "EXPERT_AXIS", "HEAD_AXIS", "HIDDEN_AXIS", "SEQUENCE_AXIS")
