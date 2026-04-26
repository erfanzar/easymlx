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

"""Package-local scripts mirroring EasyDeL.

This module exposes checkpoint conversion utilities for converting
Hugging Face model checkpoints into the easymlx format. Both single
and batch conversion helpers are provided.
"""

from __future__ import annotations

from .convert_hf_to_easymlx import convert_hf_checkpoint
from .convert_hf_to_easymlx_batch import convert_hf_checkpoints

__all__ = ("convert_hf_checkpoint", "convert_hf_checkpoints")
