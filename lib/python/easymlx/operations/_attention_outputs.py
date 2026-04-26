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

"""Attention output containers.

Provides a dataclass for wrapping the outputs of attention operations,
including the attention output tensor, optional attention weights, and
an optional cache view reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ._operation_impl import OperationOutput


@dataclass(slots=True)
class AttentionOutput(OperationOutput):
    """Output container for attention operations.

    Extends ``OperationOutput`` with attention-specific fields.

    Attributes:
        attention_outputs: The main attention output tensor of shape
            ``[batch, heads, seq_len, head_dim]`` or similar layout.
        attention_weights: Optional attention weight matrix, typically
            of shape ``[batch, heads, q_len, kv_len]``.
        cache_view: Optional reference to the KV cache view used or
            updated during the attention computation.
    """

    attention_outputs: mx.array | None = None
    attention_weights: mx.array | None = None
    cache_view: Any | None = None


__all__ = "AttentionOutput"
