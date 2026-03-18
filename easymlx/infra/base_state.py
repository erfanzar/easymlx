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

"""Base state containers for easymlx infra.

Provides lightweight dataclass wrappers for carrying model or optimizer
state through the easymlx infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EasyMLXBaseState:
    """Generic container for model or optimizer state.

    This dataclass wraps a nested parameter dictionary that mirrors the
    structure produced by ``mlx.utils.tree_flatten`` /
    ``mlx.utils.tree_unflatten``.

    Attributes:
        params: A nested dictionary mapping parameter names to their
            values (typically ``mx.array`` instances).

    Example::

        state = EasyMLXBaseState(params={"layer.weight": mx.zeros((4, 4))})
    """

    params: dict[str, Any]


__all__ = "EasyMLXBaseState"
