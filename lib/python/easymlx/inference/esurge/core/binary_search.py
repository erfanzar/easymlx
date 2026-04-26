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

"""Small binary-search helpers used by scheduler-style components.

Thin wrappers around :func:`bisect.bisect_left` and :func:`bisect.bisect_right`
providing a consistent API for lower-bound and upper-bound lookups.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any


def lower_bound(values: list[Any], target: Any) -> int:
    """Return the leftmost insertion index for *target* in a sorted list.

    Equivalent to ``bisect.bisect_left``. All elements to the left of the
    returned index are strictly less than *target*.

    Args:
        values: A sorted list of comparable elements.
        target: The value to search for.

    Returns:
        The insertion index such that ``values[:idx] < target``.

    Example:
        >>> lower_bound([1, 3, 5, 7], 5)
        2
    """
    return bisect_left(values, target)


def upper_bound(values: list[Any], target: Any) -> int:
    """Return the rightmost insertion index for *target* in a sorted list.

    Equivalent to ``bisect.bisect_right``. All elements to the left of the
    returned index are less than or equal to *target*.

    Args:
        values: A sorted list of comparable elements.
        target: The value to search for.

    Returns:
        The insertion index such that ``values[:idx] <= target``.

    Example:
        >>> upper_bound([1, 3, 5, 7], 5)
        3
    """
    return bisect_right(values, target)


__all__ = ("lower_bound", "upper_bound")
