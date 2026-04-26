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

"""Simple nested-structure traversal helpers.

Provides JAX-style ``tree_map`` for recursively applying a function
over nested Python containers (dicts, lists, tuples).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    """Apply a function to every leaf of a nested Python structure.

    Recursively traverses ``dict``, ``list``, and ``tuple`` containers,
    applying ``fn`` to every non-container leaf.  The container types
    are preserved in the output.

    Args:
        fn: A callable applied to each leaf value.
        tree: A (possibly nested) structure of dicts, lists, tuples,
            and leaf values.

    Returns:
        A new structure with the same nesting but with every leaf
        replaced by ``fn(leaf)``.

    Example:
        >>> tree_map(lambda x: x * 2, {"a": 1, "b": [2, 3]})
        {'a': 2, 'b': [4, 6]}
    """
    if isinstance(tree, dict):
        return {key: tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [tree_map(fn, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, value) for value in tree)
    return fn(tree)


__all__ = "tree_map"
