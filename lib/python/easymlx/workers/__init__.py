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

"""Worker namespaces for easymlx.

Provides lazy access to worker sub-packages (``esurge`` and
``response_store``) and eagerly exports :func:`get_logger`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .loggers import get_logger

__all__ = ("esurge", "get_logger", "response_store")


def __getattr__(name: str) -> Any:
    """Lazily import worker sub-packages on first access.

    Supports ``esurge`` and ``response_store`` as deferred imports so
    their heavy dependencies are only loaded when actually needed.

    Args:
        name: The attribute name being accessed on this module.

    Returns:
        The lazily imported sub-module.

    Raises:
        AttributeError: If ``name`` is not a recognized sub-package.
    """
    if name in {"esurge", "response_store"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
