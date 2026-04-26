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

"""General-purpose helpers for easymlx.

Provides environment-variable boolean checks, cache directory resolution,
and lightweight timing utilities for profiling code sections.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

_getenv = os.getenv


def check_bool_flag(name: str, default: bool = False) -> bool:
    """Check whether an environment variable represents a truthy boolean.

    The variable is considered truthy if its stripped, lowercased value is
    one of ``"1"``, ``"true"``, ``"yes"``, or ``"on"``.

    Args:
        name: Name of the environment variable to inspect.
        default: Value returned when the environment variable is not set.

    Returns:
        ``True`` if the environment variable is set to a truthy value,
        ``False`` otherwise (or ``default`` if unset).

    Example:
        >>> check_bool_flag("MY_FLAG")  # returns False if MY_FLAG is not set
        False
    """
    value = _getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_cache_dir(app_name: str = "easymlx") -> str:
    """Return the filesystem path for the application cache directory.

    Respects the ``EMLX_CACHE_HOME`` environment variable.  When it is
    not set, defaults to ``~/.cache/<app_name>``.

    Args:
        app_name: Subdirectory name appended to the cache root.

    Returns:
        Absolute path to the cache directory as a string.

    Example:
        >>> get_cache_dir()
        '/home/user/.cache/easymlx'
    """
    root = os.getenv("EMLX_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(root, app_name)


@contextmanager
def capture_time():
    """Context manager that captures wall-clock elapsed time.

    Yields a mutable dictionary containing an ``"elapsed"`` key whose
    value is updated to the total elapsed seconds when the context exits.

    Yields:
        A ``dict`` with a single key ``"elapsed"`` (initially ``0.0``).

    Example:
        >>> with capture_time() as t:
        ...     do_work()
        >>> print(t["elapsed"])
        0.123
    """
    start = time.perf_counter()
    payload = {"elapsed": 0.0}
    try:
        yield payload
    finally:
        payload["elapsed"] = max(time.perf_counter() - start, 0.0)


@dataclass
class Timer:
    """Small timing context manager.

    Measures wall-clock time between entering and exiting the context.
    The elapsed duration in seconds is stored in the :attr:`elapsed`
    attribute.

    Attributes:
        name: Optional label for the timer.
        elapsed: Elapsed time in seconds after the context exits.

    Example:
        >>> with Timer(name="load") as t:
        ...     load_model()
        >>> print(t.elapsed)
        1.234
    """

    name: str | None = None
    elapsed: float = 0.0

    def __enter__(self) -> Timer:
        """Enter the timing context and record the start time.

        Returns:
            The ``Timer`` instance itself.
        """
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the timing context and compute the elapsed duration.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.
        """
        del exc_type, exc, tb
        self.elapsed = max(time.perf_counter() - self._start, 0.0)


class Timers(dict[str, Timer]):
    """Collection of named timers stored in a ``dict``.

    Provides a convenience method to create and start a :class:`Timer`
    by name, storing it in the dictionary for later retrieval.

    Example:
        >>> timers = Timers()
        >>> timer = timers.start("preprocess")
        >>> # ... do work ...
        >>> timer.__exit__(None, None, None)
        >>> print(timers["preprocess"].elapsed)
    """

    def start(self, name: str) -> Timer:
        """Create, start, and store a new :class:`Timer` under the given name.

        Args:
            name: Key used to store the timer in this dictionary.

        Returns:
            The newly created and started :class:`Timer` instance.
        """
        timer = Timer(name=name)
        timer.__enter__()
        self[name] = timer
        return timer


__all__ = ("Timer", "Timers", "capture_time", "check_bool_flag", "get_cache_dir")
