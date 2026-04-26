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

"""Lifecycle helpers for local eSurge instances.

Thin wrapper functions that delegate to the corresponding methods on an
:class:`~easymlx.inference.esurge.esurge_engine.eSurge` instance, allowing
callers to manage engine state without direct method access.
"""

from __future__ import annotations

from ..esurge_engine import eSurge


def pause_engine(engine: eSurge) -> None:
    """Pause the engine's background scheduling loop.

    Args:
        engine: The eSurge engine instance to pause.
    """
    engine.pause()


def resume_engine(engine: eSurge) -> None:
    """Resume the engine's background scheduling loop.

    Args:
        engine: The eSurge engine instance to resume.
    """
    engine.resume()


def terminate_engine(engine: eSurge) -> None:
    """Terminate the engine, releasing all resources.

    Args:
        engine: The eSurge engine instance to terminate.
    """
    engine.terminate()


def close_engine(engine: eSurge) -> None:
    """Gracefully close the engine, flushing in-flight requests.

    Args:
        engine: The eSurge engine instance to close.
    """
    engine.close()


__all__ = ("close_engine", "pause_engine", "resume_engine", "terminate_engine")
