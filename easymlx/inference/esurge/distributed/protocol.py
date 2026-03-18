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

"""Distributed control-plane protocol constants and hashing helpers.

Defines the RPC command vocabulary (``CMD_*``), status codes (``STATUS_*``),
the :class:`WorkerInfo` descriptor, and deterministic hashing utilities
used by both leader and worker nodes to verify lockstep execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CMD_HELLO = "hello"
CMD_HEALTH = "health"
CMD_STEP = "step"
CMD_SHUTDOWN = "shutdown"

STATUS_OK = "ok"
STATUS_ERROR = "error"


@dataclass(slots=True)
class WorkerInfo:
    """Distributed worker descriptor.

    Attributes:
        worker_id: Unique string identifier for this worker.
        host: Hostname or IP address of the worker.
        port: Control-plane port number.
        rank: Optional rank index within the distributed world.
    """

    worker_id: str
    host: str = "localhost"
    port: int = 0
    rank: int | None = None

    @property
    def endpoint(self) -> str:
        """Return the ZMQ TCP endpoint string for this worker.

        Returns:
            A string of the form ``"tcp://<host>:<port>"``.
        """
        return f"tcp://{self.host}:{self.port}"


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalize a value into JSON-serializable form.

    Handles primitives, bytes, dicts, lists, tuples, sets, and objects
    exposing ``.tolist()`` or ``.item()`` (e.g. numpy scalars/arrays).

    Args:
        value: Arbitrary Python value to canonicalize.

    Returns:
        A JSON-serializable equivalent of *value*.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return [_canonicalize(v) for v in sorted(value, key=repr)]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _canonicalize(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonicalize(item())
        except Exception:
            pass

    return repr(value)


def _hash_payload(payload: Any) -> str:
    """Compute a SHA-256 hex digest over a canonicalized JSON encoding.

    Args:
        payload: Arbitrary value to hash (will be canonicalized first).

    Returns:
        Lowercase hexadecimal SHA-256 digest string.
    """
    encoded = json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Build a stable SHA-256 fingerprint for configuration payloads.

    Used during the leader/worker handshake to verify that both sides are
    running with identical configurations.

    Args:
        config: A mapping of configuration key-value pairs.

    Returns:
        A hexadecimal SHA-256 digest string.
    """

    return _hash_payload(dict(config))


def compute_sampled_digest(req_ids: list["str"], sampled_token_ids: list[list["int"]]) -> str:
    """Build a stable SHA-256 digest for sampled outputs in lockstep checks.

    The leader computes this digest from its own model output and compares
    it against the digest reported by each worker to ensure deterministic
    execution.

    Args:
        req_ids: List of request id strings in batch order.
        sampled_token_ids: Parallel list of sampled token-id sequences.

    Returns:
        A hexadecimal SHA-256 digest string.
    """

    req_ids_norm = [str(req_id) for req_id in req_ids]
    token_ids_norm = [[int(token_id) for token_id in row] for row in sampled_token_ids]
    return _hash_payload({"req_ids": req_ids_norm, "sampled_token_ids": token_ids_norm})


__all__ = (
    "CMD_HEALTH",
    "CMD_HELLO",
    "CMD_SHUTDOWN",
    "CMD_STEP",
    "STATUS_ERROR",
    "STATUS_OK",
    "WorkerInfo",
    "compute_sampled_digest",
    "make_config_fingerprint",
)
