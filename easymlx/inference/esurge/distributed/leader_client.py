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

"""Leader-side control-plane RPC client for distributed eSurge workers.

Provides :class:`WorkerRpcClient`, a ZMQ REQ-socket wrapper used by the
leader node to send handshake, health-check, step, and shutdown commands
to individual worker nodes.
"""

from __future__ import annotations

from typing import Any

try:
    import zmq
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without pyzmq.
    zmq = None  # type: ignore["assignment"]

from .protocol import CMD_HEALTH, CMD_HELLO, CMD_SHUTDOWN, CMD_STEP

_MISSING_ZMQ_MSG = "Distributed worker RPC requires `pyzmq`. Install `pyzmq` to enable distributed mode."


class WorkerRpcClient:
    """REQ client for one worker control-plane endpoint.

    Wraps a ZMQ REQ socket with typed helpers for each control-plane
    command. Supports an "in-flight step" model where :meth:`begin_step`
    sends the step payload and :meth:`finish_step` blocks until the
    worker responds.

    Args:
        endpoint: ZMQ endpoint string (e.g. ``"tcp://10.0.0.2:5555"``).
        auth_token: Shared secret used to authenticate RPC messages.
        connect_timeout_s: Timeout in seconds for the initial connection
            and send operations.
        step_timeout_s: Timeout in seconds for receiving step results.

    Raises:
        RuntimeError: If ``pyzmq`` is not installed.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth_token: str,
        connect_timeout_s: float,
        step_timeout_s: float,
    ) -> None:
        if zmq is None:
            raise RuntimeError(_MISSING_ZMQ_MSG)

        self.endpoint = str(endpoint)
        self._auth_token = str(auth_token)
        self._step_timeout_ms = max(1, int(float(step_timeout_s) * 1000.0))
        self._connect_timeout_ms = max(1, int(float(connect_timeout_s) * 1000.0))
        self._context = zmq.Context.instance()  # type: ignore[union-attr]
        self._socket = self._create_socket()
        self._inflight_step_id: int | None = None

    def _create_socket(self) -> Any:
        """Create and configure a new ZMQ REQ socket connected to the endpoint.

        Returns:
            A connected ZMQ REQ socket.
        """
        socket = self._context.socket(zmq.REQ)  # type: ignore[union-attr]
        socket.setsockopt(zmq.LINGER, 0)  # type: ignore[union-attr]
        socket.setsockopt(zmq.SNDTIMEO, self._connect_timeout_ms)  # type: ignore[union-attr]
        socket.setsockopt(zmq.RCVTIMEO, self._step_timeout_ms)  # type: ignore[union-attr]
        socket.connect(self.endpoint)
        return socket

    @property
    def has_inflight_step(self) -> bool:
        """Whether there is a step awaiting a response.

        Returns:
            ``True`` if :meth:`begin_step` has been called without a
            matching :meth:`finish_step`.
        """
        return self._inflight_step_id is not None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a request and synchronously receive the response.

        Args:
            payload: Dictionary to send as a pickled Python object.

        Returns:
            The response dictionary from the worker.

        Raises:
            RuntimeError: If the response is not a dictionary.
        """
        self._socket.send_pyobj(payload)
        response = self._socket.recv_pyobj()
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid worker response type from {self.endpoint}: {type(response)!r}")
        return response

    def hello(self) -> dict[str, Any]:
        """Perform the handshake with the worker.

        Returns:
            Worker metadata including ``"status"``, ``"rank"``,
            ``"world_size"``, and ``"config_fingerprint"``.
        """
        return self._request({"cmd": CMD_HELLO, "auth_token": self._auth_token})

    def health(self) -> dict[str, Any]:
        """Query the worker's health status.

        Returns:
            A dictionary with ``"status"``, ``"rank"``,
            ``"steps_executed"``, ``"last_step_id"``, and
            ``"last_error"`` fields.
        """
        return self._request({"cmd": CMD_HEALTH, "auth_token": self._auth_token})

    def begin_step(self, *, step_id: int, scheduler_output: Any) -> None:
        """Send a step command to the worker (non-blocking receive).

        The step payload is sent immediately. The caller must later invoke
        :meth:`finish_step` to collect the response.

        Args:
            step_id: Monotonically increasing step identifier.
            scheduler_output: Scheduler output to be executed by the worker.

        Raises:
            RuntimeError: If a step is already in flight on this client.
        """
        if self._inflight_step_id is not None:
            raise RuntimeError(
                f"Worker {self.endpoint} already has in-flight step {self._inflight_step_id}; "
                f"cannot send step {step_id}."
            )
        self._socket.send_pyobj(
            {
                "cmd": CMD_STEP,
                "auth_token": self._auth_token,
                "step_id": int(step_id),
                "scheduler_output": scheduler_output,
            }
        )
        self._inflight_step_id = int(step_id)

    def finish_step(self) -> dict[str, Any]:
        """Block until the worker responds to the in-flight step.

        Returns:
            The worker's step result dictionary containing ``"status"``,
            ``"step_id"``, ``"sampled_digest"``, ``"num_reqs"``, and
            ``"timing_ms"``.

        Raises:
            RuntimeError: If no step is currently in flight, or if the
                response type is invalid.
        """
        if self._inflight_step_id is None:
            raise RuntimeError(f"No in-flight step for worker {self.endpoint}")
        try:
            response = self._socket.recv_pyobj()
        finally:
            self._inflight_step_id = None
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid worker step response type from {self.endpoint}: {type(response)!r}")
        return response

    def shutdown(self) -> None:
        """Send a shutdown command to the worker.

        Errors are silently swallowed since the worker may have already
        exited.
        """
        try:
            self._request({"cmd": CMD_SHUTDOWN, "auth_token": self._auth_token})
        except Exception:
            pass

    def reset_connection(self) -> None:
        """Close the current socket and reconnect with a fresh one.

        Clears any in-flight step state so the client can be reused after
        a communication failure.
        """
        try:
            self._socket.close(0)
        finally:
            self._socket = self._create_socket()
            self._inflight_step_id = None

    def close(self) -> None:
        """Close the underlying ZMQ socket immediately."""
        self._socket.close(0)


LeaderClient = WorkerRpcClient
"""Backward-compatible alias for :class:`WorkerRpcClient`."""


__all__ = ("LeaderClient", "WorkerRpcClient")
