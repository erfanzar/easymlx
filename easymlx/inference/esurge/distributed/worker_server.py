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

"""Worker-side control-plane server for distributed eSurge.

Provides :class:`WorkerControlServer`, a threaded ZMQ REP server that
listens for commands from the leader node (hello, health, step, shutdown)
and dispatches model execution via a user-supplied callback.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

try:
    import zmq
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without pyzmq.
    zmq = None  # type: ignore["assignment"]

from .protocol import CMD_HEALTH, CMD_HELLO, CMD_SHUTDOWN, CMD_STEP, STATUS_ERROR, STATUS_OK, compute_sampled_digest

_MISSING_ZMQ_MSG = "Distributed worker server requires `pyzmq`. Install `pyzmq` to enable distributed mode."


class WorkerControlServer:
    """Threaded REP server used by worker ranks in distributed mode.

    Runs a ZMQ REP socket in a daemon thread, processing incoming
    control-plane commands from the leader.

    Args:
        bind_host: Network interface to bind to (e.g. ``"0.0.0.0"``).
        port: TCP port number for the control socket.
        auth_token: Shared secret that incoming messages must carry.
        rank: This worker's rank in the distributed world.
        world_size: Total number of nodes in the distributed setup.
        config_fingerprint: SHA-256 digest of the shared configuration,
            returned during the hello handshake.
        execute_step: Callback invoked with the scheduler output when the
            leader dispatches a step. Must return a model output object
            containing ``req_ids`` and ``sampled_token_ids``.

    Raises:
        RuntimeError: If ``pyzmq`` is not installed, or if the server
            thread fails to start within 10 seconds.
    """

    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        auth_token: str,
        rank: int,
        world_size: int,
        config_fingerprint: str,
        execute_step: Callable[[Any], Any],
    ) -> None:
        if zmq is None:
            raise RuntimeError(_MISSING_ZMQ_MSG)

        self._bind_host = str(bind_host)
        self._port = int(port)
        self._auth_token = str(auth_token)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._config_fingerprint = str(config_fingerprint)
        self._execute_step = execute_step

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._steps_executed = 0
        self._last_step_id = -1
        self._last_error: str | None = None

    @property
    def endpoint(self) -> str:
        """Return the ZMQ TCP endpoint this server binds to.

        Returns:
            A string of the form ``"tcp://<bind_host>:<port>"``.
        """
        return f"tcp://{self._bind_host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """Check whether the server thread is alive.

        Returns:
            ``True`` if the background thread exists and is still running.
        """
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background server thread.

        Blocks until the ZMQ socket is bound and ready, or raises after a
        10-second timeout.

        Raises:
            RuntimeError: If the server fails to become ready within 10
                seconds.
        """
        if self.is_running:
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._serve_loop, name=f"esurge-worker-rpc-{self._rank}", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise RuntimeError(f"Worker control server failed to start on {self.endpoint}")

    def stop(self) -> None:
        """Signal the server to stop and wait for the thread to join.

        Sends a shutdown command over a temporary socket to unblock the
        receive loop, then joins the thread with a 5-second timeout.
        """
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()

        context = zmq.Context.instance()  # type: ignore[union-attr]
        request_socket = context.socket(zmq.REQ)  # type: ignore[union-attr]
        request_socket.setsockopt(zmq.LINGER, 0)  # type: ignore[union-attr]
        request_socket.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[union-attr]
        request_socket.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[union-attr]
        try:
            request_socket.connect(self.endpoint)
            request_socket.send_pyobj({"cmd": CMD_SHUTDOWN, "auth_token": self._auth_token})
            request_socket.recv_pyobj()
        except Exception:
            pass
        finally:
            request_socket.close(0)

        thread.join(timeout=5.0)
        self._thread = None

    def _is_authorized(self, message: dict[str, Any]) -> bool:
        """Verify that a message carries a valid auth token.

        Args:
            message: The incoming RPC message dictionary.

        Returns:
            ``True`` if the token matches the server's configured token.
        """
        return str(message.get("auth_token", "")) == self._auth_token

    @staticmethod
    def _extract_runner_output(model_output: Any) -> tuple[list["str"], list[list["int"]]]:
        """Normalize model output into (req_ids, sampled_token_ids).

        Handles both dict-style and attribute-style model outputs.

        Args:
            model_output: The object returned by the ``execute_step``
                callback.

        Returns:
            A 2-tuple of ``(req_ids, sampled_token_ids)`` where each
            element is a list of strings or a list of int lists,
            respectively.
        """
        if isinstance(model_output, dict):
            req_ids = list(model_output.get("req_ids", []))
            sampled_token_ids = list(model_output.get("sampled_token_ids", []))
        else:
            req_ids = list(getattr(model_output, "req_ids", []))
            sampled_token_ids = list(getattr(model_output, "sampled_token_ids", []))

        normalized_req_ids = [str(req_id) for req_id in req_ids]
        normalized_sampled = [[int(token) for token in row] for row in sampled_token_ids]
        return normalized_req_ids, normalized_sampled

    def _serve_loop(self) -> None:
        """Main server loop running in the background thread.

        Creates a ZMQ REP socket, binds it, and enters a polling loop
        that dispatches incoming commands until a shutdown signal is
        received.
        """
        context = zmq.Context()  # type: ignore["operator"]
        socket = context.socket(zmq.REP)  # type: ignore[union-attr]
        socket.setsockopt(zmq.LINGER, 0)  # type: ignore[union-attr]
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[union-attr]
        socket.bind(self.endpoint)
        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    message = socket.recv_pyobj()
                except zmq.error.Again:  # type: ignore[union-attr]
                    continue

                if not isinstance(message, dict):
                    socket.send_pyobj({"status": STATUS_ERROR, "error": "Invalid message payload."})
                    continue
                if not self._is_authorized(message):
                    socket.send_pyobj({"status": STATUS_ERROR, "error": "Unauthorized control-plane request."})
                    continue

                cmd = message.get("cmd")
                if cmd == CMD_HELLO:
                    socket.send_pyobj(
                        {
                            "status": STATUS_OK,
                            "rank": self._rank,
                            "world_size": self._world_size,
                            "config_fingerprint": self._config_fingerprint,
                        }
                    )
                    continue

                if cmd == CMD_HEALTH:
                    socket.send_pyobj(
                        {
                            "status": STATUS_OK,
                            "rank": self._rank,
                            "world_size": self._world_size,
                            "steps_executed": self._steps_executed,
                            "last_step_id": self._last_step_id,
                            "last_error": self._last_error,
                        }
                    )
                    continue

                if cmd == CMD_STEP:
                    step_id = int(message.get("step_id", -1))
                    scheduler_output = message.get("scheduler_output")
                    started = time.perf_counter()
                    try:
                        model_output = self._execute_step(scheduler_output)
                        req_ids, sampled_token_ids = self._extract_runner_output(model_output)
                        sampled_digest = compute_sampled_digest(req_ids, sampled_token_ids)
                        self._steps_executed += 1
                        self._last_step_id = step_id
                        self._last_error = None
                        socket.send_pyobj(
                            {
                                "status": STATUS_OK,
                                "step_id": step_id,
                                "sampled_digest": sampled_digest,
                                "num_reqs": len(req_ids),
                                "timing_ms": (time.perf_counter() - started) * 1000.0,
                            }
                        )
                    except Exception as exc:
                        self._last_step_id = step_id
                        self._last_error = str(exc)
                        socket.send_pyobj(
                            {
                                "status": STATUS_ERROR,
                                "step_id": step_id,
                                "timing_ms": (time.perf_counter() - started) * 1000.0,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    continue

                if cmd == CMD_SHUTDOWN:
                    socket.send_pyobj({"status": STATUS_OK})
                    break

                socket.send_pyobj({"status": STATUS_ERROR, "error": f"Unknown command: {cmd}"})
        finally:
            socket.close(0)
            context.term()


WorkerServer = WorkerControlServer
"""Backward-compatible alias for :class:`WorkerControlServer`."""


__all__ = ("WorkerControlServer", "WorkerServer")
