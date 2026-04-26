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

"""Distributed controller for leader/worker lockstep coordination.

Orchestrates the control-plane flow for multi-node eSurge inference. The
leader discovers workers via DNS, performs a handshake to validate
configuration parity, and dispatches scheduler outputs to each worker on
every step. Workers execute the step and return a sampled-output digest
that the leader verifies for deterministic lockstep.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .discovery import DiscoveryResult, resolve_service_hosts
from .leader_client import WorkerRpcClient
from .protocol import STATUS_OK, WorkerInfo, compute_sampled_digest
from .worker_server import WorkerControlServer

logger = logging.getLogger("easymlx.esurge.distributed")


@dataclass(frozen=True)
class StepDispatch:
    """Token returned when a leader dispatches a step to worker ranks.

    Attributes:
        step_id: Monotonically increasing identifier for the dispatched
            step.
    """

    step_id: int


def resolve_distributed_role(role: str, rank: int) -> str:
    """Resolve ``auto``/``leader``/``worker`` role to an explicit string.

    Args:
        role: One of ``"auto"``, ``"leader"``, or ``"worker"``.
        rank: The node's rank in the distributed world.

    Returns:
        ``"leader"`` if the node should act as the leader, ``"worker"``
        otherwise.

    Raises:
        ValueError: If *role* is not one of the three accepted values, or
            if the role/rank combination is invalid (e.g. ``leader`` on
            rank != 0).
    """

    normalized = str(role).strip().lower()
    if normalized not in {"auto", "leader", "worker"}:
        raise ValueError(f"Invalid `distributed_role`: {role!r}")
    if normalized == "auto":
        return "leader" if int(rank) == 0 else "worker"
    if normalized == "leader" and int(rank) != 0:
        raise ValueError("`distributed_role='leader'` requires rank 0.")
    if normalized == "worker" and int(rank) == 0:
        raise ValueError("`distributed_role='worker'` cannot be used with rank 0.")
    return normalized


class DistributedController:
    """Coordinate leader/worker control-plane flow for distributed eSurge.

    On :meth:`start`, the controller resolves the service name to a list
    of hosts, then either starts a worker RPC server (worker role) or
    connects to every remote worker and validates the handshake (leader
    role).

    Args:
        enabled: Whether distributed mode is active.
        role: ``"leader"`` or ``"worker"``.
        rank: This node's rank.
        world_size: Total number of nodes.
        service_name: DNS service name for discovery.
        control_port: TCP port for the control-plane RPC.
        control_bind_host: Interface the worker server binds to.
        advertise_addr: Address advertised to other nodes.
        auth_token: Shared authentication secret.
        step_timeout_s: Timeout in seconds for step responses.
        connect_timeout_s: Timeout in seconds for initial connections.
        verify_sampling_digest: Whether the leader verifies sampled-output
            digests from workers.
        config_fingerprint: SHA-256 fingerprint of the shared config.
        execute_step: Callback for workers to execute a step.
        resolve_hosts_fn: DNS resolution function (overridable for tests).
        worker_client_cls: Class used to create worker RPC clients.
        worker_server_cls: Class used to create the worker RPC server.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        role: str = "leader",
        rank: int = 0,
        world_size: int = 1,
        service_name: str | None = None,
        control_port: int = 0,
        control_bind_host: str = "0.0.0.0",
        advertise_addr: str | None = None,
        auth_token: str = "",
        step_timeout_s: float = 30.0,
        connect_timeout_s: float = 10.0,
        verify_sampling_digest: bool = True,
        config_fingerprint: str = "",
        execute_step: Callable[[Any], Any] | None = None,
        resolve_hosts_fn: Callable[[str, int | None], DiscoveryResult] = resolve_service_hosts,
        worker_client_cls: type[WorkerRpcClient] = WorkerRpcClient,
        worker_server_cls: type[WorkerControlServer] = WorkerControlServer,
    ) -> None:
        self.enabled = bool(enabled)
        self.role = str(role)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.service_name = service_name
        self.control_port = int(control_port)
        self.control_bind_host = str(control_bind_host)
        self.advertise_addr = advertise_addr
        self.auth_token = str(auth_token)
        self.step_timeout_s = float(step_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.verify_sampling_digest = bool(verify_sampling_digest)
        self.config_fingerprint = str(config_fingerprint)
        self._execute_step = execute_step
        self._resolve_hosts_fn = resolve_hosts_fn
        self._worker_client_cls = worker_client_cls
        self._worker_server_cls = worker_server_cls

        self.workers: list[WorkerInfo] = []
        self._started = False
        self._step_counter = 0
        self._discovery: DiscoveryResult | None = None
        self._worker_server: WorkerControlServer | None = None
        self._worker_clients: dict[int, WorkerRpcClient] = {}

    @property
    def is_leader(self) -> bool:
        """Whether this controller is in leader mode.

        Returns:
            ``True`` if distributed mode is enabled and the role is
            ``"leader"``.
        """
        return self.enabled and self.role == "leader"

    @property
    def is_worker(self) -> bool:
        """Whether this controller is in worker mode.

        Returns:
            ``True`` if distributed mode is enabled and the role is
            ``"worker"``.
        """
        return self.enabled and self.role == "worker"

    @property
    def has_remote_workers(self) -> bool:
        """Whether the leader has established connections to remote workers.

        Returns:
            ``True`` if there is at least one connected worker client.
        """
        return bool(self._worker_clients)

    def add_worker(self, worker: WorkerInfo) -> None:
        """Append a worker descriptor to the worker list.

        Args:
            worker: The :class:`WorkerInfo` to add.
        """
        self.workers.append(worker)

    def start(self) -> None:
        """Initialize the distributed control plane.

        For workers, this starts the RPC server. For the leader, this
        connects to every remote worker and validates the handshake
        (config fingerprint, rank, and world size).

        Raises:
            ValueError: If the service name is missing, DNS resolution
                fails, or any worker handshake validation fails.
        """
        if not self.enabled or self._started:
            return

        if self.service_name is None:
            raise ValueError("`distributed_service_name` must be provided when distributed_mode=True")

        self._discovery = self._resolve_hosts_fn(self.service_name, self.world_size)
        hosts = list(self._discovery.hosts)
        if self.rank < 0 or self.rank >= len(hosts):
            raise ValueError(f"Invalid distributed rank {self.rank} for hosts={hosts}")

        if self.advertise_addr is None:
            self.advertise_addr = hosts[self.rank]

        self.workers = [
            WorkerInfo(worker_id=f"worker-{worker_rank}", host=host, port=self.control_port, rank=worker_rank)
            for worker_rank, host in enumerate(hosts)
        ]

        if self.is_worker:
            if self._execute_step is None:
                raise ValueError("Worker distributed role requires an execute_step callback")
            self._worker_server = self._worker_server_cls(
                bind_host=self.control_bind_host,
                port=self.control_port,
                auth_token=self.auth_token,
                rank=self.rank,
                world_size=self.world_size,
                config_fingerprint=self.config_fingerprint,
                execute_step=self._execute_step,
            )
            self._worker_server.start()
            logger.info(
                "Started distributed worker control server rank=%s endpoint=%s",
                self.rank,
                self._worker_server.endpoint,
            )

        if self.is_leader:
            for worker_rank, host in enumerate(hosts):
                if worker_rank == self.rank:
                    continue
                endpoint = f"tcp://{host}:{self.control_port}"
                client = self._worker_client_cls(
                    endpoint=endpoint,
                    auth_token=self.auth_token,
                    connect_timeout_s=self.connect_timeout_s,
                    step_timeout_s=self.step_timeout_s,
                )
                hello = client.hello()

                if hello.get("status") != STATUS_OK:
                    client.close()
                    raise ValueError(
                        f"Distributed worker handshake failed: rank={worker_rank} endpoint={endpoint} response={hello}"
                    )
                worker_fp = str(hello.get("config_fingerprint", ""))
                if worker_fp != self.config_fingerprint:
                    client.close()
                    raise ValueError(
                        "Distributed worker config mismatch: "
                        f"rank={worker_rank} endpoint={endpoint} worker_fp={worker_fp} "
                        f"leader_fp={self.config_fingerprint}"
                    )

                returned_rank = int(hello.get("rank", -1))
                if returned_rank != worker_rank:
                    client.close()
                    raise ValueError(
                        "Distributed worker rank mismatch: "
                        f"expected={worker_rank} got={returned_rank} endpoint={endpoint}"
                    )

                returned_world = int(hello.get("world_size", -1))
                if returned_world != self.world_size:
                    client.close()
                    raise ValueError(
                        "Distributed worker world-size mismatch: "
                        f"expected={self.world_size} got={returned_world} endpoint={endpoint}"
                    )

                self._worker_clients["worker_rank"] = client

        self._started = True

    def dispatch_step(self, scheduler_output: Any) -> StepDispatch | None:
        """Dispatch a scheduler step to all connected workers.

        Args:
            scheduler_output: The scheduler output to broadcast.

        Returns:
            A :class:`StepDispatch` token for later verification, or
            ``None`` if this node is not a leader or has no workers.

        Raises:
            ValueError: If dispatching to any worker fails. Already-
                dispatched workers are cleaned up before re-raising.
        """
        if not self.is_leader or not self._worker_clients:
            return None

        self._step_counter += 1
        step_id = self._step_counter
        dispatched_clients: list[tuple[int, WorkerRpcClient]] = []
        current_client: WorkerRpcClient | None = None
        current_rank = -1
        try:
            for current_rank, client in self._worker_clients.items():
                current_client = client
                client.begin_step(step_id=step_id, scheduler_output=scheduler_output)
                dispatched_clients.append((current_rank, client))
        except Exception as exc:
            for rank, dispatched_client in dispatched_clients:
                try:
                    dispatched_client.finish_step()
                except Exception:
                    pass
                try:
                    dispatched_client.reset_connection()
                except Exception:
                    logger.debug("Failed to reset worker RPC connection rank=%s", rank, exc_info=True)

            if current_client is not None and all(current_client is not c for _, c in dispatched_clients):
                try:
                    current_client.reset_connection()
                except Exception:
                    logger.debug("Failed to reset failed worker RPC connection rank=%s", current_rank, exc_info=True)

            raise ValueError(
                "Distributed step synchronization failure: "
                f"failed to dispatch step_id={step_id} to worker_rank={current_rank}"
            ) from exc

        return StepDispatch(step_id=step_id)

    def verify_step(self, dispatch: StepDispatch | None, model_output: Any) -> None:
        """Verify that all workers produced identical sampled outputs.

        Collects responses from every worker and compares their digest
        against the leader's own computed digest.

        Args:
            dispatch: The dispatch token from :meth:`dispatch_step`, or
                ``None`` to skip verification.
            model_output: The leader's model output, which must have
                ``req_ids`` and ``sampled_token_ids`` attributes.

        Raises:
            ValueError: If any worker reports an error, returns a
                mismatched step id, request count, or sampling digest.
        """
        if dispatch is None or not self._worker_clients:
            return

        req_ids = [str(req_id) for req_id in list(getattr(model_output, "req_ids", []))]
        sampled_token_ids = [
            [int(token) for token in row] for row in list(getattr(model_output, "sampled_token_ids", []))
        ]
        expected_digest = compute_sampled_digest(req_ids, sampled_token_ids)
        expected_num_reqs = len(req_ids)

        for worker_rank, client in self._worker_clients.items():
            try:
                response = client.finish_step()
            except Exception as exc:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} did not respond"
                ) from exc

            if response.get("status") != STATUS_OK:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} error={response.get('error')}"
                )

            worker_step_id = int(response.get("step_id", -1))
            if worker_step_id != dispatch.step_id:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} expected_step_id={dispatch.step_id} got={worker_step_id}"
                )

            worker_num_reqs = int(response.get("num_reqs", -1))
            if worker_num_reqs != expected_num_reqs:
                raise ValueError(
                    "Distributed step synchronization failure: "
                    f"worker_rank={worker_rank} step_id={dispatch.step_id} "
                    f"expected_num_reqs={expected_num_reqs} got={worker_num_reqs}"
                )

            if self.verify_sampling_digest:
                worker_digest = str(response.get("sampled_digest", ""))
                if worker_digest != expected_digest:
                    raise ValueError(
                        "Distributed step synchronization failure: "
                        f"worker_rank={worker_rank} step_id={dispatch.step_id} digest mismatch"
                    )

    def shutdown(self) -> None:
        """Shut down the distributed control plane.

        Stops the worker server (if running) and closes all worker client
        connections.
        """
        if self._worker_server is not None:
            try:
                self._worker_server.stop()
            except Exception:
                logger.debug("Failed to stop worker control server cleanly", exc_info=True)
            self._worker_server = None

        for client in self._worker_clients.values():
            try:
                client.shutdown()
            except Exception:
                logger.debug("Failed to shutdown worker client cleanly", exc_info=True)
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        self._worker_clients.clear()
        self._started = False


__all__ = ("DistributedController", "StepDispatch", "resolve_distributed_role")
