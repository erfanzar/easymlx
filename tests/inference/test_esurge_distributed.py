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

from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest

from easymlx.inference.esurge.distributed.controller import DistributedController, resolve_distributed_role
from easymlx.inference.esurge.distributed.discovery import DiscoveryResult, discover_workers, resolve_service_hosts
from easymlx.inference.esurge.distributed.leader_client import WorkerRpcClient
from easymlx.inference.esurge.distributed.leader_client import zmq as leader_zmq
from easymlx.inference.esurge.distributed.protocol import STATUS_OK, compute_sampled_digest, make_config_fingerprint
from easymlx.inference.esurge.distributed.worker_server import WorkerControlServer
from easymlx.inference.esurge.distributed.worker_server import zmq as worker_zmq


def test_make_config_fingerprint_is_stable() -> None:
    left = {"a": 1, "b": {"x": 2, "y": [1, 2, 3]}}
    right = {"b": {"y": [1, 2, 3], "x": 2}, "a": 1}
    assert make_config_fingerprint(left) == make_config_fingerprint(right)


def test_compute_sampled_digest_changes_with_output() -> None:
    digest_a = compute_sampled_digest(["r1"], [[1, 2]])
    digest_b = compute_sampled_digest(["r1"], [[1, 3]])
    assert digest_a != digest_b


def test_resolve_service_hosts_sorts_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 0)),
    ]

    def fake_getaddrinfo(*args, **kwargs):  # type: ignore[no-untyped-def]
        return entries

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    result = resolve_service_hosts("svc.local", world_size=2)
    assert result.hosts == ["10.0.0.1", "10.0.0.2"]


def test_discover_workers_maps_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(name: str, world_size: int | None = None) -> DiscoveryResult:
        assert name == "svc.local"
        assert world_size == 2
        return DiscoveryResult(hosts=["host-a", "host-b"])

    monkeypatch.setattr("easymlx.inference.esurge.distributed.discovery.resolve_service_hosts", fake_resolve)
    workers = discover_workers("svc.local", world_size=2, control_port=7000)
    assert [worker.worker_id for worker in workers] == ["worker-0", "worker-1"]
    assert [worker.host for worker in workers] == ["host-a", "host-b"]
    assert [worker.port for worker in workers] == [7000, 7000]


def test_resolve_distributed_role_validation() -> None:
    assert resolve_distributed_role("auto", 0) == "leader"
    assert resolve_distributed_role("auto", 1) == "worker"
    with pytest.raises(ValueError):
        resolve_distributed_role("leader", 1)
    with pytest.raises(ValueError):
        resolve_distributed_role("worker", 0)
    with pytest.raises(ValueError):
        resolve_distributed_role("bad", 0)


@dataclass
class _ModelOutput:
    req_ids: list[str]
    sampled_token_ids: list[list[int]]


def test_controller_leader_start_dispatch_verify_shutdown() -> None:
    expected_digest = compute_sampled_digest(["r1"], [[42]])

    class FakeClient:
        def __init__(self, *, endpoint: str, auth_token: str, connect_timeout_s: float, step_timeout_s: float) -> None:
            self.endpoint = endpoint
            self._inflight: int | None = None
            self.closed = False

        def hello(self) -> dict[str, object]:
            return {"status": STATUS_OK, "rank": 1, "world_size": 2, "config_fingerprint": "fp"}

        def begin_step(self, *, step_id: int, scheduler_output: object) -> None:
            self._inflight = step_id

        def finish_step(self) -> dict[str, object]:
            assert self._inflight is not None
            step_id = self._inflight
            self._inflight = None
            return {
                "status": STATUS_OK,
                "step_id": step_id,
                "num_reqs": 1,
                "sampled_digest": expected_digest,
            }

        def reset_connection(self) -> None:
            self._inflight = None

        def shutdown(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=2,
        service_name="svc.local",
        control_port=7788,
        auth_token="token",
        verify_sampling_digest=True,
        config_fingerprint="fp",
        resolve_hosts_fn=lambda service_name, world_size: DiscoveryResult(hosts=["10.0.0.1", "10.0.0.2"]),
        worker_client_cls=FakeClient,  # type: ignore[arg-type]
    )
    controller.start()
    dispatch = controller.dispatch_step({"batch": 1})
    assert dispatch is not None
    controller.verify_step(dispatch, _ModelOutput(req_ids=["r1"], sampled_token_ids=[[42]]))
    controller.shutdown()


def test_controller_worker_requires_execute_step() -> None:
    controller = DistributedController(
        enabled=True,
        role="worker",
        rank=1,
        world_size=2,
        service_name="svc.local",
        control_port=7788,
        auth_token="token",
        config_fingerprint="fp",
        resolve_hosts_fn=lambda service_name, world_size: DiscoveryResult(hosts=["10.0.0.1", "10.0.0.2"]),
    )
    with pytest.raises(ValueError, match="requires an execute_step callback"):
        controller.start()


def test_controller_worker_start_and_shutdown() -> None:
    class FakeServer:
        last_instance: "FakeServer | None" = None

        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.started = False
            self.stopped = False
            self.endpoint = "tcp://worker:7000"
            FakeServer.last_instance = self

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    controller = DistributedController(
        enabled=True,
        role="worker",
        rank=1,
        world_size=2,
        service_name="svc.local",
        control_port=7788,
        auth_token="token",
        config_fingerprint="fp",
        execute_step=lambda payload: _ModelOutput(req_ids=["r"], sampled_token_ids=[[1]]),
        resolve_hosts_fn=lambda service_name, world_size: DiscoveryResult(hosts=["10.0.0.1", "10.0.0.2"]),
        worker_server_cls=FakeServer,  # type: ignore[arg-type]
    )
    controller.start()
    assert FakeServer.last_instance is not None
    assert FakeServer.last_instance.started is True
    controller.shutdown()
    assert FakeServer.last_instance.stopped is True


def test_controller_handshake_mismatch_raises() -> None:
    class BadClient:
        def __init__(self, *, endpoint: str, auth_token: str, connect_timeout_s: float, step_timeout_s: float) -> None:
            self.endpoint = endpoint

        def hello(self) -> dict[str, object]:
            return {"status": STATUS_OK, "rank": 1, "world_size": 2, "config_fingerprint": "wrong"}

        def close(self) -> None:
            pass

    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=2,
        service_name="svc.local",
        control_port=7788,
        auth_token="token",
        config_fingerprint="expected",
        resolve_hosts_fn=lambda service_name, world_size: DiscoveryResult(hosts=["10.0.0.1", "10.0.0.2"]),
        worker_client_cls=BadClient,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="config mismatch"):
        controller.start()


def test_zmq_guards_when_dependency_missing() -> None:
    if leader_zmq is None:
        with pytest.raises(RuntimeError, match="pyzmq"):
            WorkerRpcClient(
                endpoint="tcp://127.0.0.1:5555",
                auth_token="token",
                connect_timeout_s=1.0,
                step_timeout_s=1.0,
            )
    if worker_zmq is None:
        with pytest.raises(RuntimeError, match="pyzmq"):
            WorkerControlServer(
                bind_host="127.0.0.1",
                port=5555,
                auth_token="token",
                rank=1,
                world_size=2,
                config_fingerprint="fp",
                execute_step=lambda payload: _ModelOutput(req_ids=[], sampled_token_ids=[]),
            )
