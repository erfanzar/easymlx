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

"""DNS-based service discovery helpers for distributed eSurge.

Resolves a service name via DNS into a deterministic, sorted list of host
addresses that the distributed controller uses to assign ranks and
establish connections.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass

from .protocol import WorkerInfo


@dataclass(frozen=True)
class DiscoveryResult:
    """Result of distributed service-name resolution.

    Attributes:
        hosts: Deterministically sorted list of unique host addresses
            resolved from the service name.
    """

    hosts: list["str"]

    @property
    def world_size(self) -> int:
        """Return the number of discovered hosts.

        Returns:
            Length of the :attr:`hosts` list.
        """
        return len(self.hosts)

    @property
    def rank_to_host(self) -> dict[int, str]:
        """Map rank indices to their corresponding host addresses.

        Returns:
            A dictionary mapping ``{rank: host}``.
        """
        return {rank: host for rank, host in enumerate(self.hosts)}


def _host_sort_key(host: str) -> tuple[int, str]:
    """Produce a sort key that orders IP addresses before hostnames.

    Args:
        host: A hostname or IP address string.

    Returns:
        A tuple ``(priority, normalized_string)`` where IP addresses get
        priority 0 (sorted numerically) and hostnames get priority 1.
    """
    try:
        ip = ipaddress.ip_address(host)
        return (0, f"{int(ip):039d}")
    except ValueError:
        return (1, "host")


def resolve_service_hosts(service_name: str, world_size: int | None = None) -> DiscoveryResult:
    """Resolve service DNS into a deterministic list of hosts.

    Performs a DNS lookup on *service_name*, deduplicates the returned
    addresses, and sorts them so that every node in the cluster agrees on
    the rank-to-host mapping.

    Args:
        service_name: A DNS name that resolves to the set of worker hosts.
        world_size: If provided, the number of resolved hosts must match
            exactly or a :class:`ValueError` is raised.

    Returns:
        A :class:`DiscoveryResult` with the sorted host list.

    Raises:
        ValueError: If the service name is empty, DNS resolution fails,
            no hosts are found, or the resolved host count does not match
            *world_size*.
    """

    normalized_name = str(service_name or "").strip()
    if not normalized_name:
        raise ValueError("`distributed_service_name` must be a non-empty DNS name.")

    try:
        entries = socket.getaddrinfo(normalized_name, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Failed to resolve distributed service {normalized_name!r}: {exc}") from exc

    hosts: list["str"] = []
    seen: set["str"] = set()
    for entry in entries:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        host = str(sockaddr[0])
        if host in seen:
            continue
        hosts.append(host)
        seen.add(host)

    hosts.sort(key=_host_sort_key)
    if not hosts:
        raise ValueError(f"No hosts resolved for distributed service {normalized_name!r}.")

    if world_size is not None and int(world_size) != len(hosts):
        raise ValueError(
            "Distributed world size mismatch: "
            f"resolved_hosts={len(hosts)} expected_world_size={int(world_size)} "
            f"service={normalized_name!r} hosts={hosts}"
        )
    return DiscoveryResult(hosts=hosts)


def discover_workers(
    service_name: str | None = None,
    *,
    world_size: int | None = None,
    control_port: int = 0,
) -> list[WorkerInfo]:
    """Discover workers from DNS and return typed worker descriptors.

    If *service_name* is omitted, this function checks the
    ``EASYMLX_DISTRIBUTED_SERVICE_NAME`` environment variable. If no service
    is configured, an empty list is returned.

    Args:
        service_name: DNS name of the distributed service. Falls back to
            the ``EASYMLX_DISTRIBUTED_SERVICE_NAME`` environment variable.
        world_size: Expected number of workers. Validated against the
            number of resolved hosts.
        control_port: Port number used for the control-plane RPC.

    Returns:
        A list of :class:`WorkerInfo` instances, one per resolved host,
        or an empty list when no service is configured.
    """

    resolved_name = service_name or os.environ.get("EASYMLX_DISTRIBUTED_SERVICE_NAME")
    if not resolved_name:
        return []
    discovery = resolve_service_hosts(resolved_name, world_size=world_size)
    workers: list[WorkerInfo] = []
    for rank, host in enumerate(discovery.hosts):
        workers.append(
            WorkerInfo(
                worker_id=f"worker-{rank}",
                host=host,
                port=int(control_port),
                rank=rank,
            )
        )
    return workers


__all__ = ("DiscoveryResult", "discover_workers", "resolve_service_hosts")
