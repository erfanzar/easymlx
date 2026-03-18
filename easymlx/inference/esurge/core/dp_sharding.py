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

"""Data-parallel shard helpers for page ownership decisions.

Provides functions and a dataclass for determining which data-parallel
shard owns a given page, enabling distributed page-table management.
"""

from __future__ import annotations

from dataclasses import dataclass


def pages_per_dp_shard(total_pages: int, data_parallel_size: int) -> int:
    """Compute the number of pages assigned to each data-parallel shard.

    Uses ceiling division to ensure all pages are covered even when the
    total is not evenly divisible by the world size.

    Args:
        total_pages: Total number of pages across all shards.
        data_parallel_size: Number of data-parallel shards (world size).

    Returns:
        Pages per shard (ceiling division), or ``0`` when *total_pages*
        is zero.

    Example:
        >>> pages_per_dp_shard(10, 3)
        4
    """
    pages = max(int(total_pages), 0)
    world = max(int(data_parallel_size), 1)
    if pages == 0:
        return 0
    return (pages + world - 1) // world


def dp_shard_for_page_id(page_id: int, *, total_pages: int, data_parallel_size: int) -> int:
    """Determine which data-parallel shard owns a given page.

    Args:
        page_id: The page index to look up.
        total_pages: Total number of pages across all shards.
        data_parallel_size: Number of data-parallel shards (world size).

    Returns:
        The zero-based shard index that owns *page_id*, clamped to
        ``[0, data_parallel_size - 1]``.

    Example:
        >>> dp_shard_for_page_id(7, total_pages=10, data_parallel_size=3)
        1
    """
    world = max(int(data_parallel_size), 1)
    if world == 1:
        return 0
    span = max(pages_per_dp_shard(total_pages, world), 1)
    shard = int(page_id) // span
    return max(min(shard, world - 1), 0)


@dataclass(slots=True)
class DataParallelSharding:
    """Description of one shard in a data-parallel topology.

    Encapsulates the world size, local rank, and total page count so that
    each shard can compute its owned page range.

    Attributes:
        world_size: Total number of data-parallel shards.
        rank: Zero-based rank of this shard.
        total_pages: Total number of pages across all shards.
    """

    world_size: int = 1
    rank: int = 0
    total_pages: int = 0

    @property
    def enabled(self) -> bool:
        """Return whether data-parallel sharding is active (world_size > 1)."""
        return self.world_size > 1

    @property
    def pages_per_shard(self) -> int:
        """Return the number of pages assigned to each shard."""
        return pages_per_dp_shard(self.total_pages, self.world_size)

    def local_page_range(self) -> tuple[int, int]:
        """Return the ``[start, end)`` page range owned by this shard.

        Returns:
            A tuple ``(start, end)`` where *start* is inclusive and *end*
            is exclusive. Returns ``(0, 0)`` when there are no pages.
        """
        span = self.pages_per_shard
        if span <= 0:
            return (0, 0)
        start = max(int(self.rank), 0) * span
        end = min(start + span, max(self.total_pages, 0))
        return start, end
