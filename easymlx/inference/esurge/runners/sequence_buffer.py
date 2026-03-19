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

"""Row-oriented sequence buffer for runner execution.

The buffer tracks request rows, token state, and page ids in a mutable, compact
structure that runner/scheduler code can update without touching engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SequenceRow:
    """Mutable row state for one active request.

    Attributes:
        request_id: Unique request identifier.
        prompt_token_ids: Token IDs of the prompt.
        output_token_ids: Tokens generated so far.
        num_computed_tokens: Number of tokens whose KV entries have been
            computed.
        page_ids: KV-cache page indices assigned to this row.
        metadata: Arbitrary per-request metadata.
    """

    request_id: str
    prompt_token_ids: list["int"] = field(default_factory=list)
    output_token_ids: list["int"] = field(default_factory=list)
    num_computed_tokens: int = 0
    page_ids: list["int"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_token_ids(self) -> list["int"]:
        """Return the full token sequence (prompt + output).

        Returns:
            Concatenation of ``prompt_token_ids`` and
            ``output_token_ids``.
        """
        return list(self.prompt_token_ids) + list(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Return the total number of tokens in this row.

        Returns:
            Sum of prompt and output token counts.
        """
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class SequenceBuffer:
    """Sparse row buffer with stable request-id indexing and hole compaction.

    Rows may contain ``None`` (holes) left behind by removed sequences.
    The buffer supports explicit compaction to close gaps and trim
    trailing holes, which keeps the active row indices dense and
    contiguous for efficient KV-cache addressing.

    Attributes:
        max_num_rows: Optional hard cap on the number of rows.
    """

    def __init__(self, *, max_num_rows: int | None = None):
        """Initialize the sequence buffer.

        Args:
            max_num_rows: Maximum number of rows.  ``None`` means
                unlimited.
        """
        self.max_num_rows = None if max_num_rows is None else max(int(max_num_rows), 0)
        self._rows: list[SequenceRow | None] = []
        self._req_id_to_index: dict[str, int] = {}
        self._mutation_counter: int = 0
        self._cached_page_table: np.ndarray | None = None
        self._page_table_version: int = -1

    def __len__(self) -> int:
        """Return the number of active (non-hole) rows.

        Returns:
            Count of rows with an active request.
        """
        return len(self._req_id_to_index)

    @property
    def row_capacity(self) -> int:
        """Return the total allocated row capacity (including holes).

        Returns:
            Length of the internal row list.
        """
        return len(self._rows)

    @property
    def req_ids(self) -> list[str | None]:
        """Return a list of request IDs (or ``None`` for holes) by row index.

        Returns:
            List matching the internal row layout.
        """
        return [row.request_id if row is not None else None for row in self._rows]

    @property
    def req_id_to_index(self) -> dict[str, int]:
        """Return a snapshot of the request-ID-to-row-index mapping.

        Returns:
            Shallow copy of the internal mapping.
        """
        return dict(self._req_id_to_index)

    @property
    def active_row_indices(self) -> list["int"]:
        """Return the indices of all non-hole rows.

        Returns:
            Sorted list of active row indices.
        """
        return [index for index, row in enumerate(self._rows) if row is not None]

    def clear(self) -> None:
        """Remove all rows and reset the request-ID mapping."""
        self._rows.clear()
        self._req_id_to_index.clear()

    def has_request(self, request_id: str) -> bool:
        """Check whether a request is tracked in the buffer.

        Args:
            request_id: The request to look up.

        Returns:
            ``True`` if the request exists.
        """
        return request_id in self._req_id_to_index

    def get_row_index(self, request_id: str) -> int:
        """Return the row index for a tracked request.

        Args:
            request_id: The request to look up.

        Returns:
            The integer row index.

        Raises:
            KeyError: If the request is not tracked.
        """
        return self._req_id_to_index[request_id]

    def get_row(self, request_id: str) -> SequenceRow:
        """Return the :class:`SequenceRow` for a tracked request.

        Args:
            request_id: The request to look up.

        Returns:
            The mutable :class:`SequenceRow`.

        Raises:
            KeyError: If the request is not tracked or the row is a
                hole.
        """
        row = self._rows[self.get_row_index(request_id)]
        if row is None:
            raise KeyError("request_id")
        return row

    def get_row_by_index(self, row_index: int) -> SequenceRow:
        """Return the :class:`SequenceRow` at a given row index.

        Args:
            row_index: The row index to query.

        Returns:
            The mutable :class:`SequenceRow`.

        Raises:
            ValueError: If *row_index* is out of range.
            KeyError: If the row is a hole.
        """
        self._validate_row_index(row_index)
        row = self._rows[row_index]
        if row is None:
            raise KeyError(f"Row {row_index} is empty")
        return row

    def _validate_row_index(self, row_index: int) -> None:
        """Validate that *row_index* is within bounds.

        Args:
            row_index: The index to validate.

        Raises:
            ValueError: If the index is negative or out of range.
        """
        if row_index < 0:
            raise ValueError(f"row_index must be >= 0, got {row_index}")
        if row_index >= len(self._rows):
            raise ValueError(f"row_index {row_index} is out of range for buffer size {len(self._rows)}")

    def _first_free_row(self) -> int | None:
        """Return the index of the first hole, or ``None`` if dense.

        Returns:
            The index of the first ``None`` row, or ``None``.
        """
        for index, row in enumerate(self._rows):
            if row is None:
                return index
        return None

    def _ensure_capacity_for_row(self, row_index: int) -> None:
        """Grow the internal list so *row_index* is valid.

        Args:
            row_index: The row index that must be reachable.

        Raises:
            RuntimeError: If *row_index* would exceed ``max_num_rows``.
        """
        if self.max_num_rows is not None and row_index >= self.max_num_rows:
            raise RuntimeError(f"Cannot allocate row {row_index}; max_num_rows={self.max_num_rows} would be exceeded")
        while row_index >= len(self._rows):
            self._rows.append(None)

    def begin_sequence(
        self,
        request_id: str,
        prompt_token_ids: list["int"] | tuple[int, ...] = (),
        *,
        row_index: int | None = None,
        page_ids: list["int"] | tuple[int, ...] = (),
        num_computed_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert or replace a request row and return its row index.

        If *request_id* already exists it is removed first.  If
        *row_index* is ``None``, the first free row (or a new row) is
        used.

        Args:
            request_id: Unique request identifier.
            prompt_token_ids: Token IDs of the prompt.
            row_index: Explicit target row, or ``None`` for automatic.
            page_ids: KV-cache page indices.
            num_computed_tokens: Tokens already computed.
            metadata: Arbitrary metadata dict.

        Returns:
            The row index where the sequence was placed.

        Raises:
            ValueError: If *request_id* is empty.
            RuntimeError: If capacity would be exceeded.
        """

        if not request_id:
            raise ValueError("request_id is required")
        if request_id in self._req_id_to_index:
            old_index = self._req_id_to_index[request_id]
            self._rows[old_index] = None
            self._req_id_to_index.pop(request_id, None)

        target_index = row_index
        if target_index is None:
            target_index = self._first_free_row()
            if target_index is None:
                target_index = len(self._rows)
        self._ensure_capacity_for_row(target_index)

        existing = self._rows[target_index]
        if existing is not None:
            self._req_id_to_index.pop(existing.request_id, None)

        row = SequenceRow(
            request_id=request_id,
            prompt_token_ids=[int(token) for token in prompt_token_ids],
            output_token_ids=[],
            num_computed_tokens=max(int(num_computed_tokens), 0),
            page_ids=[int(page_id) for page_id in page_ids],
            metadata=dict(metadata or {}),
        )
        self._rows[target_index] = row
        self._req_id_to_index[request_id] = target_index
        return target_index

    def remove_sequence(self, request_id: str) -> SequenceRow:
        """Remove a sequence and leave a hole at its row index.

        Args:
            request_id: The request to remove.

        Returns:
            The removed :class:`SequenceRow`.

        Raises:
            KeyError: If the request is not tracked.
        """
        row_index = self._req_id_to_index.pop(request_id)
        row = self._rows[row_index]
        if row is None:
            raise KeyError(request_id)
        self._rows[row_index] = None
        return row

    def append_output_tokens(self, request_id: str, token_ids: list["int"] | tuple[int, ...]) -> None:
        """Append generated tokens to a sequence's output.

        Args:
            request_id: The request to update.
            token_ids: Token IDs to append.

        Raises:
            KeyError: If the request is not tracked.
        """
        row = self.get_row(request_id)
        row.output_token_ids.extend(int(token_id) for token_id in token_ids)

    def set_num_computed_tokens(self, request_id: str, value: int) -> None:
        """Set the computed-token count for a sequence.

        Args:
            request_id: The request to update.
            value: New computed-token count (clamped to >= 0).

        Raises:
            KeyError: If the request is not tracked.
        """
        row = self.get_row(request_id)
        row.num_computed_tokens = max(int(value), 0)

    def set_page_ids(self, request_id: str, page_ids: list["int"] | tuple[int, ...]) -> None:
        """Replace the page IDs for a sequence.

        Args:
            request_id: The request to update.
            page_ids: New page ID list.

        Raises:
            KeyError: If the request is not tracked.
        """
        row = self.get_row(request_id)
        row.page_ids = [int(page_id) for page_id in page_ids]
        self._mutation_counter += 1

    def swap_rows(self, first_index: int, second_index: int) -> None:
        """Swap two rows in-place, updating the request-ID mapping.

        Args:
            first_index: First row index.
            second_index: Second row index.

        Raises:
            ValueError: If either index is out of range.
        """
        self._validate_row_index(first_index)
        self._validate_row_index(second_index)
        if first_index == second_index:
            return
        self._rows[first_index], self._rows[second_index] = self._rows[second_index], self._rows[first_index]
        first_row = self._rows[first_index]
        second_row = self._rows[second_index]
        if first_row is not None:
            self._req_id_to_index[first_row.request_id] = first_index
        if second_row is not None:
            self._req_id_to_index[second_row.request_id] = second_index

    def move_row(self, from_index: int, to_index: int) -> None:
        """Move a row from one index to another, swapping if necessary.

        Args:
            from_index: Source row index.
            to_index: Destination row index.

        Raises:
            ValueError: If *from_index* is out of range or *to_index*
                is negative.
            RuntimeError: If *to_index* would exceed capacity.
        """
        self._validate_row_index(from_index)
        if to_index < 0:
            raise ValueError(f"to_index must be >= 0, got {to_index}")
        self._ensure_capacity_for_row(to_index)
        if from_index == to_index:
            return
        source = self._rows[from_index]
        destination = self._rows[to_index]
        self._rows[to_index] = source
        self._rows[from_index] = destination
        if source is not None:
            self._req_id_to_index[source.request_id] = to_index
        if destination is not None:
            self._req_id_to_index[destination.request_id] = from_index

    def compact_holes(self, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
        """Shift rows left in ``[start, end)`` to remove interior holes.

        Args:
            start: Start index (inclusive) of the compaction range.
            end: End index (exclusive).  Defaults to the buffer length.

        Returns:
            List of ``(from_index, to_index)`` moves performed.

        Raises:
            ValueError: If *start* is negative or *end < start*.
        """

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end is None:
            end = len(self._rows)
        if end < start:
            raise ValueError(f"end must be >= start (start={start}, end={end})")
        end = min(end, len(self._rows))

        moved: list[tuple[int, int]] = []
        write_index = start
        for read_index in range(start, end):
            row = self._rows[read_index]
            if row is None:
                continue
            if write_index != read_index:
                self._rows[write_index] = row
                self._rows[read_index] = None
                self._req_id_to_index[row.request_id] = write_index
                moved.append((read_index, write_index))
            write_index += 1
        return moved

    def trim_trailing_holes(self) -> None:
        """Remove trailing ``None`` rows from the buffer."""
        while self._rows and self._rows[-1] is None:
            self._rows.pop()

    def page_table(self, *, pad_value: int = -1, max_pages_per_row: int | None = None) -> np.ndarray:
        """Build a dense page-table matrix from row page ids.

        Caches the result until the buffer is modified (rows added/removed
        or page_ids changed).

        Args:
            pad_value: Value used for unused page slots.
            max_pages_per_row: Column width of the output.  Defaults to
                the maximum page count across all active rows.

        Returns:
            A 2-D ``int32`` NumPy array of shape
            ``(row_capacity, max_pages_per_row)``.
        """
        version = getattr(self, "_page_table_version", -1)
        current = getattr(self, "_mutation_counter", 0)
        cached = getattr(self, "_cached_page_table", None)
        if cached is not None and version == current and max_pages_per_row is None:
            return cached

        if max_pages_per_row is None:
            max_pages_per_row = max((len(row.page_ids) for row in self._rows if row is not None), default=0)
        max_pages_per_row = max(int(max_pages_per_row), 0)
        table = np.full((len(self._rows), max_pages_per_row), int(pad_value), dtype=np.int32)
        if max_pages_per_row > 0:
            for row_index, row in enumerate(self._rows):
                if row is None or not row.page_ids:
                    continue
                page_ids = row.page_ids[:max_pages_per_row]
                table[row_index, : len(page_ids)] = np.asarray(page_ids, dtype=np.int32)
        self._cached_page_table = table
        self._page_table_version = current
        return table

    def snapshot(self) -> list[SequenceRow | None]:
        """Return a deep copy of the current row state.

        Returns:
            List of :class:`SequenceRow` copies (or ``None`` for holes).
        """
        return [
            (
                None
                if row is None
                else SequenceRow(
                    request_id=row.request_id,
                    prompt_token_ids=list(row.prompt_token_ids),
                    output_token_ids=list(row.output_token_ids),
                    num_computed_tokens=row.num_computed_tokens,
                    page_ids=list(row.page_ids),
                    metadata=dict(row.metadata),
                )
            )
            for row in self._rows
        ]


__all__ = ("SequenceBuffer", "SequenceRow")
