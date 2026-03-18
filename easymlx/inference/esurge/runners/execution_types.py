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

"""Execution request/response types for model runners.

These types are the contract between scheduler/core planning and runner
execution. They intentionally keep legacy fields (``prompt``, ``sampling_params``,
``output``) so older call paths remain valid while the new runtime migrates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..sampling_params import SamplingParams


@dataclass(slots=True, frozen=True)
class ScheduledSequence:
    """One request row selected by the scheduler for this step.

    Carries the token IDs to process, the row index in the sequence
    buffer, and cache page ownership information.

    Attributes:
        request_id: Unique request identifier.
        row_index: Row index in the :class:`SequenceBuffer`.
        token_ids: Token IDs to process in this step.
        num_computed_tokens: Tokens already computed for this request.
        page_ids: KV-cache page IDs assigned to this request.
        meta: Arbitrary per-sequence metadata.
    """

    request_id: str
    row_index: int
    token_ids: list["int"]
    num_computed_tokens: int = 0
    page_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def query_len(self) -> int:
        """Return the number of tokens to process in this step.

        Returns:
            Length of ``token_ids``.
        """
        return len(self.token_ids)


@dataclass(slots=True, frozen=True)
class CacheOperation:
    """Cache/page-table operation requested by the scheduler.

    Represents an explicit cache mutation (e.g. allocate, release, copy)
    that the runner should apply before or after a forward step.

    Attributes:
        op: Operation name (e.g. ``"alloc"``, ``"release"``,
            ``"copy"``).
        request_id: Request to which this operation applies.
        row_index: Optional row index for row-level operations.
        page_ids: Page IDs involved in the operation.
        meta: Arbitrary operation metadata.
    """

    op: str
    request_id: str
    row_index: int | None = None
    page_ids: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionRequest:
    """Runner execution input for one scheduling step.

    Bundles all sequences scheduled for a single forward pass together
    with cache operations, per-request sampling params, and optional
    multimodal inputs.

    Attributes:
        step_id: Monotonically increasing step counter.
        mode: Execution mode (``"prefill"``, ``"decode"``, or
            ``"mixed"``).
        sequences: List of :class:`ScheduledSequence` entries.
        cache_ops: Cache operations to apply during this step.
        sampling_by_request: Per-request :class:`SamplingParams`
            overrides.
        page_table: Optional external page-table object.
        multimodal: Optional multimodal input bundle.
        extra: Extra keyword arguments forwarded to the model callable.
        prompt: Legacy single-prompt field for backward compatibility.
        sampling_params: Legacy single sampling params for backward
            compatibility.
    """

    step_id: int = 0
    mode: str = "decode"
    sequences: list[ScheduledSequence] = field(default_factory=list)
    cache_ops: list[CacheOperation] = field(default_factory=list)
    sampling_by_request: dict[str, SamplingParams] = field(default_factory=dict)
    page_table: Any | None = None
    multimodal: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    prompt: str | None = None
    sampling_params: SamplingParams | None = None

    @property
    def request_ids(self) -> list["str"]:
        """Return the request IDs of all scheduled sequences.

        Returns:
            List of request ID strings.
        """
        return [sequence.request_id for sequence in self.sequences]

    @property
    def row_indices(self) -> list["int"]:
        """Return the row indices of all scheduled sequences.

        Returns:
            List of integer row indices.
        """
        return [sequence.row_index for sequence in self.sequences]

    @property
    def is_legacy(self) -> bool:
        """Return whether this request uses the legacy single-prompt path.

        Returns:
            ``True`` if ``prompt`` is set and ``sequences`` is empty.
        """
        return bool(self.prompt is not None and not self.sequences)

    def sampling_for(self, request_id: str) -> SamplingParams | None:
        """Look up the sampling parameters for a given request.

        Checks ``sampling_by_request`` first, then falls back to the
        legacy ``sampling_params`` field.

        Args:
            request_id: The request to look up.

        Returns:
            The :class:`SamplingParams` for the request, or ``None``.
        """
        specific = self.sampling_by_request.get(request_id)
        if specific is not None:
            return specific
        return self.sampling_params


@dataclass(slots=True)
class ExecutionUpdate:
    """Normalized per-request update emitted by a runner step.

    One update is produced for each :class:`ScheduledSequence` in the
    step, carrying the sampled token(s) and optional finish signals.

    Attributes:
        request_id: Unique request identifier.
        row_index: Row index in the sequence buffer.
        sampled_token_ids: Token IDs sampled for this request.
        finish_reason: Finish reason if the request is done.
        num_computed_tokens: Updated computed-token count.
        metrics: Per-request step-level metrics.
        tool_calls: Optional tool calls extracted from the output.
        reasoning_content: Optional reasoning text.
    """

    request_id: str
    row_index: int
    sampled_token_ids: list["int"]
    finish_reason: str | None = None
    num_computed_tokens: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass(slots=True)
class ExecutionResult:
    """Runner execution output, normalized for engine consumption.

    Aggregates all per-request updates from a single forward step
    together with the raw model output and timing information.

    Attributes:
        step_id: Step counter matching the originating request.
        updates: Per-request :class:`ExecutionUpdate` objects.
        output: Raw model forward-pass output.
        logits: Optional NumPy logits array.
        elapsed_seconds: Wall-clock execution time in seconds.
        error: Error message if the step failed.
        metadata: Arbitrary step-level metadata.
    """

    step_id: int = 0
    updates: list[ExecutionUpdate] = field(default_factory=list)
    output: Any = None
    logits: Any | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def req_ids(self) -> list["str"]:
        """Return the request IDs from all updates.

        Returns:
            List of request ID strings.
        """
        return [update.request_id for update in self.updates]

    @property
    def sampled_token_ids(self) -> list[list["int"]]:
        """Return the sampled token IDs from all updates.

        Returns:
            List of per-request token ID lists.
        """
        return [list(update.sampled_token_ids) for update in self.updates]

    def as_engine_payload(self) -> dict[str, Any]:
        """Serialize the result as a dictionary for engine consumption.

        Returns:
            A dictionary with keys ``step_id``, ``req_ids``,
            ``sampled_token_ids``, ``elapsed_seconds``, ``error``,
            and ``metadata``.
        """
        return {
            "step_id": self.step_id,
            "req_ids": self.req_ids,
            "sampled_token_ids": self.sampled_token_ids,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


__all__ = ("CacheOperation", "ExecutionRequest", "ExecutionResult", "ExecutionUpdate", "ScheduledSequence")
