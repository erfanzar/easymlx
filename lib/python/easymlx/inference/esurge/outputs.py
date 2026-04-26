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

"""Output types for the easymlx eSurge MVP.

Defines the structured output objects that the inference engine emits for
each completion request. :class:`CompletionOutput` represents a single
generated sequence, while :class:`RequestOutput` wraps one or more
completions together with prompt metadata, streaming deltas, and
performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionOutput:
    """Output of a single completion sequence.

    Attributes:
        index: Zero-based index of this completion within the request's
            ``n`` completions.
        text: The generated text.
        token_ids: List of generated token IDs.
        cumulative_logprob: Sum of log-probabilities for all generated
            tokens, or ``None`` if log-probs were not requested.
        logprobs: Per-token log-probability dictionaries mapping token
            IDs to their log-probabilities, or ``None``.
        finish_reason: Reason generation stopped (e.g. ``"eos"``,
            ``"length"``, ``"stop"``), or ``None`` if still in progress.
        tool_calls: Optional list of tool-call dicts extracted from the
            generated text.
        reasoning_content: Optional chain-of-thought reasoning text
            produced alongside the main completion.
    """

    index: int
    text: str
    token_ids: list["int"]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass
class RequestOutput:
    """Output of a request with basic metrics.

    Combines one or more :class:`CompletionOutput` objects with prompt
    metadata, accumulated / delta streaming state, and per-request
    performance counters.

    Attributes:
        request_id: Unique identifier for the originating request.
        prompt: The original prompt text (or list of texts for multi-
            prompt requests).
        prompt_token_ids: Token IDs of the prompt(s).
        outputs: List of :class:`CompletionOutput` objects.
        finished: Whether the request has completed generation.
        metrics: Optional engine-level metric dictionary.
        accumulated_text: Full generated text accumulated so far.
        delta_text: Text generated since the last output event.
        reasoning_content: Accumulated reasoning text, if applicable.
        delta_reasoning_content: Reasoning text since the last event.
        tool_calls: Accumulated tool-call dicts, if applicable.
        delta_tool_calls: Tool-call dicts since the last event.
        tokens_per_second: Average generation throughput.
        num_generated_tokens: Total tokens generated so far.
        time_spent_generating: Wall-clock generation time in seconds.
        first_token_time: Time-to-first-token in seconds, or ``None``.
        processing_time: Total processing time in seconds.
        update_seq: Monotonically increasing sequence number for
            accumulated updates.
        delta_seq: Sequence number for the latest delta.
    """

    request_id: str
    prompt: str | list["str"]
    prompt_token_ids: list["int"] | list[list["int"]]
    outputs: list[CompletionOutput]
    finished: bool = False
    metrics: dict[str, Any] | None = None

    accumulated_text: str = ""
    delta_text: str = ""
    reasoning_content: str | None = None
    delta_reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    delta_tool_calls: list[dict[str, Any]] | None = None
    tokens_per_second: float = 0.0
    num_generated_tokens: int = 0
    time_spent_generating: float = 0.0
    first_token_time: float | None = None
    processing_time: float = 0.0

    update_seq: int = 0
    delta_seq: int = 0

    def get_text(self) -> str:
        """Return the generated text of the first completion output.

        Returns:
            The text from ``outputs[0]``, or an empty string if no
            outputs are present.
        """
        return self.outputs[0].text if self.outputs else ""
