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

"""Identity reasoning parser -- pass-through that treats all output as content.

This parser performs no reasoning extraction. It is used for models that do
not produce separate reasoning content, or as a fallback when no reasoning
format is detected. Registered under the names ``"identity"``, ``"none"``,
and ``"passthrough"``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(["identity", "none", "passthrough"])
class IdentityReasoningParser(ReasoningParser):
    """Pass-through parser: no reasoning extraction, all text is content."""

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Always returns ``True`` since there is no reasoning section.

        Args:
            input_ids: Sequence of token IDs (ignored).

        Returns:
            Always ``True``.
        """
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return all token IDs unchanged (no reasoning tokens to strip).

        Args:
            input_ids: Complete sequence of token IDs.

        Returns:
            A copy of the input token IDs.
        """
        return list(input_ids)

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Return ``(None, full_output)`` -- all text is treated as content.

        Args:
            model_output: Complete text output from the model.
            request: Optional request context (unused).

        Returns:
            Tuple of ``(None, model_output)``.
        """
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request=None,
    ) -> DeltaMessage | None:
        """Return the delta text as content (no reasoning extraction).

        Args:
            previous_text: Text accumulated before this chunk (unused).
            current_text: Text including current chunk (unused).
            delta_text: New text in current chunk.
            previous_token_ids: Token IDs before current chunk (unused).
            current_token_ids: Token IDs including current chunk (unused).
            delta_token_ids: New token IDs in current chunk (unused).
            request: Optional request context (unused).

        Returns:
            DeltaMessage with content set to *delta_text*, or ``None`` if
            *delta_text* is empty.
        """
        return DeltaMessage(content=delta_text) if delta_text else None
