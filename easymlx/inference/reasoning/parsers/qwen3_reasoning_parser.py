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

"""Reasoning parser for Qwen3 models.

Format: <think>reasoning content</think>response

Qwen3 is strict about requiring both tags unless prompt context indicates
that the start tag was already injected by the chat template.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...openai_api_modules import DeltaMessage
from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["qwen3", "qwen3_reasoning"])
class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3 models with strict tag enforcement.

    Qwen3 models use ``<think>...</think>`` tags but enforce stricter rules
    than most reasoning parsers:

    - If neither the start nor the end tag is present, the output is
      treated as plain content (no reasoning).
    - If the start tag is present but the end tag is missing, the
      output may still be treated as reasoning if the prompt context
      indicates reasoning was already active.
    - If only the end tag is present, the output is treated as content
      unless prompt-gated asymmetric mode is active.

    Attributes:
        start_token (str): The ``<think>`` tag marking reasoning start.
        end_token (str): The ``</think>`` tag marking reasoning end.

    Example:
        >>> parser = Qwen3ReasoningParser()
        >>> reasoning, content = parser.extract_reasoning(
        ...     "<think>analysis</think>conclusion"
        ... )
        >>> reasoning
        'analysis'
        >>> content
        'conclusion'
    """

    start_token = "<think>"
    end_token = "</think>"

    def extract_reasoning(self, model_output: str, request=None) -> tuple[str | None, str | None]:
        """Extract reasoning with strict tag requirements.

        Both ``<think>`` and ``</think>`` tags are generally required.
        The output is treated as content if:

        - No end token is found and no start token or prompt-gated
          reasoning context is present.
        - The start token is missing and prompt-gated mode is not active.

        Args:
            model_output (str): The raw model output string to parse.
            request: Optional request context (unused directly, but
                prompt-gating state may be checked internally).

        Returns:
            tuple[str | None, str | None]: A two-element tuple of
                ``(reasoning, content)``. Returns ``(None, model_output)``
                when strict tag requirements are not met.
        """
        if self.end_token not in model_output:
            if self.start_token in model_output or self._is_prompt_reasoning_active():
                return super().extract_reasoning(model_output, request)
            return None, model_output

        if self.start_token not in model_output:
            if self._is_prompt_reasoning_active():
                return super().extract_reasoning(model_output, request)
            return None, model_output

        return super().extract_reasoning(model_output, request)

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
        """Extract reasoning content incrementally with strict mode.

        Applies Qwen3-specific strict logic during streaming: if no start
        tag has been observed after enough text has been generated, all
        subsequent output is treated as content rather than reasoning.

        Args:
            previous_text (str): Text accumulated before this chunk.
            current_text (str): Full text including the current chunk.
            delta_text (str): New text in the current chunk.
            previous_token_ids (Sequence[int]): Token IDs up to the
                previous chunk.
            current_token_ids (Sequence[int]): All token IDs so far.
            delta_token_ids (Sequence[int]): New token IDs in this chunk.
            request: Optional request context (unused).

        Returns:
            DeltaMessage | None: A delta containing ``reasoning_content``
                or ``content`` depending on strict mode evaluation.
                Returns ``None`` if the delta is empty.
        """
        has_start_in_current = self.start_token in current_text or (
            self._start_token_id is not None and self._start_token_id in current_token_ids
        )

        if (
            not self._is_prompt_reasoning_active()
            and current_text
            and not has_start_in_current
            and len(current_text) > len(self.start_token)
        ):
            return DeltaMessage(content=delta_text) if delta_text else None

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
