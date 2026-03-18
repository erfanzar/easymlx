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

"""Parser helpers for reasoning/tool extraction.

Provides :func:`extract_reasoning_and_tools`, which applies the configured
reasoning and/or tool parsers to a generated text string and returns a
structured dictionary of extracted components.
"""

from __future__ import annotations

from typing import Any

from easymlx.inference.reasoning import ReasoningParserManager
from easymlx.inference.tools import ToolParserManager


def extract_reasoning_and_tools(
    text: str,
    *,
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
    tokenizer: Any | None = None,
    request: Any | None = None,
) -> dict[str, Any]:
    """Apply the configured reasoning/tool parsers to a generated string.

    The function first applies the reasoning parser (if configured) to
    split the text into reasoning content and visible output. It then
    applies the tool parser (if configured) to extract structured tool
    calls from the visible text.

    Args:
        text: The raw generated text to parse.
        reasoning_parser: Name of the reasoning parser to use, or
            ``None`` to skip reasoning extraction.
        tool_parser: Name of the tool parser to use, or ``None`` to skip
            tool-call extraction.
        tokenizer: Optional tokenizer instance passed to the parser
            constructors.
        request: Optional request context passed to the parsers'
            ``extract_*`` methods.

    Returns:
        A dictionary with keys:
            - ``"text"``: The visible output text after parsing.
            - ``"reasoning_content"``: Extracted reasoning string, or
              ``None``.
            - ``"tool_calls"``: List of tool-call dicts, or ``None``.
    """

    visible_text = text
    reasoning_content = None
    tool_calls = None

    if reasoning_parser is not None:
        parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
        parser = parser_cls(tokenizer)
        reasoning_content, visible_text = parser.extract_reasoning(visible_text, request)

    if tool_parser is not None and visible_text:
        parser_cls = ToolParserManager.get_tool_parser(tool_parser)
        parser = parser_cls(tokenizer)
        result = parser.extract_tool_calls(visible_text, request)
        if result.tools_called:
            tool_calls = [tc.model_dump() for tc in result.tool_calls]
            visible_text = result.content or ""
        else:
            visible_text = result.content or visible_text

    return {
        "text": visible_text,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


__all__ = ("extract_reasoning_and_tools",)
