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

"""Utility helpers for the easymlx eSurge MVP.

This module provides normalization and conversion functions used across the
eSurge inference pipeline. Functions handle prompt normalization, stop-sequence
deduplication, chat-template message/tool canonicalization, and token truncation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def normalize_prompts(prompts: str | Iterable["str"]) -> list["str"]:
    """Normalize a prompt input to a list of strings.

    Args:
        prompts: A single prompt string or an iterable of prompt strings.

    Returns:
        A list of prompt strings. A single string is wrapped in a list;
        iterable elements are cast to ``str``.
    """
    if isinstance(prompts, str):
        return [prompts]
    return [str(prompt) for prompt in prompts]


def coerce_mapping_like(value: Any) -> Any:
    """Attempt to parse a string as JSON; return non-strings unchanged.

    This is useful for normalizing function-call arguments that may arrive
    as either a JSON string or an already-parsed dict/list.

    Args:
        value: The value to coerce. If a string, JSON parsing is attempted.
            Non-string values are returned as-is.

    Returns:
        The parsed JSON object if *value* was a valid JSON string, otherwise
        the original *value* unchanged.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def normalize_stop_sequences(stop: Any) -> list["str"]:
    """Normalize stop sequences to a deduplicated list of non-empty strings.

    Args:
        stop: Stop sequence(s) in any supported format: ``None``, a single
            string, or a list/tuple/set of strings. Non-string elements are
            cast to ``str``.

    Returns:
        A deduplicated list of non-empty stop sequence strings, preserving
        first-seen order.
    """
    if stop is None:
        return []
    if isinstance(stop, str):
        candidates = [stop]
    elif isinstance(stop, (list, tuple, set)):
        candidates = list(stop)
    else:
        candidates = [stop]

    normalized: list["str"] = []
    seen: set["str"] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        value = candidate if isinstance(candidate, str) else str(candidate)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def normalize_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize chat messages for use with HuggingFace chat templates.

    Ensures that ``tool_calls[].function.arguments`` and
    ``function_call.arguments`` fields are proper dictionaries rather
    than JSON strings, which is what most chat templates expect.

    Args:
        messages: List of message dictionaries, potentially containing
            ``tool_calls`` or ``function_call`` fields with string arguments.

    Returns:
        A new list of message dictionaries with normalized argument fields.
        Original messages are not mutated.
    """
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                if not isinstance(raw_call, dict):
                    continue
                call = dict(raw_call)
                function_payload = call.get("function")
                if isinstance(function_payload, dict):
                    function_dict = dict(function_payload)
                    arguments = coerce_mapping_like(function_dict.get("arguments"))
                    if arguments is None:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {"value": str(arguments)}
                    function_dict["arguments"] = arguments
                    call["function"] = function_dict
                normalized_calls.append(call)
            msg["tool_calls"] = normalized_calls

        function_call = msg.get("function_call")
        if isinstance(function_call, dict):
            fc = dict(function_call)
            arguments = coerce_mapping_like(fc.get("arguments"))
            if isinstance(arguments, dict):
                fc["arguments"] = arguments
            msg["function_call"] = fc
        normalized_messages.append(msg)
    return normalized_messages


def normalize_chat_template_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Normalize tool definitions for chat template rendering.

    Extracts the ``function`` payload from OpenAI-format tool dicts,
    validates that each tool has a non-empty name, and ensures the
    ``parameters`` field is a dictionary (parsing JSON strings if needed).

    Args:
        tools: List of tool definition dictionaries in OpenAI format,
            or ``None``/empty list.

    Returns:
        A normalized list of function-payload dictionaries suitable for
        chat template rendering, or ``None`` if no valid tools were found.
    """
    if not tools:
        return None

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        payload = dict(tool.get("function") if isinstance(tool.get("function"), dict) else tool)
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        parameters = payload.get("parameters", {})
        if isinstance(parameters, str):
            parsed = coerce_mapping_like(parameters)
            parameters = parsed if isinstance(parsed, dict) else {}
        elif not isinstance(parameters, dict):
            parameters = {}
        payload["parameters"] = parameters
        normalized.append(payload)

    return normalized or None


def to_structured_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert message content fields to structured ``[{type, text}]`` format.

    Ensures every message's ``content`` is a list of typed content parts,
    converting plain strings, ``None``, single dicts, and mixed lists
    to the uniform ``[{"type": "text", "text": ...}]`` structure.

    Args:
        messages: List of message dictionaries with varying content formats.

    Returns:
        A new list of message dictionaries with uniformly structured
        ``content`` fields. Original messages are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif content is None:
            msg["content"] = []
        elif isinstance(content, dict):
            msg["content"] = [content]
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, str):
                    parts.append({"type": "text", "text": part})
                    continue
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                if part.get("type") in ("text", "input_text", "output_text"):
                    text = part.get("text", part.get("content", ""))
                    parts.append({"type": "text", "text": "" if text is None else str(text)})
                else:
                    parts.append(part)
            msg["content"] = parts
        normalized.append(msg)
    return normalized


def truncate_tokens(token_ids: list["int"], max_length: int, *, mode: str = "left") -> tuple[list["int"], int]:
    """Truncate a token ID sequence to a maximum length.

    Args:
        token_ids: The token ID sequence to truncate.
        max_length: Maximum number of tokens to keep. If <= 0, returns
            an empty list.
        mode: Truncation strategy. One of:
            - ``"left"``: Keep the last ``max_length`` tokens (default).
            - ``"right"``: Keep the first ``max_length`` tokens.
            - ``"middle"``: Keep tokens from both ends, removing the middle.

    Returns:
        A tuple of ``(truncated_ids, num_removed)`` where ``truncated_ids``
        is the resulting token list and ``num_removed`` is the number of
        tokens that were discarded.

    Raises:
        ValueError: If *mode* is not one of ``"left"``, ``"right"``,
            or ``"middle"``.
    """
    if max_length <= 0:
        return [], len(token_ids)
    if len(token_ids) <= max_length:
        return token_ids, 0

    if mode == "left":
        return token_ids[-max_length:], len(token_ids) - max_length
    if mode == "right":
        return token_ids[:max_length], len(token_ids) - max_length
    if mode == "middle":
        keep_left = max_length // 2
        keep_right = max_length - keep_left
        return token_ids[:keep_left] + token_ids[-keep_right:], len(token_ids) - max_length
    raise ValueError(f"Unsupported truncate_mode={mode!r}")
