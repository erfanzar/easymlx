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

"""Input/output helpers for the eSurge layout.

Provides :func:`build_chat_prompt`, which renders a list of chat messages
into a single prompt string using the tokenizer's chat template when
available.
"""

from __future__ import annotations

from typing import Any

from ..utils import normalize_chat_template_messages, normalize_chat_template_tools


def build_chat_prompt(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Construct a chat prompt using the tokenizer template when available.

    If the tokenizer exposes an ``apply_chat_template`` method it is used
    to render the messages with the model's native template. Otherwise a
    simple ``"role: content"`` fallback format is produced.

    Args:
        messages: List of chat message dicts, each containing at least
            ``"role"`` and ``"content"`` keys.
        tokenizer: Optional tokenizer instance with an
            ``apply_chat_template`` method.
        tools: Optional list of tool definition dicts to pass through to
            the chat template.

    Returns:
        A single string containing the rendered prompt.
    """

    normalized_messages = normalize_chat_template_messages(messages)
    normalized_tools = normalize_chat_template_tools(tools)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=normalized_tools,
        )
    return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in normalized_messages)


__all__ = "build_chat_prompt"
