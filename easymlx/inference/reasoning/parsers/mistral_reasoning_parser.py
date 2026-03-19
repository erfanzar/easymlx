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

"""Reasoning parser for Mistral models.

Format: [THINK]reasoning content[/THINK]actual response
Uses special tokens rather than XML tags.
"""

from ..abstract_reasoning import ReasoningParserManager
from ..basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module(["mistral"])
class MistralReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Mistral models.

    Mistral models use ``[THINK]...[/THINK]`` special tokens to delimit
    reasoning content rather than XML-style ``<think>`` tags. All parsing
    logic is inherited from ``BaseThinkingReasoningParser``; only the
    token strings are overridden.

    Attributes:
        start_token (str): The ``[THINK]`` token marking reasoning start.
        end_token (str): The ``[/THINK]`` token marking reasoning end.

    Example:
        >>> parser = MistralReasoningParser()
        >>> reasoning, content = parser.extract_reasoning(
        ...     "[THINK]let me think[/THINK]the answer is 42"
        ... )
        >>> reasoning
        'let me think'
        >>> content
        'the answer is 42'
    """

    start_token = "[THINK]"
    end_token = "[/THINK]"
