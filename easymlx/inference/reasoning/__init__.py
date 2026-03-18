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

"""Reasoning parser registry for easymlx.

This package provides a pluggable system for extracting reasoning/thinking
content from Large Language Model outputs. It supports multiple model-specific
formats (``<think>``/``</think>``, ``[THINK]``/``[/THINK]``, channel tags,
text delimiters, etc.) and includes auto-detection based on model type,
chat template, or tokenizer vocabulary.

Key components:
    - ``ReasoningParser``: Abstract base class for all reasoning parsers.
    - ``ReasoningParserManager``: Registry for parser implementations.
    - ``detect_reasoning_parser``: Auto-detection from model configuration.
    - ``BaseThinkingReasoningParser``: Reusable base for token-delimited formats.
    - Model-specific parsers for DeepSeek, Qwen3, Mistral, Granite, etc.
"""

from __future__ import annotations

from .abstract_reasoning import ReasoningParser, ReasoningParserManager
from .auto_detect import detect_reasoning_parser, get_reasoning_tags, make_reasoning_stripper
from .basic_parsers import BaseThinkingReasoningParser
from .parsers import (
    DeepSeekR1ReasoningParser,
    DeepSeekV3ReasoningParser,
    Ernie45ReasoningParser,
    GptOssReasoningParser,
    GraniteReasoningParser,
    HunyuanA13BReasoningParser,
    IdentityReasoningParser,
    MiniMaxM2AppendThinkReasoningParser,
    MiniMaxM2ReasoningParser,
    MistralReasoningParser,
    Olmo3ReasoningParser,
    Qwen3ReasoningParser,
    SeedOSSReasoningParser,
    Step3p5ReasoningParser,
    Step3ReasoningParser,
)

__all__ = (
    "BaseThinkingReasoningParser",
    "DeepSeekR1ReasoningParser",
    "DeepSeekV3ReasoningParser",
    "Ernie45ReasoningParser",
    "GptOssReasoningParser",
    "GraniteReasoningParser",
    "HunyuanA13BReasoningParser",
    "IdentityReasoningParser",
    "MiniMaxM2AppendThinkReasoningParser",
    "MiniMaxM2ReasoningParser",
    "MistralReasoningParser",
    "Olmo3ReasoningParser",
    "Qwen3ReasoningParser",
    "ReasoningParser",
    "ReasoningParserManager",
    "SeedOSSReasoningParser",
    "Step3ReasoningParser",
    "Step3p5ReasoningParser",
    "detect_reasoning_parser",
    "get_reasoning_tags",
    "make_reasoning_stripper",
)
