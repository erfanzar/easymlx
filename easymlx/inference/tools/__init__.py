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

"""Tool/function parser registry for EasyMLX inference.

This package provides the central registry and all built-in tool call
parsers used to extract function/tool invocations from LLM outputs.
Each parser handles a model-specific output format (JSON, XML, pythonic,
custom delimiters, etc.) and supports both batch and streaming extraction.

The main entry points are:

- :class:`ToolParser` -- abstract base class for all parsers.
- :class:`ToolParserManager` -- registry for looking up parsers by name.
- :func:`detect_tool_parser` -- auto-detect the correct parser from
  model type, tokenizer vocabulary, or chat template.

Example:
    >>> from easymlx.inference.tools import ToolParserManager, detect_tool_parser
    >>> name = detect_tool_parser(model_type="qwen3")
    >>> parser_cls = ToolParserManager.get_tool_parser(name)
    >>> parser = parser_cls(tokenizer)
    >>> result = parser.extract_tool_calls(model_output, request)
"""

from __future__ import annotations

from .abstract_tool import ToolParser, ToolParserManager
from .auto_detect import detect_tool_parser
from .parsers import (
    DeepSeekV3ToolParser,
    DeepSeekV31ToolParser,
    DeepSeekV32ToolParser,
    Ernie45ToolParser,
    FunctionGemmaToolParser,
    GigaChat3ToolParser,
    Glm4MoeModelToolParser,
    Glm47MoeModelToolParser,
    Granite20bFCToolParser,
    GraniteToolParser,
    HermesToolParser,
    HunyuanA13BToolParser,
    Internlm2ToolParser,
    JambaToolParser,
    KimiK2ToolParser,
    Llama3JsonToolParser,
    Llama4PythonicToolParser,
    LongcatFlashToolParser,
    MinimaxM2ToolParser,
    MinimaxToolParser,
    MistralToolParser,
    Olmo3PythonicToolParser,
    OpenAIToolParser,
    Phi4MiniJsonToolParser,
    PythonicToolParser,
    Qwen3CoderToolParser,
    Qwen3XMLToolParser,
    SeedOssToolParser,
    Step3p5ToolParser,
    Step3ToolParser,
    xLAMToolParser,
)

__all__ = (
    "DeepSeekV3ToolParser",
    "DeepSeekV31ToolParser",
    "DeepSeekV32ToolParser",
    "Ernie45ToolParser",
    "FunctionGemmaToolParser",
    "GigaChat3ToolParser",
    "Glm4MoeModelToolParser",
    "Glm47MoeModelToolParser",
    "Granite20bFCToolParser",
    "GraniteToolParser",
    "HermesToolParser",
    "HunyuanA13BToolParser",
    "Internlm2ToolParser",
    "JambaToolParser",
    "KimiK2ToolParser",
    "Llama3JsonToolParser",
    "Llama4PythonicToolParser",
    "LongcatFlashToolParser",
    "MinimaxM2ToolParser",
    "MinimaxToolParser",
    "MistralToolParser",
    "Olmo3PythonicToolParser",
    "OpenAIToolParser",
    "Phi4MiniJsonToolParser",
    "PythonicToolParser",
    "Qwen3CoderToolParser",
    "Qwen3XMLToolParser",
    "SeedOssToolParser",
    "Step3ToolParser",
    "Step3p5ToolParser",
    "ToolParser",
    "ToolParserManager",
    "detect_tool_parser",
    "xLAMToolParser",
)
