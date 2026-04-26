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

from __future__ import annotations

from easymlx.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easymlx.inference.parsing import DelegatingParser
from easymlx.inference.reasoning import ReasoningParserManager, detect_reasoning_parser
from easymlx.inference.tools import ToolParserManager, detect_tool_parser
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _tool_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(
        WordLevel(
            vocab={
                "<unk>": 0,
                "<tool_call>": 1,
                "</tool_call>": 2,
                "<|tool_call>": 3,
                "<tool_call|>": 4,
                "<|channel>": 5,
                "<channel|>": 6,
            },
            unk_token="<unk>",
        )
    )
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


def test_tool_and_reasoning_autodetect_matches_easydel_current_mappings():
    assert detect_tool_parser(model_type="qwen3_5") == "qwen3_coder"
    assert detect_tool_parser(model_type="qwen3_5_text") == "qwen3_coder"
    assert detect_tool_parser(model_type="qwen3_next") == "qwen3_xml"
    assert detect_tool_parser(model_type="gemma4") == "gemma4"
    assert detect_reasoning_parser(model_type="gemma4") == "gemma4"
    assert "gemma4" in ToolParserManager.available_parsers()
    assert "gemma4" in ReasoningParserManager.available_parsers()


def test_delegating_parser_buffers_tool_protocol_and_flushes_final_tool_call():
    tokenizer = _tool_tokenizer()
    tool_parser = ToolParserManager.get_tool_parser("hermes")(tokenizer)
    request = ChatCompletionRequest(
        model="dummy",
        messages=[ChatMessage(role="user", content="hi")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )
    parser = DelegatingParser(tool_parser=tool_parser, tool_request=request)

    partial = parser.process_delta("<tool", "<tool", [], "", [])
    assert partial.delta_content == ""
    assert partial.accumulated_content == ""

    final_text = '<tool_call>{"name":"lookup","arguments":{"city":"paris"}}</tool_call>'
    final = parser.process_final(final_text, [])

    assert final.accumulated_content == ""
    assert final.tool_calls is not None
    assert final.tool_calls[0].function.name == "lookup"
    assert final.delta_tool_calls is not None
