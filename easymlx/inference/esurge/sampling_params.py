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

"""Minimal sampling parameters for the easymlx eSurge MVP.

Provides a lightweight :class:`SamplingParams` dataclass that mirrors
the subset of EasyDeL's sampling configuration needed for the MVP
runtime. Unsupported fields are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    """Lightweight sampling parameters mapped to easymlx generation APIs.

    Note: This is a subset of EasyDeL's SamplingParams. Unsupported fields are
    intentionally omitted for the MVP.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.  Values > 1 increase
            randomness; values < 1 sharpen the distribution.
        top_k: Top-k filtering threshold.  ``0`` disables top-k.
        top_p: Nucleus (top-p) probability threshold.  ``1.0`` disables
            nucleus sampling.
        do_sample: Whether to use stochastic sampling.  ``False``
            selects greedy (argmax) decoding.
        presence_penalty: Additive penalty applied once to tokens that
            already appeared in the generated output.
        repetition_penalty: Multiplicative penalty applied to tokens
            that already appeared in the full prompt+output history.
        stop: Optional list of stop strings that terminate generation.
        eos_token_id: End-of-sequence token ID override.
        pad_token_id: Padding token ID override.
        n: Number of completions to generate per request.
        include_stop_str_in_output: Whether to include the matched stop
            string in the output text.
        ignore_stop_strings_in_reasoning: If ``True``, stop strings are
            not checked inside reasoning spans.
        tools: Optional tool definitions for function-calling models.
        tool_choice: Tool selection mode (``"auto"``, ``"none"``, or a
            specific tool dict).
        response_format: Optional structured output format descriptor
            (e.g. JSON schema).
        extra_args: Catch-all dictionary for additional model-specific
            arguments.
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: list["str"] | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    n: int = 1
    include_stop_str_in_output: bool = False
    ignore_stop_strings_in_reasoning: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    extra_args: dict[str, Any] | None = field(default_factory=dict)

    def to_generation_kwargs(self) -> dict[str, Any]:
        """Convert sampling parameters to a keyword-argument dictionary.

        The returned dictionary is suitable for passing directly to
        easymlx generation APIs.

        Returns:
            A dictionary of generation keyword arguments.
        """
        return {
            "max_new_tokens": int(self.max_tokens),
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "do_sample": bool(self.do_sample),
            "presence_penalty": float(self.presence_penalty),
            "repetition_penalty": float(self.repetition_penalty),
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "include_stop_str_in_output": bool(self.include_stop_str_in_output),
            "ignore_stop_strings_in_reasoning": self.ignore_stop_strings_in_reasoning,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "response_format": self.response_format,
        }
