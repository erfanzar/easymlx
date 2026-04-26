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

"""Structured sampling metadata used by scheduler/runner components.

This module defines :class:`SamplingMetadata`, an immutable snapshot of the
generation controls (temperature, top-p, top-k, etc.) for a single
request. It is constructed from either a
:class:`~easymlx.inference.esurge.sampling_params.SamplingParams` or an
:class:`~easymlx.inference.esurge.request.EngineRequest`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..request import EngineRequest
from ..sampling_params import SamplingParams


@dataclass(slots=True)
class SamplingMetadata:
    """Snapshot of generation controls for one request.

    Attributes:
        temperature: Sampling temperature; higher values increase
            randomness.
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling limit (0 disables top-k filtering).
        max_tokens: Maximum number of tokens to generate, or ``None`` for
            unlimited.
        do_sample: If ``True``, use stochastic sampling; otherwise use
            greedy decoding.
        stop: Optional list of stop strings that terminate generation.
        eos_token_id: Optional end-of-sequence token id.
    """

    temperature: float
    top_p: float
    top_k: int
    max_tokens: int | None
    do_sample: bool
    stop: list["str"] | None
    eos_token_id: int | None

    @classmethod
    def from_sampling_params(cls, sampling_params: SamplingParams) -> SamplingMetadata:
        """Construct metadata from a :class:`SamplingParams` instance.

        Args:
            sampling_params: The sampling parameters to extract values from.

        Returns:
            A new :class:`SamplingMetadata` populated from the given params.
        """
        return cls(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
            max_tokens=sampling_params.max_tokens,
            do_sample=bool(sampling_params.do_sample),
            stop=list(sampling_params.stop) if sampling_params.stop else None,
            eos_token_id=sampling_params.eos_token_id,
        )

    @classmethod
    def from_request(cls, request: EngineRequest) -> SamplingMetadata:
        """Construct metadata from an :class:`EngineRequest`.

        If the request carries no explicit sampling parameters, sensible
        defaults are used (temperature=1.0, greedy decoding).

        Args:
            request: The engine request to read sampling configuration from.

        Returns:
            A new :class:`SamplingMetadata` for the request.
        """
        if request.sampling_params is None:
            return cls(
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_tokens=None,
                do_sample=False,
                stop=None,
                eos_token_id=request.eos_token_id,
            )
        metadata = cls.from_sampling_params(request.sampling_params)
        if metadata.eos_token_id is None:
            metadata.eos_token_id = request.eos_token_id
        return metadata
