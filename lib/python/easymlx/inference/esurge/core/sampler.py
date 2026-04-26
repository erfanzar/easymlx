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

"""Sampling helpers used by scheduler/runner flows.

Provides lightweight conversion utilities for translating
:class:`~easymlx.inference.esurge.sampling_params.SamplingParams` into
model-runner keyword arguments and for performing greedy (argmax) token
selection from logit arrays.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..sampling_params import SamplingParams


def sampling_params_to_kwargs(sampling_params: SamplingParams | None) -> dict[str, Any]:
    """Convert a :class:`SamplingParams` instance into model-generate kwargs.

    Args:
        sampling_params: The sampling parameters to convert. If ``None``,
            an empty dictionary is returned.

    Returns:
        A dictionary of keyword arguments suitable for passing to a model
        runner's ``generate`` method.
    """

    if sampling_params is None:
        return {}
    return sampling_params.to_generation_kwargs()


def argmax_token(logits: Any) -> int:
    """Select the token id with the highest logit score (greedy decoding).

    Args:
        logits: A rank-1 array of logit values whose length equals
            the vocabulary size.

    Returns:
        The integer index of the maximum element.

    Raises:
        ValueError: If *logits* is not a 1-D array.
    """
    arr = logits if isinstance(logits, mx.array) else mx.array(logits)
    if arr.ndim != 1:
        raise ValueError(f"Expected rank-1 logits, got shape={arr.shape}")
    token_id = mx.argmax(arr, axis=-1).astype(mx.int32)
    mx.eval(token_id)
    return int(token_id.item())
