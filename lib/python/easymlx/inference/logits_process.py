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

"""Logits post-processing helpers.

This module provides lightweight utility functions for manipulating logits
(unnormalized log-probabilities) during text generation. These are used by
the eSurge inference engine for temperature scaling and probability conversion.
"""

from __future__ import annotations

import math
from typing import Any


def apply_temperature(logits: Any, temperature: float) -> Any:
    """Scale logits by a temperature value.

    Temperature controls the randomness of sampling: lower values make
    the distribution sharper (more deterministic), while higher values
    make it flatter (more random).

    Args:
        logits: Logit values to scale. Can be a scalar, list, numpy array,
            or mx.array.
        temperature: Temperature divisor. Values <= 0 result in no scaling
            (returns logits unchanged).

    Returns:
        The scaled logits (``logits / temperature``). Returns the original
        logits unchanged if temperature <= 0 or if division fails.
    """
    if temperature <= 0:
        return logits
    try:
        return logits / float(temperature)
    except Exception:
        return logits


def softmax(values: list["float"]) -> list["float"]:
    """Compute the softmax of a list of values (numerically stable).

    Uses the max-subtraction trick for numerical stability to prevent
    overflow in the exponential computation.

    Args:
        values: A list of float values (logits) to convert to
            probabilities.

    Returns:
        A list of floats that sum to 1.0, representing the softmax
        probabilities. Returns an empty list if *values* is empty.
    """
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


__all__ = ("apply_temperature", "softmax")
