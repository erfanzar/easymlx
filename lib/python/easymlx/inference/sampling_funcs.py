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

"""Small sampling helper functions for token selection.

This module provides lightweight, pure-Python sampling utilities used
during inference to select the next token from a model's output
distribution. Two strategies are offered:

- **Greedy selection**: picks the token with the highest logit.
- **Probabilistic sampling**: samples a token index according to a
  probability distribution.

These helpers are intentionally simple and operate on plain Python lists
rather than framework-specific tensor types.

Example:
    >>> from easymlx.inference.sampling_funcs import greedy_select, sample_from_probs
    >>> logits = [0.1, 0.9, 0.3]
    >>> greedy_select(logits)
    1
    >>> probs = [0.1, 0.7, 0.2]
    >>> token_id = sample_from_probs(probs)  # probabilistic
"""

from __future__ import annotations

import random


def greedy_select(logits: list["float"]) -> int:
    """Select the index with the highest logit value (argmax).

    Args:
        logits (list[float]): A list of logit scores, one per vocabulary
            token. Must be non-empty.

    Returns:
        int: The index of the maximum logit value. If multiple indices
            share the maximum, the first one encountered is returned.

    Example:
        >>> greedy_select([0.1, 0.9, 0.3])
        1
        >>> greedy_select([0.5, 0.5, 0.1])
        0
    """
    return max(range(len(logits)), key=logits.__getitem__)


def sample_from_probs(probabilities: list["float"]) -> int:
    """Sample a token index from a probability distribution.

    Uses weighted random sampling (``random.choices``) to select one
    index according to the provided probability weights. The
    probabilities do not need to sum to 1; they are treated as relative
    weights.

    Args:
        probabilities (list[float]): A list of probability weights, one
            per vocabulary token. Must be non-empty and contain at least
            one positive value.

    Returns:
        int: A sampled index in ``[0, len(probabilities))``.

    Example:
        >>> import random
        >>> random.seed(42)
        >>> sample_from_probs([0.1, 0.7, 0.2])
        1
    """
    return random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


__all__ = ("greedy_select", "sample_from_probs")
