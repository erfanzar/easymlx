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

"""Decoder helpers for inference.

Provides utility functions for decoding token ID sequences back into
human-readable text using a tokenizer.
"""

from __future__ import annotations

from typing import Any


def decode_tokens(tokenizer: Any, token_ids: list["int"], *, skip_special_tokens: bool = True) -> str:
    """Decode a list of token IDs into a string using the given tokenizer.

    Args:
        tokenizer: A tokenizer object that implements a ``decode`` method
            (e.g., a HuggingFace ``PreTrainedTokenizer``).
        token_ids: List of integer token IDs to decode.
        skip_special_tokens: If ``True``, special tokens (BOS, EOS, PAD,
            etc.) are omitted from the output string.

    Returns:
        The decoded text string.

    Example:
        >>> decode_tokens(tokenizer, [101, 7592, 102])
        'hello'
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


__all__ = "decode_tokens"
