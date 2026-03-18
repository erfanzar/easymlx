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

"""Shared utility helpers for scheduler/core components.

Re-exports common normalisation helpers from the parent ``utils`` module and
adds :func:`prompt_token_hash` for computing stable SHA-256 digests of
token-id sequences (used by prefix caching).
"""

from __future__ import annotations

import hashlib

from ..utils import normalize_prompts, normalize_stop_sequences, truncate_tokens


def prompt_token_hash(token_ids: list["int"], *, max_tokens: int | None = None) -> str:
    """Compute a deterministic SHA-256 hex digest for a token-id sequence.

    Each token id is encoded as an 8-byte little-endian signed integer
    before being fed to the hasher, producing results that are stable
    across runs and platforms.

    Args:
        token_ids: List of integer token ids to hash.
        max_tokens: If given, only the first *max_tokens* ids are hashed.

    Returns:
        A lowercase hexadecimal SHA-256 digest string.
    """
    values = token_ids if max_tokens is None else token_ids[: max(int(max_tokens), 0)]
    digest = hashlib.sha256()
    for token_id in values:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


__all__ = ("normalize_prompts", "normalize_stop_sequences", "prompt_token_hash", "truncate_tokens")
