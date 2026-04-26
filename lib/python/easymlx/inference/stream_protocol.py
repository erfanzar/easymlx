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

"""Streaming protocol helpers shared by inference parsers."""

from __future__ import annotations


def compute_stream_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
    """Compute a safe streaming delta from accumulated text snapshots."""

    current_text = current_text or ""
    previous_text = previous_text or ""
    fallback_delta = fallback_delta or ""

    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]

    if not current_text and previous_text and not fallback_delta:
        return ""

    max_overlap = min(len(previous_text), len(current_text))
    for overlap in range(max_overlap, 0, -1):
        if previous_text.endswith(current_text[:overlap]):
            return current_text[overlap:]

    if len(current_text) <= len(previous_text):
        if fallback_delta and not previous_text.endswith(fallback_delta):
            return fallback_delta
        return ""

    if fallback_delta and (not previous_text or not previous_text.endswith(fallback_delta)):
        return fallback_delta
    return current_text if not previous_text else ""


__all__ = ("compute_stream_delta_text",)
