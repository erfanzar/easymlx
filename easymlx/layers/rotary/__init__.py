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

"""Rotary positional embedding exports.

Re-exports rotary embedding implementations and the factory function:

- :class:`Llama3RoPE` -- Llama 3-style rotary embeddings.
- :class:`SuScaledRoPE` -- SuScaled rotary embeddings.
- :class:`YarnRoPE` -- YaRN (Yet another RoPE extensioN) embeddings.
- :func:`get_rope` -- Factory function to create a RoPE instance from
  a config.
"""

from __future__ import annotations

from ._rotary import Llama3RoPE, SuScaledRoPE, YarnRoPE, get_rope

__all__ = ("Llama3RoPE", "SuScaledRoPE", "YarnRoPE", "get_rope")
