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

"""Mixture-of-experts compatibility exports.

Re-exports MoE building blocks from both the linears and MoE
sub-packages:

- :class:`SwitchGLU`, :class:`SwitchLinear`, :class:`SwitchMLP` --
  expert feed-forward layers from :mod:`easymlx.layers.linears`.
- :class:`TopKRouter` -- top-k expert routing module.
- :func:`topk_expert_select` -- functional top-k expert selection.
"""

from __future__ import annotations

from easymlx.layers.linears import SwitchGLU, SwitchLinear, SwitchMLP

from ._moe_module import TopKRouter, topk_expert_select

__all__ = ("SwitchGLU", "SwitchLinear", "SwitchMLP", "TopKRouter", "topk_expert_select")
