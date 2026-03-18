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

"""Attention runtime exports.

Provides access to all attention-related components including:

- Operation implementations (``Vanilla``, ``ScaledDotProductAttention``,
  ``UnifiedAttention``).
- Flexible runtime dispatch (:class:`FlexibleAttentionModule`,
  :func:`scaled_dot_product_attention`).
- Mask construction helpers (:func:`create_causal_mask`,
  :func:`build_attention_mask`).
- The unified attention performer (:class:`AttentionPerformer`).
- Paged attention utilities (:class:`PagedAttentionInputs`,
  :func:`prepare_paged_attention_inputs`).
"""

from __future__ import annotations

from easymlx.operations import (
    ScaledDotProductAttention,
    UnifiedAttention,
    UnifiedAttn,
    Vanilla,
    VanillaAttention,
)

from ._flexible import (
    AttentionMechanisms,
    FlexibleAttentionModule,
    build_attention_mask,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from ._performer import AttentionPerformer, PagedAttentionInputs, prepare_paged_attention_inputs

__all__ = (
    "AttentionMechanisms",
    "AttentionPerformer",
    "FlexibleAttentionModule",
    "PagedAttentionInputs",
    "ScaledDotProductAttention",
    "UnifiedAttention",
    "UnifiedAttn",
    "Vanilla",
    "VanillaAttention",
    "build_attention_mask",
    "create_attention_mask",
    "create_causal_mask",
    "prepare_paged_attention_inputs",
    "scaled_dot_product_attention",
)
