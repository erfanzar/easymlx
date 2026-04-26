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

"""Attention kernel implementations.

This package provides concrete attention kernel implementations including:

- ``ScaledDotProductAttention``: Wrapper around ``mx.fast.scaled_dot_product_attention``.
- ``VanillaAttention``: Explicit matmul-based attention with attention weight output.
- ``UnifiedAttention``: Paged attention with Metal kernel and NumPy reference fallback.
- ``PageAttention``: Dedicated paged attention runtime backed by block-tabled KV caches.
"""

from __future__ import annotations

from .gated_delta_rule import GatedDeltaRuleOp, GatedDeltaRuleOutput
from .page_attention import PageAttention, PageAttn, page_attention, paged_attention
from .scaled_dot_product_attention import ScaledDotProductAttention
from .unified_attention import UnifiedAttention, UnifiedAttnConfig, UnifiedAttnMetadata
from .vanilla_attention import Vanilla, VanillaAttention

ScaledDotProductAttn = ScaledDotProductAttention
UnifiedAttn = UnifiedAttention
VanillaAttn = VanillaAttention
GatedDeltaRule = GatedDeltaRuleOp

__all__ = (
    "GatedDeltaRule",
    "GatedDeltaRuleOp",
    "GatedDeltaRuleOutput",
    "PageAttention",
    "PageAttn",
    "ScaledDotProductAttention",
    "ScaledDotProductAttn",
    "UnifiedAttention",
    "UnifiedAttn",
    "UnifiedAttnConfig",
    "UnifiedAttnMetadata",
    "Vanilla",
    "VanillaAttention",
    "VanillaAttn",
    "page_attention",
    "paged_attention",
)
