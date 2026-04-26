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

"""easymlx layer library.

Re-exports sub-packages providing reusable neural-network building blocks
optimized for MLX-based serving:

- :mod:`attention` -- attention kernels and runtime dispatch.
- :mod:`embeddings` -- embedding utilities (e.g. grid sampling).
- :mod:`linears` -- switch-style MoE linear layers.
- :mod:`moe` -- mixture-of-experts routing and feed-forward blocks.
- :mod:`quantization` -- quantized linear layer variants.
- :mod:`rotary` -- rotary positional embedding implementations.
"""

from __future__ import annotations

from . import attention, embeddings, linears, moe, quantization, rotary

__all__ = ("attention", "embeddings", "linears", "moe", "quantization", "rotary")
