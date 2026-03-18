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

"""Mixins for easymlx infra (serving-only).

Re-exports the three core mixins that compose
:class:`~easymlx.infra.base_module.EasyMLXBaseModule`:

- :class:`EasyBridgeMixin` -- checkpoint I/O and HuggingFace conversion.
- :class:`EasyGenerationMixin` -- greedy and sampling text generation.
- :class:`OperationCacheMixin` -- cache requirement discovery.
"""

from .bridge import EasyBridgeMixin
from .generation import EasyGenerationMixin
from .operation_cache import OperationCacheMixin

__all__ = ("EasyBridgeMixin", "EasyGenerationMixin", "OperationCacheMixin")
