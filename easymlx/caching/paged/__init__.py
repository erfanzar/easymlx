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

"""Paged KV cache for serving and unified attention.

Re-exports :class:`PageCache`, :class:`PageMetadata`, :class:`PagedKVCache`,
and the :func:`build_query_start_loc` helper from the implementation modules.
"""

from .page_cache import PageCache
from .page_metadata import PageMetadata
from .paged_kv_cache import PagedKVCache, build_query_start_loc

__all__ = ("PageCache", "PageMetadata", "PagedKVCache", "build_query_start_loc")
