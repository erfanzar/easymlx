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

"""Standard transformer KV cache for auto-regressive generation.

Three-tier pattern:
  TransformerCacheConfig  — static description
  TransformerCacheView    — single-layer mutable key/value storage
  TransformerCache        — multi-layer container
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView


@dataclass(slots=True)
class TransformerCacheConfig(BaseCacheConfig):
    """Static description of a standard KV cache."""

    batch_size: int = 1
    num_hidden_layers: int = 1
    num_heads: int = 1
    head_dim: int = 64
    max_sequence_length: int = 4096
    num_key_value_heads: int | None = None
    sliding_window: int | None = None
    dtype: mx.Dtype = mx.float16

    @property
    def effective_num_kv_heads(self) -> int:
        """Return the effective number of key/value heads.

        Falls back to ``num_heads`` when ``num_key_value_heads`` is ``None``
        (i.e., multi-head attention without grouped-query attention).

        Returns:
            The number of KV heads to use.
        """
        return self.num_key_value_heads or self.num_heads

    @classmethod
    def create(cls, **kwargs: Any) -> TransformerCacheConfig:
        """Create a :class:`TransformerCacheConfig` from keyword arguments.

        Args:
            **kwargs: Fields of :class:`TransformerCacheConfig`.

        Returns:
            A new configuration instance.
        """
        return cls(**kwargs)


class TransformerCacheView(BaseCacheView):
    """Mutable key/value state for a single transformer layer.

    Stores KV tensors in BHLD layout: ``[batch, num_kv_heads, seq_len, head_dim]``.
    This matches the layout expected by ``mx.fast.scaled_dot_product_attention``
    and ``mx.fast.rope``, eliminating transposes in the attention hot path.

    On each call to :meth:`concatenate_to_cache` the new K/V are appended
    and the full (possibly truncated by sliding window) cache is returned.
    """

    __slots__ = ("_offset", "keys", "max_sequence_length", "sliding_window", "values")

    def __init__(
        self,
        *,
        sliding_window: int | None = None,
        max_sequence_length: int = 4096,
    ) -> None:
        """Initialize a transformer cache view for a single layer.

        Args:
            sliding_window: Optional sliding-window size. When set, the cache
                is truncated to the most recent *sliding_window* tokens after
                each append.
            max_sequence_length: Maximum number of tokens the cache is
                designed to hold.
        """
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self._offset: int = 0
        self.sliding_window = sliding_window
        self.max_sequence_length = max_sequence_length

    @property
    def offset(self) -> int:
        """Return the total number of tokens that have been appended.

        Note that ``offset`` may differ from :attr:`seq_len` when a sliding
        window is active, since ``offset`` counts all appended tokens while
        ``seq_len`` counts only those still retained in the cache.
        """
        return self._offset

    @property
    def seq_len(self) -> int:
        """Return the number of tokens currently retained in the cache."""
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def concatenate_to_cache(
        self,
        key_states: mx.array,
        value_states: mx.array,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array, TransformerCacheView]:
        """Append *key_states* / *value_states* and return full cache.

        Parameters
        ----------
        key_states, value_states : mx.array
            Shape ``[batch, num_kv_heads, new_tokens, head_dim]`` (BHLD).

        Returns
        -------
        key_cache, value_cache : mx.array
            Full (or sliding-window-truncated) key/value cache in BHLD.
        updated_view : TransformerCacheView
            ``self`` (mutated in-place for MLX eager semantics).
        """
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = mx.concatenate([self.keys, key_states], axis=2)
            self.values = mx.concatenate([self.values, value_states], axis=2)

        self._offset += key_states.shape[2]

        if self.sliding_window is not None and self.keys.shape[2] > self.sliding_window:
            self.keys = self.keys[:, :, -self.sliding_window :]
            self.values = self.values[:, :, -self.sliding_window :]

        return self.keys, self.values, self

    def reset(self) -> None:
        """Clear this layer's cache."""
        self.keys = None
        self.values = None
        self._offset = 0


@dataclass(slots=True)
class TransformerCache(BaseCache):
    """Multi-layer container of :class:`TransformerCacheView` instances."""

    config: TransformerCacheConfig
    _views: list[TransformerCacheView | None] = field(default_factory=list)

    @property
    def views(self) -> list[TransformerCacheView | None]:
        """Return the ordered list of per-layer transformer cache views."""
        return self._views

    @classmethod
    def init_cache(cls, config: TransformerCacheConfig, **kwargs: Any) -> TransformerCache:
        """Allocate a transformer cache from the given configuration.

        Creates one :class:`TransformerCacheView` per layer with the
        configured sliding window and maximum sequence length.

        Args:
            config: A :class:`TransformerCacheConfig` describing cache
                dimensions.
            **kwargs: Reserved for future use.

        Returns:
            A new :class:`TransformerCache` with empty per-layer views.
        """
        views: list[TransformerCacheView | None] = [
            TransformerCacheView(
                sliding_window=config.sliding_window,
                max_sequence_length=config.max_sequence_length,
            )
            for _ in range(config.num_hidden_layers)
        ]
        return cls(config=config, _views=views)

    def clear(self) -> None:
        """Reset all layer views to their initial empty state."""
        for view in self._views:
            if view is not None:
                view.reset()

    @property
    def offset(self) -> int:
        """Return the current offset of the first non-``None`` view.

        Returns:
            The token offset of the first active view, or ``0`` if all
            views are ``None``.
        """
        for view in self._views:
            if view is not None:
                return view.offset
        return 0


__all__ = ("TransformerCache", "TransformerCacheConfig", "TransformerCacheView")
