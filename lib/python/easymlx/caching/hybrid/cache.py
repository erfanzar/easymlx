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

"""Hybrid cache for models mixing full attention and recurrent/linear layers.

Examples: FalconH1 (parallel hybrid), Qwen3Next (GatedDeltaNet + attention).

Three-tier pattern:
  HybridCacheConfig  — per-layer type routing table
  HybridCacheView    — single layer, allocates only what the layer type needs
  HybridCache        — multi-layer container
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
PARALLEL_HYBRID = "parallel_hybrid"


@dataclass(slots=True)
class HybridCacheConfig(BaseCacheConfig):
    """Static description of a hybrid cache.

    *layer_types* is a tuple with one entry per layer specifying the cache
    mode.  Supported values: ``"full_attention"``, ``"linear_attention"``,
    ``"parallel_hybrid"`` (both KV and recurrent).
    """

    batch_size: int = 1
    num_hidden_layers: int = 1
    layer_types: tuple[str, ...] = ()

    num_heads: int = 1
    head_dim: int = 64
    num_key_value_heads: int | None = None
    max_sequence_length: int = 4096
    sliding_window: int | None = None

    conv_dim: int = 0
    conv_kernel_size: int = 4
    recurrent_state_shape: tuple[int, ...] = ()

    dtype: mx.Dtype = mx.float16

    def __post_init__(self) -> None:
        """Validate and default-fill ``layer_types``.

        Raises:
            ValueError: If ``layer_types`` length does not equal
                ``num_hidden_layers``.
        """
        if not self.layer_types:
            self.layer_types = tuple(FULL_ATTENTION for _ in range(self.num_hidden_layers))
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal num_hidden_layers ({self.num_hidden_layers})"
            )

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
    def create(cls, **kwargs: Any) -> HybridCacheConfig:
        """Create a :class:`HybridCacheConfig` from keyword arguments.

        Args:
            **kwargs: Fields of :class:`HybridCacheConfig`.

        Returns:
            A new configuration instance.
        """
        return cls(**kwargs)


class HybridCacheView(BaseCacheView):
    """Single-layer cache that allocates only what the layer type needs.

    For ``full_attention`` layers: keys / values (standard KV cache).
    For ``linear_attention`` layers: conv_state / recurrent_state.
    For ``parallel_hybrid`` layers: all of the above.
    """

    __slots__ = (
        "_kv_offset",
        "conv_state",
        "keys",
        "layer_type",
        "recurrent_state",
        "seqlen_offset",
        "sliding_window",
        "values",
    )

    def __init__(
        self,
        *,
        layer_type: str = FULL_ATTENTION,
        sliding_window: int | None = None,
        conv_state: mx.array | None = None,
        recurrent_state: mx.array | None = None,
    ) -> None:
        """Initialize a hybrid cache view for a single layer.

        Args:
            layer_type: The type of layer this view services. One of
                ``"full_attention"``, ``"linear_attention"``, or
                ``"parallel_hybrid"``.
            sliding_window: Optional sliding-window size for attention layers.
                When set, the KV cache is truncated to the most recent
                *sliding_window* tokens.
            conv_state: Pre-allocated convolution state buffer for recurrent
                or parallel-hybrid layers.
            recurrent_state: Pre-allocated recurrent state buffer for
                recurrent or parallel-hybrid layers.
        """
        self.layer_type = layer_type
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self._kv_offset: int = 0
        self.sliding_window = sliding_window
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.seqlen_offset: int = 0

    @property
    def offset(self) -> int:
        """Return the current cache offset.

        For attention-based layer types (``full_attention``, ``parallel_hybrid``),
        returns the KV offset. For ``linear_attention``, returns the
        sequence-length offset.

        Returns:
            Number of tokens processed so far.
        """
        if self.layer_type in (FULL_ATTENTION, PARALLEL_HYBRID):
            return self._kv_offset
        return self.seqlen_offset

    def _concat_kv(self, key_states: mx.array, value_states: mx.array) -> tuple[mx.array, mx.array]:
        """Append KV states in BHLD layout ``[batch, heads, seq, dim]``.

        If a sliding window is configured and the resulting cache exceeds
        it, the oldest tokens are trimmed.

        Args:
            key_states: New key projections to append.
            value_states: New value projections to append.

        Returns:
            A tuple of the full (or window-truncated) key and value caches.
        """
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = mx.concatenate([self.keys, key_states], axis=2)
            self.values = mx.concatenate([self.values, value_states], axis=2)
        self._kv_offset += key_states.shape[2]

        if self.sliding_window is not None and self.keys.shape[2] > self.sliding_window:
            self.keys = self.keys[:, :, -self.sliding_window :]
            self.values = self.values[:, :, -self.sliding_window :]

        return self.keys, self.values

    def update_conv_state(self, conv_state: mx.array) -> None:
        """Replace the convolution state buffer.

        Args:
            conv_state: New convolution state array.
        """
        self.conv_state = conv_state

    def update_recurrent_state(self, state: mx.array) -> None:
        """Replace the recurrent state buffer.

        Args:
            state: New recurrent state array.
        """
        self.recurrent_state = state

    def concatenate_to_cache(
        self,
        key_states: mx.array,
        value_states: mx.array,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array, HybridCacheView]:
        """Route update to the appropriate handler based on *layer_type*.

        For ``full_attention``:
            key_states / value_states are standard KV projections.
        For ``linear_attention``:
            key_states → conv_state, value_states → recurrent_state.
        For ``parallel_hybrid``:
            Pass ``conv_state`` and ``recurrent_state`` via kwargs.
        """
        if self.layer_type == FULL_ATTENTION:
            k, v = self._concat_kv(key_states, value_states)
            return k, v, self

        if self.layer_type == LINEAR_ATTENTION:
            self.conv_state = key_states
            self.recurrent_state = value_states
            self.seqlen_offset += 1
            return key_states, value_states, self

        if self.layer_type == PARALLEL_HYBRID:
            k, v = self._concat_kv(key_states, value_states)
            if "conv_state" in kwargs:
                self.conv_state = kwargs["conv_state"]
            if "recurrent_state" in kwargs:
                self.recurrent_state = kwargs["recurrent_state"]
            self.seqlen_offset += 1
            return k, v, self

        raise ValueError(f"Unknown layer_type: {self.layer_type!r}")

    def reset(self) -> None:
        """Reset all cached state to initial empty values."""
        self.keys = None
        self.values = None
        self._kv_offset = 0
        self.conv_state = None
        self.recurrent_state = None
        self.seqlen_offset = 0


@dataclass(slots=True)
class HybridCache(BaseCache):
    """Multi-layer container of :class:`HybridCacheView` instances."""

    config: HybridCacheConfig
    _views: list[HybridCacheView | None] = field(default_factory=list)

    @property
    def views(self) -> list[HybridCacheView | None]:
        """Return the ordered list of per-layer hybrid cache views."""
        return self._views

    @classmethod
    def init_cache(cls, config: HybridCacheConfig, **kwargs: Any) -> HybridCache:
        """Allocate a hybrid cache from the given configuration.

        Each layer is initialized according to its ``layer_type`` in the
        config: attention layers get ``None`` conv/recurrent state, while
        linear-attention and parallel-hybrid layers get zero-initialized
        state buffers.

        Args:
            config: A :class:`HybridCacheConfig` with per-layer type routing.
            **kwargs: Reserved for future use.

        Returns:
            A new :class:`HybridCache` with pre-allocated views.
        """
        batch = config.batch_size
        dtype = config.dtype

        views: list[HybridCacheView | None] = []
        for layer_type in config.layer_types:
            conv_state = None
            rec_state = None

            if layer_type in (LINEAR_ATTENTION, PARALLEL_HYBRID):
                if config.conv_dim > 0:
                    conv_state = mx.zeros(
                        (batch, config.conv_dim, config.conv_kernel_size),
                        dtype=dtype,
                    )
                if config.recurrent_state_shape:
                    rec_state = mx.zeros(
                        (batch, *config.recurrent_state_shape),
                        dtype=dtype,
                    )

            views.append(
                HybridCacheView(
                    layer_type=layer_type,
                    sliding_window=config.sliding_window,
                    conv_state=conv_state,
                    recurrent_state=rec_state,
                )
            )
        return cls(config=config, _views=views)

    def clear(self) -> None:
        """Reset all layer views to their initial empty state."""
        for view in self._views:
            if view is not None:
                view.reset()


__all__ = (
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "PARALLEL_HYBRID",
    "HybridCache",
    "HybridCacheConfig",
    "HybridCacheView",
)
