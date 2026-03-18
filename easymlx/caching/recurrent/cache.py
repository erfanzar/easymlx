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

"""Recurrent / SSM cache for Mamba, Mamba2, GatedDeltaNet, and similar models.

Three-tier pattern:
  RecurrentCacheConfig — static description
  RecurrentCacheView   — single-layer conv + recurrent state
  RecurrentCache       — multi-layer container
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView


@dataclass(slots=True)
class RecurrentCacheConfig(BaseCacheConfig):
    """Static description of a recurrent cache.

    *recurrent_state_shape* is the shape of the recurrent state **without** the
    batch dimension.  For example:
      - Mamba:  ``(intermediate_size, state_size)``
      - Mamba2: ``(num_heads, head_dim, state_size)``
    """

    batch_size: int = 1
    num_hidden_layers: int = 1
    conv_dim: int = 0
    conv_kernel_size: int = 4
    recurrent_state_shape: tuple[int, ...] = ()
    dtype: mx.Dtype = mx.float16

    @classmethod
    def create(cls, **kwargs: Any) -> RecurrentCacheConfig:
        """Create a :class:`RecurrentCacheConfig` from keyword arguments.

        Args:
            **kwargs: Fields of :class:`RecurrentCacheConfig`.

        Returns:
            A new configuration instance.
        """
        return cls(**kwargs)

    @classmethod
    def create_for_mamba(
        cls,
        *,
        batch_size: int,
        num_hidden_layers: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel_size: int = 4,
        dtype: mx.Dtype = mx.float16,
    ) -> RecurrentCacheConfig:
        """Create a configuration tailored for Mamba (SSM v1) models.

        The recurrent state shape is set to
        ``(intermediate_size, state_size)`` and the convolution dimension
        to ``intermediate_size``.

        Args:
            batch_size: Number of sequences in a batch.
            num_hidden_layers: Number of Mamba layers.
            intermediate_size: Inner dimension of the Mamba block.
            state_size: SSM state dimension.
            conv_kernel_size: Convolution kernel width.
            dtype: Data type for state buffers.

        Returns:
            A :class:`RecurrentCacheConfig` configured for Mamba.
        """
        return cls(
            batch_size=batch_size,
            num_hidden_layers=num_hidden_layers,
            conv_dim=intermediate_size,
            conv_kernel_size=conv_kernel_size,
            recurrent_state_shape=(intermediate_size, state_size),
            dtype=dtype,
        )

    @classmethod
    def create_for_mamba2(
        cls,
        *,
        batch_size: int,
        num_hidden_layers: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_dim: int,
        conv_kernel_size: int = 4,
        dtype: mx.Dtype = mx.float16,
    ) -> RecurrentCacheConfig:
        """Create a configuration tailored for Mamba2 models.

        The recurrent state shape is set to
        ``(num_heads, head_dim, state_size)``.

        Args:
            batch_size: Number of sequences in a batch.
            num_hidden_layers: Number of Mamba2 layers.
            num_heads: Number of SSM heads.
            head_dim: Per-head dimension.
            state_size: SSM state dimension.
            conv_dim: Convolution input dimension.
            conv_kernel_size: Convolution kernel width.
            dtype: Data type for state buffers.

        Returns:
            A :class:`RecurrentCacheConfig` configured for Mamba2.
        """
        return cls(
            batch_size=batch_size,
            num_hidden_layers=num_hidden_layers,
            conv_dim=conv_dim,
            conv_kernel_size=conv_kernel_size,
            recurrent_state_shape=(num_heads, head_dim, state_size),
            dtype=dtype,
        )


class RecurrentCacheView(BaseCacheView):
    """Mutable conv + recurrent state for a single layer.

    Attributes
    ----------
    conv_state : mx.array | None
        Rolling convolution buffer ``[batch, conv_dim, conv_kernel_size]``.
    recurrent_state : mx.array | None
        SSM / linear-attention state ``[batch, *recurrent_state_shape]``.
    positions : mx.array | None
        Current position indices ``[batch]``.
    seqlen_offset : int
        Cumulative sequence-length offset for continuation.
    """

    __slots__ = ("conv_state", "positions", "recurrent_state", "seqlen_offset")

    def __init__(
        self,
        *,
        conv_state: mx.array | None = None,
        recurrent_state: mx.array | None = None,
        positions: mx.array | None = None,
        seqlen_offset: int = 0,
    ) -> None:
        """Initialize a recurrent cache view for a single layer.

        Args:
            conv_state: Pre-allocated rolling convolution buffer of shape
                ``[batch, conv_dim, conv_kernel_size]``, or ``None``.
            recurrent_state: Pre-allocated recurrent state of shape
                ``[batch, *recurrent_state_shape]``, or ``None``.
            positions: Current position indices of shape ``[batch]``,
                or ``None``.
            seqlen_offset: Cumulative sequence-length offset for
                continuation across generation steps.
        """
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.positions = positions
        self.seqlen_offset = seqlen_offset

    @property
    def offset(self) -> int:
        """Return the current sequence-length offset."""
        return self.seqlen_offset

    @property
    def ssm_states(self) -> mx.array | None:
        """Return the recurrent (SSM) state, or ``None`` if uninitialized."""
        return self.recurrent_state

    def update_ssm_state(self, state: mx.array) -> None:
        """Replace the SSM recurrent state.

        Args:
            state: New recurrent state array.
        """
        self.recurrent_state = state

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
    ) -> tuple[mx.array, mx.array, RecurrentCacheView]:
        """Update recurrent state.

        For recurrent caches, *key_states* maps to ``conv_state`` and
        *value_states* maps to ``recurrent_state``.  The returned "key" and
        "value" are the updated conv and recurrent states respectively.
        """
        conv_state = key_states
        recurrent_state = value_states
        cache_position = kwargs.get("cache_position", None)

        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        if cache_position is not None:
            self.positions = cache_position
            self.seqlen_offset = int(mx.max(cache_position).item()) + 1
        else:
            self.seqlen_offset += 1

        return self.conv_state, self.recurrent_state, self

    def reset(self) -> None:
        """Reset all cached state to initial empty values."""
        self.conv_state = None
        self.recurrent_state = None
        self.positions = None
        self.seqlen_offset = 0


@dataclass(slots=True)
class RecurrentCache(BaseCache):
    """Multi-layer container of :class:`RecurrentCacheView` instances."""

    config: RecurrentCacheConfig
    _views: list[RecurrentCacheView | None] = field(default_factory=list)

    @property
    def views(self) -> list[RecurrentCacheView | None]:  # type: ignore["override"]
        """Return the ordered list of per-layer recurrent cache views."""
        return self._views

    @classmethod
    def init_cache(cls, config: RecurrentCacheConfig, **kwargs: Any) -> RecurrentCache:  # type: ignore["override"]
        """Allocate a recurrent cache from the given configuration.

        Each layer gets zero-initialized convolution and recurrent state
        buffers according to the config dimensions.

        Args:
            config: A :class:`RecurrentCacheConfig` describing state shapes.
            **kwargs: Reserved for future use.

        Returns:
            A new :class:`RecurrentCache` with pre-allocated views.
        """
        batch = config.batch_size
        dtype = config.dtype

        views: list[RecurrentCacheView | None] = []
        for _ in range(config.num_hidden_layers):
            conv_state = (
                mx.zeros((batch, config.conv_dim, config.conv_kernel_size), dtype=dtype) if config.conv_dim > 0 else None
            )
            rec_state = (
                mx.zeros((batch, *config.recurrent_state_shape), dtype=dtype) if config.recurrent_state_shape else None
            )
            views.append(
                RecurrentCacheView(
                    conv_state=conv_state,
                    recurrent_state=rec_state,
                    positions=mx.zeros((batch,), dtype=mx.int32),
                )
            )
        return cls(config=config, _views=views)

    def clear(self) -> None:
        """Reset all layer views to their initial empty state."""
        for view in self._views:
            if view is not None:
                view.reset()


__all__ = ("RecurrentCache", "RecurrentCacheConfig", "RecurrentCacheView")
