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

"""Flexible attention runtime over easymlx operations.

Provides a runtime dispatcher that selects between dense (SDPA / vanilla)
and paged attention execution paths based on the presence of cache
metadata. Also includes mask-construction helpers used throughout the
model layer code.
"""

from __future__ import annotations

import typing as tp
from enum import StrEnum
from functools import lru_cache

import mlx.core as mx

from easymlx.caching import PageCacheView, PageMetadata
from easymlx.operations import (
    AttentionOutput,
    ExecutionMode,
    OperationExecutor,
    OperationMetadata,
    OperationRegistry,
)


class AttentionMechanisms(StrEnum):
    """Supported attention mechanism identifiers.

    Attributes:
        AUTO: Automatically select the best available mechanism.
        SDPA: Scaled dot-product attention.
        VANILLA: Naive (loop-based) attention implementation.
        PAGED_ATTENTION: Paged attention for serving workloads.
        UNIFIED_ATTENTION: Unified attention supporting both dense and
            paged paths.
    """

    AUTO = "auto"
    SDPA = "sdpa"
    VANILLA = "vanilla"
    PAGED_ATTENTION = "page_attention"
    UNIFIED_ATTENTION = "unified_attention"


_DEFAULT_PAGED_ATTENTION_MECHANISM = AttentionMechanisms.UNIFIED_ATTENTION


def create_causal_mask(seq_len: int, *, window_size: int | None = None) -> mx.array:
    """Create a lower-triangular boolean causal mask.

    Args:
        seq_len: The sequence length.
        window_size: If provided, restrict attention to a sliding
            window of this size. Positions outside the window are
            masked.

    Returns:
        A boolean ``mx.array`` of shape ``[seq_len, seq_len]`` where
        ``True`` indicates an allowed attention position.
    """
    indices = mx.arange(seq_len, dtype=mx.int32)
    q_idx = indices[:, None]
    k_idx = indices[None, :]
    mask = q_idx >= k_idx
    if window_size is not None:
        mask = mask & ((q_idx - k_idx) < int(window_size))
    return mask


def _normalize_attention_mask(attention_mask: mx.array | tp.Any, *, batch_size: int, seq_len: int) -> mx.array:
    """Normalize an attention mask to shape ``[batch_size, seq_len]``.

    Args:
        attention_mask: A 1-D or 2-D mask indicating valid token
            positions.
        batch_size: The expected batch dimension.
        seq_len: The expected sequence dimension.

    Returns:
        A boolean ``mx.array`` of shape ``[batch_size, seq_len]``.
    """
    token_mask = mx.array(attention_mask).astype(mx.bool_)
    if token_mask.ndim == 1:
        token_mask = token_mask[None, :]
    if token_mask.shape[0] == 1 and batch_size != 1:
        token_mask = mx.broadcast_to(token_mask, (batch_size, seq_len))
    return token_mask


def build_attention_mask(
    attention_mask: mx.array | tp.Any | None,
    *,
    batch_size: int,
    seq_len: int,
    window_size: int | None = None,
) -> mx.array | str | None:
    """Build a 4-D attention mask suitable for SDPA kernels.

    When no explicit mask is needed (single-token decode, no sliding
    window), returns ``None`` or the string ``"causal"`` as a sentinel
    for kernel-level optimisation.

    Args:
        attention_mask: Optional 1-D or 2-D token mask. ``None`` means
            all tokens are valid.
        batch_size: Batch dimension.
        seq_len: Sequence dimension.
        window_size: Optional sliding-window size.

    Returns:
        A 4-D boolean ``mx.array`` of shape
        ``[batch, 1, seq_len, seq_len]``, the string ``"causal"``, or
        ``None``.
    """
    if attention_mask is None and window_size is None:
        return None if seq_len <= 1 else "causal"

    causal = create_causal_mask(seq_len, window_size=window_size)
    if attention_mask is None:
        return causal[None, None, :, :]

    token_mask = _normalize_attention_mask(attention_mask, batch_size=batch_size, seq_len=seq_len)
    return causal[None, None, :, :] & token_mask[:, None, None, :]


def create_attention_mask(
    h: mx.array,
    attention_mask: mx.array | tp.Any | None = None,
    *,
    window_size: int | None = None,
    cache: tp.Any | None = None,
) -> mx.array | str | None:
    """Create an attention mask from hidden states and an optional token mask.

    Convenience wrapper around :func:`build_attention_mask` that
    infers ``batch_size`` and ``seq_len`` from the shape of *h*.

    Args:
        h: Hidden states of shape ``[batch, seq_len, ...]``.
        attention_mask: Optional token-level mask.
        window_size: Optional sliding-window size.
        cache: Unused; present for API compatibility.

    Returns:
        See :func:`build_attention_mask`.
    """
    del cache
    batch_size, seq_len = h.shape[:2]
    return build_attention_mask(
        attention_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        window_size=window_size,
    )


def _normalize_mechanism(value: AttentionMechanisms | str | None) -> AttentionMechanisms:
    """Coerce a string or ``None`` to an :class:`AttentionMechanisms` enum.

    Args:
        value: Mechanism name or ``None``.

    Returns:
        The corresponding enum member. ``None`` maps to ``AUTO``.
    """
    if value is None:
        return AttentionMechanisms.AUTO
    if isinstance(value, AttentionMechanisms):
        return value
    mechanism = str(value).lower()
    aliases = {
        "paged": AttentionMechanisms.UNIFIED_ATTENTION,
        "unified": AttentionMechanisms.UNIFIED_ATTENTION,
        "paged_attention": AttentionMechanisms.PAGED_ATTENTION,
    }
    if mechanism in aliases:
        return aliases[mechanism]
    return AttentionMechanisms(mechanism)


def _coerce_cache_metadata(
    cache: tp.Any | None,
    cache_metadata: PageMetadata | None,
) -> PageMetadata | None:
    """Extract :class:`PageMetadata` from arguments.

    If *cache_metadata* is already provided it is returned directly.
    Otherwise, if *cache* is a :class:`PageMetadata` instance it is
    used instead.

    Args:
        cache: A generic cache argument that may be ``PageMetadata``.
        cache_metadata: An explicit metadata argument.

    Returns:
        A :class:`PageMetadata` or ``None``.
    """
    if cache_metadata is not None:
        return cache_metadata
    if isinstance(cache, PageMetadata):
        return cache
    return None


def _resolve_dense_impl_name(mechanism: AttentionMechanisms) -> str:
    """Map an :class:`AttentionMechanisms` value to a dense operation name.

    Args:
        mechanism: The requested mechanism.

    Returns:
        The operation name string for the dense path.

    Raises:
        ValueError: If *mechanism* is not a supported dense mechanism.
    """
    if mechanism in {AttentionMechanisms.AUTO, AttentionMechanisms.SDPA}:
        return AttentionMechanisms.SDPA.value
    if mechanism == AttentionMechanisms.VANILLA:
        return AttentionMechanisms.VANILLA.value
    raise ValueError(f"Unsupported dense attention mechanism: {mechanism.value!r}")


def _resolve_paged_impl_name(mechanism: AttentionMechanisms) -> str:
    """Map an :class:`AttentionMechanisms` value to a paged operation name.

    Args:
        mechanism: The requested mechanism.

    Returns:
        The operation name string for the paged path.

    Raises:
        ValueError: If *mechanism* is not a supported paged mechanism.
    """
    if mechanism == AttentionMechanisms.PAGED_ATTENTION:
        return AttentionMechanisms.PAGED_ATTENTION.value
    if mechanism in {AttentionMechanisms.AUTO, AttentionMechanisms.UNIFIED_ATTENTION}:
        return AttentionMechanisms.UNIFIED_ATTENTION.value
    raise ValueError(f"Unsupported paged attention mechanism: {mechanism.value!r}")


def _unwrap_attention_output(result: AttentionOutput | mx.array) -> mx.array:
    """Extract the raw attention output array from an :class:`AttentionOutput`.

    Args:
        result: Either a raw ``mx.array`` or an :class:`AttentionOutput`
            wrapper.

    Returns:
        The attention output tensor.

    Raises:
        ValueError: If *result* is an :class:`AttentionOutput` with
            ``attention_outputs`` set to ``None``.
    """
    if isinstance(result, AttentionOutput):
        if result.attention_outputs is None:
            raise ValueError("Attention operation returned no attention_outputs.")
        return result.attention_outputs
    return result


class FlexibleAttentionModule:
    """Unified runtime switch for dense and paged attention execution.

    Maintains two :class:`OperationExecutor` instances -- one for dense
    attention and one for paged attention -- and selects the appropriate
    one at call time based on the presence of *cache_metadata* and
    *cache_view* arguments.

    Args:
        attention_mechanism: The dense attention mechanism to use.
        paged_attention_mechanism: The paged attention mechanism to use.
        metadata: Optional operation metadata forwarded to both
            executors.
    """

    def __init__(
        self,
        *,
        attention_mechanism: AttentionMechanisms | str = AttentionMechanisms.AUTO,
        paged_attention_mechanism: AttentionMechanisms | str = _DEFAULT_PAGED_ATTENTION_MECHANISM,
        metadata: OperationMetadata | None = None,
    ) -> None:
        """Initialize the flexible attention module.

        Args:
            attention_mechanism: Dense attention mechanism identifier.
            paged_attention_mechanism: Paged attention mechanism
                identifier.
            metadata: Optional metadata forwarded to the operation
                constructors.
        """
        self.attention_mechanism = _normalize_mechanism(attention_mechanism)
        self.paged_attention_mechanism = _normalize_mechanism(paged_attention_mechanism)
        self.metadata = metadata
        self._dense_executor = OperationExecutor.from_operations(
            mixin=OperationRegistry.create(_resolve_dense_impl_name(self.attention_mechanism), metadata=metadata)
        )
        self._paged_executor = OperationExecutor.from_operations(
            mixin=OperationRegistry.create(_resolve_paged_impl_name(self.paged_attention_mechanism), metadata=metadata)
        )

    def _get_executor(
        self,
        *,
        cache_metadata: PageMetadata | None,
        cache_view: PageCacheView | None,
    ) -> OperationExecutor:
        """Select the appropriate executor based on cache arguments.

        Args:
            cache_metadata: Paged attention metadata (if available).
            cache_view: Paged cache view (if available).

        Returns:
            The paged executor if both *cache_metadata* and
            *cache_view* are provided, otherwise the dense executor.

        Raises:
            ValueError: If only one of *cache_metadata* and
                *cache_view* is provided.
        """
        if cache_metadata is not None or cache_view is not None:
            if cache_metadata is None or cache_view is None:
                raise ValueError("cache_metadata and cache_view must be provided together for paged attention.")
            return self._paged_executor
        return self._dense_executor

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        cache: tp.Any | None = None,
        cache_metadata: PageMetadata | None = None,
        cache_view: PageCacheView | None = None,
        scale: float,
        mask: mx.array | str | None = None,
        sinks: mx.array | None = None,
        return_output: bool = False,
        **kwargs: tp.Any,
    ) -> AttentionOutput | mx.array:
        """Execute attention with automatic dense/paged routing.

        Args:
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            cache: Generic cache argument (may contain
                :class:`PageMetadata`).
            cache_metadata: Explicit paged attention metadata.
            cache_view: Paged cache view object (:class:`PageCacheView`
                or ``None``).
            scale: Attention scale factor (typically
                ``1 / sqrt(head_dim)``).
            mask: Attention mask or the string ``"causal"``.
            sinks: Optional attention sink tokens.
            return_output: If ``True``, return the full
                :class:`AttentionOutput` wrapper.
            **kwargs: Additional keyword arguments forwarded to the
                operation.

        Returns:
            If *return_output* is ``True``, returns an
            :class:`AttentionOutput`. Otherwise returns the raw
            ``mx.array`` of attention outputs.

        Raises:
            RuntimeError: If no attention operation is configured.
        """
        cache_metadata = _coerce_cache_metadata(cache, cache_metadata)
        executor = self._get_executor(cache_metadata=cache_metadata, cache_view=cache_view)
        mode = ExecutionMode.MIXED if cache_metadata is not None else ExecutionMode.PREFILL
        operation = executor.get_operation(mode=mode)
        if operation is None:
            raise RuntimeError("No attention operation is configured.")
        result = operation(
            query=queries,
            key=keys,
            value=values,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            scale=scale,
            mask=mask,
            sinks=sinks,
            **kwargs,
        )
        if not isinstance(result, AttentionOutput):
            result = AttentionOutput(attention_outputs=result)
        return result if return_output else _unwrap_attention_output(result)


@lru_cache(maxsize=8)
def _get_default_runtime(
    attention_mechanism: AttentionMechanisms | str,
    paged_attention_mechanism: AttentionMechanisms | str,
) -> FlexibleAttentionModule:
    """Get or create a cached :class:`FlexibleAttentionModule` instance.

    Args:
        attention_mechanism: Dense mechanism name.
        paged_attention_mechanism: Paged mechanism name.

    Returns:
        A cached :class:`FlexibleAttentionModule` instance.
    """
    return FlexibleAttentionModule(
        attention_mechanism=attention_mechanism,
        paged_attention_mechanism=paged_attention_mechanism,
    )


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    cache: tp.Any | None = None,
    cache_metadata: PageMetadata | None = None,
    cache_view: PageCacheView | None = None,
    scale: float,
    mask: mx.array | str | None = None,
    sinks: mx.array | None = None,
    attention_mechanism: AttentionMechanisms | str = AttentionMechanisms.AUTO,
    paged_attention_mechanism: AttentionMechanisms | str = _DEFAULT_PAGED_ATTENTION_MECHANISM,
    return_output: bool = False,
    **kwargs: tp.Any,
) -> AttentionOutput | mx.array:
    """Functional API for scaled dot-product attention.

    Dispatches to a cached :class:`FlexibleAttentionModule` based on
    the requested mechanisms.

    Args:
        queries: Query tensor.
        keys: Key tensor.
        values: Value tensor.
        cache: Generic cache argument.
        cache_metadata: Paged attention metadata.
        cache_view: Paged cache view.
        scale: Attention scale factor.
        mask: Attention mask or ``"causal"`` sentinel.
        sinks: Optional attention sink tokens.
        attention_mechanism: Dense mechanism to use.
        paged_attention_mechanism: Paged mechanism to use.
        return_output: If ``True``, return the full
            :class:`AttentionOutput`.
        **kwargs: Forwarded to the underlying operation.

    Returns:
        An ``mx.array`` of attention outputs, or an
        :class:`AttentionOutput` if *return_output* is ``True``.
    """
    module = _get_default_runtime(
        attention_mechanism=attention_mechanism,
        paged_attention_mechanism=paged_attention_mechanism,
    )
    return module(
        queries,
        keys,
        values,
        cache=cache,
        cache_metadata=cache_metadata,
        cache_view=cache_view,
        scale=scale,
        mask=mask,
        sinks=sinks,
        return_output=return_output,
        **kwargs,
    )


__all__ = (
    "AttentionMechanisms",
    "FlexibleAttentionModule",
    "build_attention_mask",
    "create_attention_mask",
    "create_causal_mask",
    "scaled_dot_product_attention",
)
