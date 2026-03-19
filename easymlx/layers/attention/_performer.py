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

"""Shared attention performer -- the single entry point for all attention.

Model modules project Q/K/V and call ``AttentionPerformer.__call__`` which
handles layout transforms, RoPE, cache updates, and kernel dispatch.
No branching should exist in the model attention modules themselves.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import mlx.core as mx

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView

from ._flexible import scaled_dot_product_attention


@dataclass(slots=True)
class PagedAttentionInputs:
    """Container for prepared paged attention inputs.

    Holds the post-processed queries, key/value caches, and metadata
    ready for a paged attention kernel.

    Attributes:
        queries: Prepared query tensor.
        key_cache: The key cache tensor.
        value_cache: The value cache tensor.
        metadata: The paged attention metadata.
    """

    queries: mx.array
    key_cache: mx.array
    value_cache: mx.array
    metadata: PageMetadata


def prepare_paged_attention_inputs(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    kv_cache: PageCacheView | PageCacheView,
    query_start_loc: tp.Sequence["int"] | PageMetadata,
    slot_ids: tp.Sequence["int"] | None = None,
    rope: tp.Any = None,
    sliding_window: int | None = None,
) -> PagedAttentionInputs:
    """Prepare inputs for paged attention execution.

    Wraps the cache and metadata objects, then calls
    ``cache_view.concatenate_to_cache`` to insert new keys/values and
    optionally apply RoPE.

    Args:
        queries: Query tensor of shape ``[T, num_heads, head_dim]`` or
            ``[B, L, num_heads, head_dim]``.
        keys: Key tensor with matching layout.
        values: Value tensor with matching layout.
        kv_cache: A :class:`PageCacheView` or :class:`PageCacheView`
            instance.
        query_start_loc: Either a sequence of per-sequence query start
            locations or a :class:`PageMetadata` object.
        slot_ids: Optional slot IDs for the paged cache. Required if
            *kv_cache* is a raw :class:`PageCacheView` without metadata
            carrying slot_ids.
        rope: Optional RoPE module applied inside the cache
            concatenation.
        sliding_window: Optional sliding-window size for the page
            metadata.

    Returns:
        A :class:`PagedAttentionInputs` containing the prepared
        tensors and metadata.
    """
    cache_metadata = (
        query_start_loc
        if isinstance(query_start_loc, PageMetadata)
        else PageMetadata(
            query_start_loc=query_start_loc,
            sliding_window=int(sliding_window) if sliding_window is not None else 0,
        )
    )
    resolved_slots = slot_ids or getattr(cache_metadata, "slot_ids", None) or ()
    prepared_queries, key_cache, value_cache, metadata = kv_cache.concatenate_to_cache(
        queries,
        keys,
        values,
        cache_metadata=cache_metadata,
        slot_ids=resolved_slots,
        rope=rope,
    )
    return PagedAttentionInputs(
        queries=prepared_queries,
        key_cache=key_cache,
        value_cache=value_cache,
        metadata=metadata,
    )


class AttentionPerformer:
    """Unified attention kernel entrypoint.

    Model attention modules should only:

    1. Project Q/K/V to ``[..., num_heads, head_dim]``
    2. Call ``self.attention_performer(q, k, v, ...)``
    3. Output-project the result

    This class handles layout transforms, RoPE application, cache
    updates, and kernel routing for both dense and paged paths.

    Args:
        scale: Attention scale factor, typically
            ``1 / sqrt(head_dim)``.
    """

    def __init__(
        self,
        *,
        scale: float,
        attn_mechanism: str | None = None,
    ):
        """Initialize the attention performer.

        Args:
            scale: The attention scale factor.
            attn_mechanism: Paged attention mechanism override. If ``None``,
                uses the module-level default (unified_attention). Pass
                ``"paged"`` or ``"page_attention"`` to use the page_attention
                Metal kernels instead.
        """
        self.scale = float(scale)
        self.attn_mechanism = attn_mechanism

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: TransformerCacheView | PageCacheView | None = None,
        cache_metadata: PageMetadata | None = None,
        rope: tp.Any = None,
    ) -> mx.array:
        """Unified attention entry point.

        Routes to :meth:`_forward_paged` or :meth:`_forward_dense`
        based on the type of *cache_view*.

        Args:
            queries: Query tensor. Shape ``[B, L, num_heads, head_dim]``
                (dense) or ``[T, num_heads, head_dim]`` (paged, flat
                tokens).
            keys: Key tensor with matching layout.
            values: Value tensor with matching layout.
            mask: Attention mask for the dense path. Ignored for paged.
            cache_view: Cache view object. A :class:`PageCacheView`
                triggers the paged path; a
                :class:`TransformerCacheView` or ``None`` triggers the
                dense path.
            cache_metadata: Required when *cache_view* is a
                :class:`PageCacheView`.
            rope: Optional RoPE module (applied internally).

        Returns:
            Attention output with the same leading shape as *queries*.
        """
        if isinstance(cache_view, PageCacheView):
            return self._forward_paged(
                queries,
                keys,
                values,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                rope=rope,
            )
        return self._forward_dense(
            queries,
            keys,
            values,
            mask=mask,
            cache_view=cache_view,
            rope=rope,
        )

    def _forward_paged(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        cache_view: PageCacheView,
        cache_metadata: PageMetadata | None,
        rope: tp.Any,
    ) -> mx.array:
        """Execute the paged attention path.

        Handles both flat ``[T, H, D]`` and batched ``[B, L, H, D]``
        inputs by reshaping to flat format before executing and
        restoring the original shape afterward.

        Args:
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            cache_view: The paged cache view.
            cache_metadata: Paged attention metadata.
            rope: Optional RoPE module.

        Returns:
            Attention output with the same leading shape as *queries*.
        """
        input_4d = queries.ndim == 4
        if input_4d:
            B, L = queries.shape[0], queries.shape[1]
            queries = queries.reshape(-1, queries.shape[-2], queries.shape[-1])
            keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
            values = values.reshape(-1, values.shape[-2], values.shape[-1])

        prepared = prepare_paged_attention_inputs(
            queries,
            keys,
            values,
            kv_cache=cache_view,
            query_start_loc=cache_metadata,
            rope=rope,
        )
        if prepared.queries.shape[0] == 0:
            out = queries
        else:
            out = self.forward(
                prepared.queries,
                prepared.key_cache,
                prepared.value_cache,
                cache_metadata=prepared.metadata,
                cache_view=cache_view,
            )

        if input_4d:
            out = out.reshape(B, L, out.shape[-2], out.shape[-1])
        return out

    def _forward_dense(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        mask: mx.array | str | None,
        cache_view: TransformerCacheView | None,
        rope: tp.Any,
    ) -> mx.array:
        """Execute the dense attention path.

        Transposes from ``[B, L, H, D]`` to ``[B, H, L, D]`` for SDPA
        computation, applies RoPE and cache updates, then transposes
        back.

        Args:
            queries: Query tensor of shape ``[B, L, H, D]``.
            keys: Key tensor of shape ``[B, L, Hkv, D]``.
            values: Value tensor of shape ``[B, L, Hkv, D]``.
            mask: Attention mask.
            cache_view: Optional transformer cache view for
                incremental decoding.
            rope: Optional RoPE module.

        Returns:
            Attention output of shape ``[B, L, H, D]``.
        """
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache_view.offset if cache_view is not None else 0
        if rope is not None:
            queries = rope(queries, offset=offset)
            keys = rope(keys, offset=offset)

        if cache_view is not None:
            keys, values, _ = cache_view.concatenate_to_cache(keys, values)

        out = self.forward(queries, keys, values, mask=mask)
        return out.transpose(0, 2, 1, 3)

    def forward(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache: tp.Any = None,
        cache_metadata: PageMetadata | None = None,
        cache_view: PageCacheView | None = None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        """Low-level SDPA dispatch.

        Prefer ``__call__`` from model modules. This method is exposed
        for direct kernel-level access.

        Args:
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            mask: Attention mask or ``"causal"`` sentinel.
            cache: Generic cache argument.
            cache_metadata: Paged attention metadata.
            cache_view: Paged cache view.
            sinks: Optional attention sink tokens.

        Returns:
            Attention output tensor.
        """
        _PAGED_MECHANISMS = {
            "paged": "page_attention",
            "page_attention": "page_attention",
            "unified": "unified_attention",
            "unified_attention": "unified_attention",
        }
        paged_mechanism_kwargs = {}
        if self.attn_mechanism is not None and self.attn_mechanism in _PAGED_MECHANISMS:
            paged_mechanism_kwargs["paged_attention_mechanism"] = _PAGED_MECHANISMS[self.attn_mechanism]
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            scale=self.scale,
            mask=mask,
            sinks=sinks,
            **paged_mechanism_kwargs,
        )


__all__ = ("AttentionPerformer", "PagedAttentionInputs", "prepare_paged_attention_inputs")
