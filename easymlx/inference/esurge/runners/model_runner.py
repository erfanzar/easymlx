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

"""Model runner for eSurge runtime execution steps.

Implements the :class:`ModelRunner`, the primary execution wrapper that
translates scheduler decisions (:class:`ExecutionRequest`) into model
forward passes, token sampling, and sequence-buffer bookkeeping.  It
manages KV-cache row mappings, paged-cache page assignment, batch
padding, warmup precompilation, and performance tracking.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import mlx.core as mx

from easymlx.workers.loggers import get_logger

from ..request import EngineRequest
from ..sampling_params import SamplingParams
from .execution_types import ExecutionRequest, ExecutionResult, ExecutionUpdate, ScheduledSequence
from .sequence_buffer import SequenceBuffer

logger = get_logger("eSurgeRunner")

_CACHE_MAXSIZE = 64
_MLX_NO_PRIMITIVE_MSG = "Attempting to eval an array without a primitive"


class _CompiledPagedCacheProxy:
    """State-backed cache proxy for ``mx.compile`` paged forwards."""

    __slots__ = ("block_size", "head_dim", "num_kv_heads", "page_vec_size", "state")

    def __init__(self, cache: Any):
        self.block_size = int(cache.block_size)
        self.num_kv_heads = int(cache.num_kv_heads)
        self.head_dim = int(cache.head_dim)
        self.page_vec_size = int(getattr(cache, "page_vec_size", 0) or 0)
        self.state = {
            "key_cache": cache.key_cache,
            "value_cache": cache.value_cache,
            "block_tables": cache.block_tables,
            "kv_lens": cache.kv_lens,
            "page_key_cache": getattr(cache, "page_key_cache", None),
            "page_value_cache": getattr(cache, "page_value_cache", None),
        }

    @property
    def key_cache(self):
        return self.state["key_cache"]

    @key_cache.setter
    def key_cache(self, value):
        self.state["key_cache"] = value

    @property
    def value_cache(self):
        return self.state["value_cache"]

    @value_cache.setter
    def value_cache(self, value):
        self.state["value_cache"] = value

    @property
    def block_tables(self):
        return self.state["block_tables"]

    @block_tables.setter
    def block_tables(self, value):
        self.state["block_tables"] = value

    @property
    def kv_lens(self):
        return self.state["kv_lens"]

    @kv_lens.setter
    def kv_lens(self, value):
        self.state["kv_lens"] = value

    @property
    def page_key_cache(self):
        return self.state["page_key_cache"]

    @page_key_cache.setter
    def page_key_cache(self, value):
        self.state["page_key_cache"] = value

    @property
    def page_value_cache(self):
        return self.state["page_value_cache"]

    @page_value_cache.setter
    def page_value_cache(self, value):
        self.state["page_value_cache"] = value

    @property
    def cache(self):
        """Return self for compatibility with code that accesses ``view.cache``."""
        return self

    @property
    def num_seqs(self) -> int:
        return int(self.block_tables.shape[0])


def _call_with_filtered_kwargs(callable_obj: Any, /, **kwargs: Any) -> Any:
    """Invoke *callable_obj* passing only the keyword arguments it accepts.

    If the callable accepts ``**kwargs``, all arguments are forwarded.
    Otherwise, only arguments whose names match formal parameters are
    passed.

    Args:
        callable_obj: The callable to invoke.
        **kwargs: Candidate keyword arguments.

    Returns:
        The return value of the callable.
    """
    signature = inspect.signature(callable_obj)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_kwargs:
        return callable_obj(**kwargs)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return callable_obj(**filtered)


def _as_mx_logits(output: Any) -> mx.array:
    """Extract logits from a model output as a 2-D MLX array."""
    logits = getattr(output, "logits", output)
    if isinstance(logits, dict):
        logits = logits.get("logits", logits)
    if not isinstance(logits, mx.array):
        logits = mx.array(logits)
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim == 1:
        logits = mx.expand_dims(logits, axis=0)
    if logits.ndim != 2:
        raise RuntimeError(f"Unsupported logits shape from runner: {logits.shape}")
    if logits.dtype != mx.float32:
        logits = logits.astype(mx.float32)
    return logits


def _argmax_token_ids(output: Any) -> list[int]:
    """Return greedy token IDs from a raw model output.

    Extracts the final-step logits and performs argmax on-device when
    possible, transferring only the resulting token IDs back to host.

    Args:
        output: Raw model output (object with ``.logits``, dict, or
            array).

    Returns:
        A 1-D list of sampled token IDs.

    Raises:
        RuntimeError: If the resulting logits have an unsupported
            number of dimensions.
    """
    token_ids = mx.argmax(_as_mx_logits(output), axis=-1).astype(mx.int64)
    _safe_eval(token_ids)
    values = token_ids.tolist()
    if isinstance(values, list):
        return [int(token_id) for token_id in values]
    return [int(values)]


def _safe_eval(*arrays: mx.array) -> None:
    """Evaluate arrays, ignoring MLX's no-primitive error for materialized values."""
    filtered = [array for array in arrays if array is not None]
    if not filtered:
        return
    try:
        mx.eval(*filtered)
    except RuntimeError as exc:
        if _MLX_NO_PRIMITIVE_MSG not in str(exc):
            raise


def _apply_temperature_mx(logits: mx.array, temperature: float) -> mx.array:
    """Scale MLX logits by temperature for sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")
    if temperature == 1.0:
        return logits
    return logits / float(temperature)


def _restrict_top_k_mx(logits: mx.array, top_k: int) -> tuple[mx.array, mx.array | None]:
    """Restrict MLX logits to a sorted top-k shortlist when requested."""
    if top_k <= 0:
        return logits, None
    vocab = int(logits.shape[-1])
    if top_k >= vocab:
        return logits, None

    top_indices = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k].astype(mx.int32)
    top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
    order = mx.argsort(top_logits, axis=-1)[:, ::-1]
    top_logits = mx.take_along_axis(top_logits, order, axis=-1)
    top_indices = mx.take_along_axis(top_indices, order, axis=-1).astype(mx.int32)
    return top_logits, top_indices


def _apply_top_p_mx(logits: mx.array, top_p: float, *, assume_sorted: bool = False) -> mx.array:
    """Apply nucleus filtering to MLX logits."""
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits

    if assume_sorted:
        sorted_logits = logits
        sorted_idx = None
    else:
        sorted_idx = mx.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)

    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cum_probs = mx.cumsum(sorted_probs, axis=-1)
    keep_sorted = cum_probs <= float(top_p)
    keep_sorted = mx.concatenate(
        [
            mx.ones((keep_sorted.shape[0], 1), dtype=mx.bool_),
            keep_sorted[:, 1:],
        ],
        axis=-1,
    )

    very_neg = mx.finfo(sorted_logits.dtype).min
    if assume_sorted:
        return mx.where(keep_sorted, sorted_logits, very_neg)

    keep = mx.put_along_axis(mx.zeros_like(keep_sorted), sorted_idx, keep_sorted, axis=-1)
    return mx.where(keep, logits, very_neg)


def _apply_repetition_penalty_row_mx(logits_row: mx.array, token_ids: list[int], repetition_penalty: float) -> mx.array:
    """Apply the standard repetition penalty to one logit row."""
    if repetition_penalty == 1.0 or not token_ids:
        return logits_row
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be > 0")

    unique_token_ids = sorted({int(token_id) for token_id in token_ids})
    if not unique_token_ids:
        return logits_row

    row = mx.expand_dims(logits_row, axis=0)
    token_idx = mx.array([unique_token_ids], dtype=mx.int32)
    selected = mx.take_along_axis(row, token_idx, axis=-1)
    penalized = mx.where(
        selected < 0,
        selected * float(repetition_penalty),
        selected / float(repetition_penalty),
    )
    return mx.put_along_axis(row, token_idx, penalized, axis=-1)[0]


def _apply_presence_penalty_row_mx(logits_row: mx.array, token_ids: list[int], presence_penalty: float) -> mx.array:
    """Apply an additive presence penalty to one logit row."""
    if presence_penalty == 0.0 or not token_ids:
        return logits_row

    unique_token_ids = sorted({int(token_id) for token_id in token_ids})
    if not unique_token_ids:
        return logits_row

    row = mx.expand_dims(logits_row, axis=0)
    token_idx = mx.array([unique_token_ids], dtype=mx.int32)
    selected = mx.take_along_axis(row, token_idx, axis=-1)
    penalized = selected - float(presence_penalty)
    return mx.put_along_axis(row, token_idx, penalized, axis=-1)[0]


class ModelRunner:
    """Row-aware execution wrapper over a model or execution callable.

    Manages the lifecycle of active sequences in a :class:`SequenceBuffer`,
    drives model forward passes with KV-cache page views, performs token
    sampling, and tracks performance metrics.

    Attributes:
        model: The model callable (or ``nn.Module.__call__``).
        sequence_buffer: Row-oriented mutable buffer of active sequences.
        kv_caches: List of KV-cache objects (typically
            :class:`PageCacheView` instances).
        max_model_len: Maximum supported sequence length.
        max_num_batched_tokens: Maximum tokens per forward pass.
        max_num_seqs: Maximum concurrent sequences.
        max_num_seq_buckets: Sorted tuple of sequence-count bucket sizes
            for Metal kernel padding.
        memory_utilization: Target memory utilization fraction.
        supports_async_overlap: Whether the runner supports overlapping
            async execution.
    """

    def __init__(
        self,
        model: Any,
        *,
        sequence_buffer: SequenceBuffer | None = None,
        kv_caches: list[Any] | None = None,
        max_num_seq_buckets: tuple[int, ...] | None = None,
        max_model_len: int = 8192,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int = 4,
        memory_utilization: float = 0.85,
        supports_async_overlap: bool = False,
        verbose: bool = False,
        seed: int = 0,
        use_compiled_forward: bool = True,
    ):
        """Initialize the model runner.

        Args:
            model: Model callable that accepts ``input_ids`` and
                optional ``cache_views`` / ``cache_metadata`` keyword
                arguments.
            sequence_buffer: Pre-existing sequence buffer.  A new one
                is created if ``None``.
            kv_caches: List of KV-cache objects to use for paged
                attention.
            max_num_seq_buckets: Explicit sequence-count bucket sizes.
                Derived automatically when ``None``.
            max_model_len: Maximum supported context length.
            max_num_batched_tokens: Token budget per forward pass.
                Defaults to ``max_model_len``.
            max_num_seqs: Maximum concurrent sequences.
            memory_utilization: Target fraction of available memory.
            supports_async_overlap: Whether overlapping async execution
                is supported.
            verbose: Log at INFO instead of DEBUG level.
            seed: Random seed for the sampling RNG.
        """
        self.model = model
        self.sequence_buffer = sequence_buffer if sequence_buffer is not None else SequenceBuffer()
        self.kv_caches = list(kv_caches or [])
        self.max_model_len = int(max_model_len)
        self.max_num_batched_tokens = int(max_num_batched_tokens) if max_num_batched_tokens else self.max_model_len
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_seq_buckets = self._init_seq_buckets(max_num_seq_buckets, self.max_num_seqs)
        self.memory_utilization = float(memory_utilization)
        self.supports_async_overlap = bool(supports_async_overlap)
        self.use_compiled_forward = bool(use_compiled_forward)
        self._sample_key = mx.random.key(seed)
        self._page_size = self._infer_page_size()
        self.log_it = logger.info if verbose else logger.debug
        self._model_signature = inspect.signature(self.model)
        self._model_param_names = frozenset(self._model_signature.parameters)
        self._model_accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in self._model_signature.parameters.values()
        )
        self._decode_step_fn = getattr(self.model, "decode_step", None)
        self._decode_step_signature = inspect.signature(self._decode_step_fn) if callable(self._decode_step_fn) else None
        self._decode_step_param_names = (
            frozenset(self._decode_step_signature.parameters) if self._decode_step_signature is not None else frozenset()
        )
        self._decode_step_accepts_kwargs = bool(
            self._decode_step_signature is not None
            and any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in self._decode_step_signature.parameters.values()
            )
        )
        self._compiled_forward_blocked = bool(
            self._model_accepts_kwargs
            or {"positions", "slot_ids", "query_lens", "multimodal", "multimodal_inputs"}.intersection(
                self._model_param_names
            )
        )
        self._cache_views = list(self.kv_caches)
        self._page_cache_view_cls: type[Any] | None = None
        self._page_metadata_cls: type[Any] | None = None
        self._build_query_start_loc_fn: Any | None = None
        self._query_start_loc_cache: dict[tuple[int, ...], Any] = {}
        self._single_token_decode_metadata: dict[int, Any] = {}

        logger.debug(
            "Initializing ModelRunner with max_model_len=%d, max_num_seqs=%d",
            self.max_model_len,
            self.max_num_seqs,
        )
        logger.debug(
            "Configuration: memory_utilization=%.2f, page_size=%d",
            self.memory_utilization,
            self._page_size,
        )
        self.log_it(
            "ModelRunner initialized: model=%s, buckets=%s, page_size=%d",
            type(model).__name__,
            self.max_num_seq_buckets or "none",
            self._page_size,
        )
        self.log_it(
            "Sequence buffer: max_num_rows=%d",
            self.sequence_buffer.max_num_rows,
        )
        self._perf_iteration = 0
        self._perf_tps_ema = 0.0
        self._compiled_cache: dict[tuple[int, int], Any] = {}
        self._compiled_forwards: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
        self._compiled_samplers: dict[tuple[tuple[int, ...], bool, float, int, float], Any] = {}
        self._compiled_batch_padders: dict[tuple[int, int], Any] = {}
        self._failed_compiled_forward_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        logger.debug("ModelRunner initialization complete")

    def _ensure_paged_helpers(self) -> tuple[type[Any], type[Any], Any]:
        """Resolve paged-cache helper types/functions once and cache them."""
        if (
            self._page_cache_view_cls is None
            or self._page_metadata_cls is None
            or self._build_query_start_loc_fn is None
        ):
            from easymlx.caching import PageCacheView, PageMetadata, build_query_start_loc

            self._page_cache_view_cls = PageCacheView
            self._page_metadata_cls = PageMetadata
            self._build_query_start_loc_fn = build_query_start_loc
        assert self._page_cache_view_cls is not None
        assert self._page_metadata_cls is not None
        assert self._build_query_start_loc_fn is not None
        return self._page_cache_view_cls, self._page_metadata_cls, self._build_query_start_loc_fn

    def _filter_model_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter model kwargs using the cached model signature."""
        if self._model_accepts_kwargs:
            return kwargs
        return {name: value for name, value in kwargs.items() if name in self._model_param_names}

    def _call_model(self, **kwargs: Any) -> Any:
        """Invoke the model using cached signature filtering."""
        return self.model(**self._filter_model_kwargs(kwargs))

    def _call_decode_step(self, **kwargs: Any) -> mx.array:
        """Invoke the model's fixed-signature decode entrypoint."""
        if not callable(self._decode_step_fn):
            raise AttributeError("Model does not expose decode_step().")
        if self._decode_step_accepts_kwargs:
            return self._decode_step_fn(**kwargs)
        return self._decode_step_fn(
            **{name: value for name, value in kwargs.items() if name in self._decode_step_param_names}
        )

    def _can_use_decode_step(self, query_lens: list[int], *, has_extra: bool) -> bool:
        """Return whether this runner can use the model's decode fast path."""
        return bool(
            callable(self._decode_step_fn)
            and not has_extra
            and query_lens
            and all(int(query_len) == 1 for query_len in query_lens)
        )

    def _query_start_loc_for(self, query_lens: tuple[int, ...]) -> Any:
        """Return cached cumulative query offsets for a query-length tuple."""
        cached = self._query_start_loc_cache.get(query_lens)
        if cached is not None:
            return cached
        _, _, build_query_start_loc = self._ensure_paged_helpers()
        cached = build_query_start_loc(query_lens)
        if len(self._query_start_loc_cache) >= _CACHE_MAXSIZE:
            oldest_key = next(iter(self._query_start_loc_cache))
            del self._query_start_loc_cache[oldest_key]
        self._query_start_loc_cache[query_lens] = cached
        return cached

    def _page_metadata_for(self, query_lens: tuple[int, ...], slot_ids: tuple[int, ...]) -> Any:
        """Build or reuse paged-attention metadata for the given decode shape."""
        _, page_metadata_cls, _ = self._ensure_paged_helpers()
        is_single_token_decode = all(int(query_len) == 1 for query_len in query_lens)
        if is_single_token_decode and len(slot_ids) == 1:
            slot_id = int(slot_ids[0])
            cached = self._single_token_decode_metadata.get(slot_id)
            if cached is not None:
                return cached
            cached = page_metadata_cls(
                query_start_loc=self._query_start_loc_for(query_lens),
                is_single_token_decode=True,
                slot_ids=slot_ids,
            )
            if len(self._single_token_decode_metadata) >= _CACHE_MAXSIZE:
                oldest_key = next(iter(self._single_token_decode_metadata))
                del self._single_token_decode_metadata[oldest_key]
            self._single_token_decode_metadata[slot_id] = cached
            return cached
        return page_metadata_cls(
            query_start_loc=self._query_start_loc_for(query_lens),
            is_single_token_decode=is_single_token_decode,
            slot_ids=slot_ids,
        )

    def _get_token_buckets(self) -> list[int]:
        """Generate token-length buckets: powers of 2 from 1 up to max_num_batched_tokens.

        Returns:
            Sorted list of integer bucket sizes.
        """
        cap = self.max_num_batched_tokens
        buckets = [1]
        v = 2
        while v <= cap:
            buckets.append(v)
            v *= 2
        if buckets[-1] != cap:
            buckets.append(cap)
        return buckets

    @staticmethod
    def _get_feasible_compile_pairs(
        token_buckets: list[int],
        req_buckets: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        """Return only ``(num_tokens, num_reqs)`` pairs where ``num_reqs <= num_tokens``.

        Args:
            token_buckets: Available token-count buckets.
            req_buckets: Available request-count buckets.

        Returns:
            List of feasible ``(num_tokens, num_reqs)`` pairs.
        """
        pairs: list[tuple[int, int]] = []
        for num_tokens in token_buckets:
            for num_reqs in req_buckets:
                if num_reqs <= num_tokens:
                    pairs.append((num_tokens, num_reqs))
        return pairs

    def precompile(self) -> None:
        """Warmup the model forward for all feasible bucket shape pairs.

        For each ``(num_tokens, num_reqs)`` pair, runs a dummy forward pass so that
        MLX traces and caches the Metal kernels. Subsequent calls with matching
        shapes avoid the first-call tracing overhead.
        """
        if not self.kv_caches:
            self.log_it("Skipping precompilation: no KV caches configured")
            return

        token_buckets = self._get_token_buckets()
        req_buckets = self.max_num_seq_buckets
        pairs = self._get_feasible_compile_pairs(token_buckets, req_buckets)

        page_cache_view_cls, page_metadata_cls, _ = self._ensure_paged_helpers()
        from easymlx.workers.loggers import ProgressLogger

        if not all(isinstance(c, page_cache_view_cls) for c in self.kv_caches):
            self.log_it("Skipping warmup: KV caches are not PageCacheView instances")
            return

        max_slots = int(self.kv_caches[0].num_seqs) if self.kv_caches else self.max_num_seqs

        progress = ProgressLogger(logger.name, logger_instance=logger)
        total = len(pairs)

        for idx, (num_tokens, num_reqs) in enumerate(pairs):
            num_reqs = min(num_reqs, max_slots)
            key = (num_tokens, num_reqs)
            if key in self._compiled_cache:
                progress.update(idx + 1, total, f"skipped ({num_tokens} tok, {num_reqs} req)")
                continue

            progress.update(idx, total, f"tracing ({num_tokens} tok, {num_reqs} req)")

            dummy_ids = mx.zeros((num_tokens,), dtype=mx.int32)
            slot_ids = list(range(num_reqs))
            query_lens = self._distribute_tokens(num_tokens, num_reqs)
            cache_views = self._cache_views
            cache_metadata = page_metadata_cls(
                query_start_loc=self._query_start_loc_for(tuple(query_lens)),
                is_single_token_decode=all(int(query_len) == 1 for query_len in query_lens),
                slot_ids=tuple(slot_ids),
            )

            try:
                compiled_entry = self._get_compiled_forward(slot_ids, query_lens)
                if compiled_entry is not None:
                    self._refresh_compiled_cache_state(compiled_entry, self.kv_caches)
                    logits = compiled_entry["fn"](dummy_ids)
                    _safe_eval(logits)
                    self._sync_compiled_cache_state(compiled_entry, self.kv_caches)
                else:
                    out = self._call_model(
                        input_ids=dummy_ids,
                        cache_views=cache_views,
                        cache_metadata=cache_metadata,
                        return_dict=True,
                    )
                    logits = getattr(out, "logits", out)
                _safe_eval(logits)
                self._compiled_cache[key] = True
            except Exception as exc:
                compile_key = (
                    tuple(int(slot_id) for slot_id in slot_ids),
                    tuple(int(query_len) for query_len in query_lens),
                )
                self._compiled_forwards.pop(compile_key, None)
                self._failed_compiled_forward_keys.add(compile_key)
                logger.warning("Warmup failed for (%d, %d): %s", num_tokens, num_reqs, exc)

            for cache in self.kv_caches:
                kv_lens = getattr(cache, "kv_lens", None)
                if kv_lens is not None:
                    for s in slot_ids:
                        kv_lens[s] = 0

        progress.complete(f"Warmup done: {len(self._compiled_cache)} shapes traced")

    @staticmethod
    def _distribute_tokens(num_tokens: int, num_reqs: int) -> list[int]:
        """Distribute *num_tokens* across *num_reqs* sequences for dummy warmup.

        Args:
            num_tokens: Total token count to distribute.
            num_reqs: Number of sequences.

        Returns:
            List of per-sequence token counts that sum to *num_tokens*.
        """
        base = num_tokens // num_reqs
        remainder = num_tokens % num_reqs
        return [base + (1 if i < remainder else 0) for i in range(num_reqs)]

    def _get_compiled_forward(
        self,
        slot_ids: list[int],
        query_lens: list[int],
    ) -> Any:
        """Return an ``mx.compile``'d forward with cache proxy captured in closure.

        The cache proxy is captured in the closure (not passed as arg) so
        ``mx.compile`` only sees ``mx.array`` inputs. We cache compiled fns
        by ``(slot_ids, query_lens)`` since ``cache_metadata`` differs.
        The underlying PageCacheView arrays are shared and mutated in-place.

        Args:
            slot_ids: Cache slot indices for this batch.
            query_lens: Per-sequence query lengths.

        Returns:
            A compiled callable, or ``None`` if KV caches are not
            configured.
        """
        if not self.kv_caches:
            return None
        if not self.use_compiled_forward:
            return None
        if any(getattr(cache, "cache_dtype_is_turboquant", False) for cache in self.kv_caches):
            return None

        page_cache_view_cls, page_metadata_cls, _ = self._ensure_paged_helpers()

        if not all(isinstance(c, page_cache_view_cls) for c in self.kv_caches):
            return None
        if not query_lens or any(int(query_len) != 1 for query_len in query_lens):
            return None
        if self._compiled_forward_blocked:
            return None

        key = (tuple(int(slot_id) for slot_id in slot_ids), tuple(int(query_len) for query_len in query_lens))
        if key in self._failed_compiled_forward_keys:
            return None
        compiled_entry = self._compiled_forwards.get(key)
        if compiled_entry is not None:
            return compiled_entry

        cache_proxies = [_CompiledPagedCacheProxy(kv_cache) for kv_cache in self.kv_caches]
        cache_views = list(cache_proxies)
        cache_metadata = page_metadata_cls(
            query_start_loc=self._query_start_loc_for(tuple(int(query_len) for query_len in query_lens)),
            is_single_token_decode=all(int(query_len) == 1 for query_len in query_lens),
            slot_ids=tuple(int(s) for s in slot_ids),
        )

        static_kwargs = self._filter_model_kwargs(
            {
                "cache_views": cache_views,
                "cache_metadata": cache_metadata,
                "return_dict": False,
            }
        )
        state_arrays = [cache_proxy.state for cache_proxy in cache_proxies]

        def compiled_forward(input_ids: mx.array) -> mx.array:
            output = self.model(input_ids=input_ids, **static_kwargs)
            return getattr(output, "logits", output)

        compiled_fn = mx.compile(compiled_forward, inputs=state_arrays, outputs=state_arrays)
        compiled_entry = {"fn": compiled_fn, "state": state_arrays, "proxies": cache_proxies}
        self._compiled_forwards[key] = compiled_entry
        return compiled_entry

    @staticmethod
    def _refresh_compiled_cache_state(
        compiled_entry: dict[str, Any],
        kv_caches: list[Any],
    ) -> None:
        """Refresh proxy state from the live caches before a compiled call."""
        for cache_proxy, cache in zip(compiled_entry["proxies"], kv_caches, strict=False):
            cache_proxy.key_cache = cache.key_cache
            cache_proxy.value_cache = cache.value_cache
            cache_proxy.block_tables = cache.block_tables
            cache_proxy.kv_lens = cache.kv_lens
            if hasattr(cache, "page_key_cache"):
                cache_proxy.page_key_cache = cache.page_key_cache
            if hasattr(cache, "page_value_cache"):
                cache_proxy.page_value_cache = cache.page_value_cache

    @staticmethod
    def _sync_compiled_cache_state(
        compiled_entry: dict[str, Any],
        kv_caches: list[Any],
    ) -> None:
        """Propagate compiled proxy state back into the live caches."""
        for cache_proxy, cache in zip(compiled_entry["proxies"], kv_caches, strict=False):
            cache.key_cache = cache_proxy.key_cache
            cache.value_cache = cache_proxy.value_cache
            cache.block_tables = cache_proxy.block_tables
            cache.kv_lens = cache_proxy.kv_lens
            if hasattr(cache, "page_key_cache"):
                cache.page_key_cache = cache_proxy.page_key_cache
            if hasattr(cache, "page_value_cache"):
                cache.page_value_cache = cache_proxy.page_value_cache

    def _infer_page_size(self) -> int:
        """Infer the KV-cache page size from the first cache object.

        Returns:
            The page (block) size, defaulting to ``16`` if no caches
            are configured.
        """
        if not self.kv_caches:
            return 16
        first_cache = self.kv_caches[0]
        return max(int(getattr(first_cache, "block_size", 16) or 16), 1)

    def pages_required(self, token_count: int) -> int:
        """Calculate the number of pages needed for *token_count* tokens.

        Args:
            token_count: Number of tokens to store.

        Returns:
            Number of pages (ceiling division by page size).
        """
        tokens = max(int(token_count), 0)
        if tokens == 0:
            return 0
        return (tokens + self._page_size - 1) // self._page_size

    def _init_seq_buckets(
        self,
        user_buckets: tuple[int, ...] | None,
        max_num_seqs: int,
    ) -> tuple[int, ...]:
        """Initialize the sequence-count bucket tuple.

        Args:
            user_buckets: Explicit bucket sizes, or ``None`` for
                automatic derivation.
            max_num_seqs: Maximum concurrent sequences.

        Returns:
            Sorted tuple of bucket sizes ending with *max_num_seqs*.
        """
        if user_buckets:
            buckets = sorted({int(b) for b in user_buckets if 0 < int(b) <= max_num_seqs})
        else:
            buckets = self._get_request_paddings(4, max_num_seqs)
        if not buckets or buckets[-1] != max_num_seqs:
            buckets.append(max_num_seqs)
        return tuple(buckets)

    @staticmethod
    def _get_request_paddings(min_bucket: int, max_bucket: int) -> list[int]:
        """Generate powers-of-two padding sizes between *min_bucket* and *max_bucket*.

        Args:
            min_bucket: Minimum bucket size (clamped to >= 4).
            max_bucket: Maximum bucket size.

        Returns:
            Sorted list of bucket sizes.
        """
        min_bucket = max(4, min(min_bucket, max_bucket))
        buckets: list[int] = []
        current = min_bucket
        while current < max_bucket:
            buckets.append(current)
            current *= 2
        if not buckets or buckets[-1] != max_bucket:
            buckets.append(max_bucket)
        return buckets

    def select_bucket_size(self, batch_size: int) -> int:
        """Select the smallest bucket that fits *batch_size*.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            The selected bucket size.
        """
        size = max(int(batch_size), 0)
        if size == 0:
            return self.max_num_seq_buckets[0]
        for bucket in self.max_num_seq_buckets:
            if size <= bucket:
                return bucket
        return self.max_num_seq_buckets[-1]

    def _clear_row_mapping(self, row_index: int) -> None:
        """Reset the KV-cache block table and length for *row_index*.

        Args:
            row_index: The row to clear.
        """
        if not self.kv_caches:
            return
        for cache in self.kv_caches:
            block_tables = getattr(cache, "block_tables", None)
            kv_lens = getattr(cache, "kv_lens", None)
            if block_tables is not None and row_index < block_tables.shape[0]:
                block_tables[row_index, :] = -1
            if kv_lens is not None and row_index < kv_lens.shape[0]:
                kv_lens[row_index] = 0
        _safe_eval(*[c.block_tables for c in self.kv_caches if hasattr(c, "block_tables")])

    @staticmethod
    def _page_ids_array(page_ids: list[int]) -> mx.array:
        """Convert page ids into an MLX int32 vector."""
        return mx.array(page_ids, dtype=mx.int32)

    @staticmethod
    def _copy_cache_row(table: mx.array, row_index: int) -> mx.array:
        """Return a detached MLX copy of one cache-table row."""
        return mx.take(table, mx.array([int(row_index)], dtype=mx.int32), axis=0)[0]

    def _zero_pages(self, page_ids: list["int"]) -> None:
        """Zero out the key and value caches for the given page IDs.

        Args:
            page_ids: List of page indices to zero.
        """
        if not page_ids or not self.kv_caches:
            return
        valid_ids = [int(p) for p in page_ids if int(p) >= 0]
        if not valid_ids:
            return
        mx.array(valid_ids, dtype=mx.int32)
        touched_arrays: list[mx.array] = []
        for cache in self.kv_caches:
            key_cache = getattr(cache, "key_cache", None)
            value_cache = getattr(cache, "value_cache", None)
            if key_cache is None or value_cache is None:
                continue
            cap = int(key_cache.shape[0])
            filtered = mx.array([p for p in valid_ids if p < cap], dtype=mx.int32)
            if filtered.size == 0:
                continue
            key_cache[filtered] = 0
            value_cache[filtered] = 0
            touched_arrays.extend([key_cache, value_cache])
            page_key_cache = getattr(cache, "page_key_cache", None)
            page_value_cache = getattr(cache, "page_value_cache", None)
            if page_key_cache is not None:
                page_key_cache[filtered] = 0
                touched_arrays.append(page_key_cache)
            if page_value_cache is not None:
                page_value_cache[filtered] = 0
                touched_arrays.append(page_value_cache)
            for attr_name in ("key_norms", "key_residual_norms", "key_qjl_signs", "value_norms"):
                aux_array = getattr(cache, attr_name, None)
                if aux_array is not None:
                    aux_array[filtered] = 0
                    touched_arrays.append(aux_array)
        if touched_arrays:
            _safe_eval(*touched_arrays)

    def _assign_row_mapping(
        self,
        row_index: int,
        page_ids: list["int"],
        *,
        cached_tokens: int = 0,
    ) -> None:
        """Write page IDs into the KV-cache block table for a row.

        Pages beyond the cached-token boundary are zeroed to prevent
        stale KV data from leaking.

        Args:
            row_index: Target row in the block table.
            page_ids: Page IDs to assign.
            cached_tokens: Number of tokens already stored in the
                assigned pages.

        Raises:
            RuntimeError: If *row_index* exceeds the cache capacity.
        """
        if not self.kv_caches:
            return
        keep_pages = self.pages_required(cached_tokens)
        if keep_pages < len(page_ids):
            self._zero_pages(page_ids[keep_pages:])
        for cache in self.kv_caches:
            block_tables = getattr(cache, "block_tables", None)
            kv_lens = getattr(cache, "kv_lens", None)
            if block_tables is None or kv_lens is None:
                continue
            if row_index >= block_tables.shape[0]:
                raise RuntimeError(f"Row index {row_index} exceeds cache capacity {block_tables.shape[0]}")
            block_tables[row_index, :] = -1
            count = min(len(page_ids), block_tables.shape[1])
            if count:
                block_tables[row_index, :count] = self._page_ids_array(page_ids[:count])
            kv_lens[row_index] = max(int(cached_tokens), 0)

    def bind_request(self, request: EngineRequest, row_index: int) -> None:
        """Bind an engine request to a row in the sequence buffer and KV cache.

        If the request already exists in the buffer but at a different
        row, the cache row is moved accordingly.

        Args:
            request: The :class:`EngineRequest` to bind.
            row_index: Target row index in the sequence buffer.
        """
        page_ids = list(request.cache_state.page_ids)
        cached_tokens = min(request.num_cached_tokens, request.total_tokens)
        metadata = dict(request.metadata)
        if self.sequence_buffer.has_request(request.request_id):
            current_index = self.sequence_buffer.get_row_index(request.request_id)
            if current_index != row_index:
                self._move_cache_row(current_index, row_index)
                self.sequence_buffer.move_row(current_index, row_index)
            row = self.sequence_buffer.get_row(request.request_id)
            row.prompt_token_ids = list(request.prompt_token_ids)
            row.output_token_ids = list(request.generated_token_ids)
            row.num_computed_tokens = int(request.num_computed_tokens)
            row.metadata = metadata
            row.page_ids = page_ids
        else:
            self.sequence_buffer.begin_sequence(
                request.request_id,
                request.prompt_token_ids,
                row_index=row_index,
                page_ids=page_ids,
                num_computed_tokens=request.num_computed_tokens,
                metadata=metadata,
            )
            if request.generated_token_ids:
                self.sequence_buffer.append_output_tokens(request.request_id, request.generated_token_ids)
        self._assign_row_mapping(row_index, page_ids, cached_tokens=cached_tokens)

    def detach_request(self, request_id: str) -> int | None:
        """Detach a request from the sequence buffer and clear its cache row.

        Also clears the failed-compiled-forward key set so that
        previously failed compilation keys can be retried for new
        requests.

        Args:
            request_id: The request to detach.

        Returns:
            The row index that was freed, or ``None`` if the request
            was not tracked.
        """
        if not self.sequence_buffer.has_request(request_id):
            return None
        row_index = self.sequence_buffer.get_row_index(request_id)
        self.sequence_buffer.remove_sequence(request_id)
        self._clear_row_mapping(row_index)
        self._failed_compiled_forward_keys.clear()
        return row_index

    def _move_cache_row(self, from_index: int, to_index: int) -> None:
        """Swap the KV-cache block-table and kv-len entries between two rows.

        Args:
            from_index: Source row index.
            to_index: Destination row index.
        """
        if from_index == to_index or not self.kv_caches:
            return
        for cache in self.kv_caches:
            block_tables = getattr(cache, "block_tables", None)
            kv_lens = getattr(cache, "kv_lens", None)
            if block_tables is None or kv_lens is None:
                continue
            from_table = self._copy_cache_row(block_tables, from_index)
            from_kv_len = int(kv_lens[from_index])
            to_table = self._copy_cache_row(block_tables, to_index)
            to_kv_len = int(kv_lens[to_index])
            block_tables[to_index] = from_table
            kv_lens[to_index] = from_kv_len
            block_tables[from_index] = to_table
            kv_lens[from_index] = to_kv_len

    def compact_rows(self) -> list[tuple[int, int]]:
        """Compact holes in the sequence buffer and synchronize the KV cache.

        Shifts rows left to eliminate gaps, then trims trailing holes.

        Returns:
            List of ``(from_index, to_index)`` moves that were
            performed.
        """
        moves = self.sequence_buffer.compact_holes()
        if not moves:
            self.sequence_buffer.trim_trailing_holes()
            return []
        for from_index, to_index in moves:
            if from_index == to_index or not self.kv_caches:
                continue
            for cache in self.kv_caches:
                block_tables = getattr(cache, "block_tables", None)
                kv_lens = getattr(cache, "kv_lens", None)
                if block_tables is None or kv_lens is None:
                    continue
                block_tables[to_index] = self._copy_cache_row(block_tables, from_index)
                kv_lens[to_index] = int(kv_lens[from_index])
                block_tables[from_index, :] = -1
                kv_lens[from_index] = 0
        self.sequence_buffer.trim_trailing_holes()
        return moves

    @staticmethod
    def _flatten_sequence_tokens(sequences: list[ScheduledSequence]) -> tuple[mx.array, list[int]]:
        """Flatten token IDs for a scheduled batch into one MLX vector."""
        query_lens = [len(sequence.token_ids) for sequence in sequences]
        flat_token_ids = [int(token_id) for sequence in sequences for token_id in sequence.token_ids]
        return mx.array(flat_token_ids, dtype=mx.int32), query_lens

    def _get_compiled_batch_padder(self, batch_size: int, max_query_len: int) -> Any:
        """Return an ``mx.compile``'d helper that pads flattened token IDs."""
        key = (int(batch_size), int(max_query_len))
        cached = self._compiled_batch_padders.get(key)
        if cached is not None:
            return cached

        def compiled_padder(flat_token_ids: mx.array, query_lens: mx.array) -> mx.array:
            positions = mx.arange(max_query_len, dtype=mx.int32)[None, :]
            query_lens = query_lens.astype(mx.int32)
            row_offsets = mx.cumsum(query_lens, axis=0).astype(mx.int32) - query_lens
            mask = positions < query_lens[:, None]
            safe_size = max(int(flat_token_ids.shape[0]), 1)
            safe_indices = mx.minimum(row_offsets[:, None] + positions, safe_size - 1)
            gathered = mx.take(flat_token_ids, safe_indices.reshape((-1,)), axis=0).reshape((batch_size, max_query_len))
            return mx.where(mask, gathered, mx.zeros((batch_size, max_query_len), dtype=mx.int32))

        cached = mx.compile(compiled_padder)
        self._compiled_batch_padders[key] = cached
        return cached

    def _pad_batch_token_ids(self, sequences: list[ScheduledSequence]) -> tuple[mx.array, list["int"]]:
        """Pad variable-length token ID sequences into a rank-2 MLX array.

        Args:
            sequences: Scheduled sequences for this step.

        Returns:
            A ``(batch_array, query_lens)`` tuple where *batch_array*
            has shape ``(num_seqs, max_query_len)`` and *query_lens*
            lists per-sequence lengths.
        """
        flat_token_ids, query_lens = self._flatten_sequence_tokens(sequences)
        max_query_len = max(query_lens, default=0)
        if not sequences or max_query_len == 0:
            return mx.zeros((len(sequences), max_query_len), dtype=mx.int32), query_lens
        compiled_padder = self._get_compiled_batch_padder(len(sequences), max_query_len)
        query_lens_array = mx.array(query_lens, dtype=mx.int32)
        batch = compiled_padder(flat_token_ids, query_lens_array)
        return batch, query_lens

    @staticmethod
    def _build_positions(sequences: list[ScheduledSequence]) -> list[list["int"]]:
        """Build per-sequence position ID lists.

        Args:
            sequences: Scheduled sequences for this step.

        Returns:
            List of per-sequence position ID lists, starting from each
            sequence's ``num_computed_tokens``.
        """
        positions: list[list["int"]] = []
        for sequence in sequences:
            start = int(sequence.num_computed_tokens)
            positions.append(list(range(start, start + len(sequence.token_ids))))
        return positions

    def _forward_step_raw(self, request: ExecutionRequest) -> Any:
        """Execute a single model forward pass and return the raw output."""
        if len(request.sequences) == 1:
            sequence = request.sequences[0]
            query_len = len(sequence.token_ids)
            query_lens = [query_len]
            slot_ids = [int(sequence.row_index)]
            positions = [list(range(int(sequence.num_computed_tokens), int(sequence.num_computed_tokens) + query_len))]
            multimodal = request.multimodal
            extra = dict(request.extra)
            extra.pop("kv_caches", None)
            has_extra = multimodal is not None or extra

            logger.debug(
                "Forward step: batch_size=1, query_lens=%s, paged_cache=%s",
                query_lens,
                bool(self.kv_caches),
            )
            use_decode_step = False

            if self.kv_caches:
                batch_token_ids = mx.array(sequence.token_ids, dtype=mx.int32)
                use_decode_step = self._can_use_decode_step(query_lens, has_extra=has_extra)

                if not has_extra:
                    compiled_entry = self._get_compiled_forward(slot_ids, query_lens)
                    if compiled_entry is not None:
                        try:
                            self._refresh_compiled_cache_state(compiled_entry, self.kv_caches)
                            compiled_fn = compiled_entry["fn"]
                            logits = compiled_fn(batch_token_ids)
                            _safe_eval(logits)
                            self._sync_compiled_cache_state(compiled_entry, self.kv_caches)
                            raw_output = {"logits": logits}
                            return raw_output
                        except (RuntimeError, ValueError):
                            key = (
                                tuple(int(slot_id) for slot_id in slot_ids),
                                tuple(int(query_len) for query_len in query_lens),
                            )
                            self._compiled_forwards.pop(key, None)
                            self._failed_compiled_forward_keys.add(key)
                            logger.debug("Compiled forward failed, falling back to eager")

                cache_views = self._cache_views
                cache_metadata = self._page_metadata_for(
                    tuple(int(query_len) for query_len in query_lens),
                    tuple(int(slot_id) for slot_id in slot_ids),
                )
            else:
                batch_token_ids = mx.array([sequence.token_ids], dtype=mx.int32)
                cache_views = None
                cache_metadata = None

            if use_decode_step:
                raw_output = self._call_decode_step(
                    input_ids=batch_token_ids,
                    cache_views=cache_views,
                    cache_metadata=cache_metadata,
                )
            else:
                raw_output = self._call_model(
                    input_ids=batch_token_ids,
                    cache_views=cache_views,
                    cache_metadata=cache_metadata,
                    query_lens=query_lens,
                    positions=positions,
                    slot_ids=slot_ids,
                    return_dict=True,
                    multimodal_inputs=multimodal,
                    multimodal=multimodal,
                    **extra,
                )
            return raw_output

        batch_token_ids, query_lens = self._pad_batch_token_ids(request.sequences)
        logger.debug(
            "Forward step: batch_size=%d, query_lens=%s, paged_cache=%s",
            len(request.sequences),
            query_lens,
            bool(self.kv_caches),
        )
        slot_ids = [int(sequence.row_index) for sequence in request.sequences]
        positions = self._build_positions(request.sequences)
        multimodal = request.multimodal
        extra = dict(request.extra)
        extra.pop("kv_caches", None)

        has_extra = multimodal is not None or extra
        use_decode_step = False

        if self.kv_caches:
            batch_token_ids, query_lens = self._flatten_sequence_tokens(request.sequences)
            use_decode_step = self._can_use_decode_step(query_lens, has_extra=has_extra)

            if not has_extra:
                compiled_entry = self._get_compiled_forward(slot_ids, query_lens)
                if compiled_entry is not None:
                    try:
                        self._refresh_compiled_cache_state(compiled_entry, self.kv_caches)
                        compiled_fn = compiled_entry["fn"]
                        logits = compiled_fn(batch_token_ids)
                        _safe_eval(logits)
                        self._sync_compiled_cache_state(compiled_entry, self.kv_caches)
                        raw_output = {"logits": logits}
                        return raw_output
                    except (RuntimeError, ValueError):
                        key = (
                            tuple(int(slot_id) for slot_id in slot_ids),
                            tuple(int(query_len) for query_len in query_lens),
                        )
                        self._compiled_forwards.pop(key, None)
                        self._failed_compiled_forward_keys.add(key)
                        logger.debug("Compiled forward failed, falling back to eager")

            cache_views = self._cache_views
            cache_metadata = self._page_metadata_for(
                tuple(int(query_len) for query_len in query_lens),
                tuple(int(slot_id) for slot_id in slot_ids),
            )
        else:
            batch_token_ids, query_lens = self._pad_batch_token_ids(request.sequences)
            cache_views = None
            cache_metadata = None

        if use_decode_step:
            raw_output = self._call_decode_step(
                input_ids=batch_token_ids,
                cache_views=cache_views,
                cache_metadata=cache_metadata,
            )
        else:
            raw_output = self._call_model(
                input_ids=batch_token_ids,
                cache_views=cache_views,
                cache_metadata=cache_metadata,
                query_lens=query_lens,
                positions=positions,
                slot_ids=slot_ids,
                return_dict=True,
                multimodal_inputs=multimodal,
                multimodal=multimodal,
                **extra,
            )
        return raw_output

    def _forward_step(self, request: ExecutionRequest) -> tuple[Any, mx.array]:
        """Execute a single model forward pass for the given execution request.

        Constructs the input tensors, cache views, and metadata, then
        invokes the model callable.

        Args:
            request: The :class:`ExecutionRequest` describing this step.

        Returns:
            A ``(raw_output, logits)`` tuple where *logits* is a 2-D
            MLX array of shape ``(batch_size, vocab_size)``.
        """
        raw_output = self._forward_step_raw(request)
        return raw_output, _as_mx_logits(raw_output)

    def _apply_updates_to_sequence_buffer(self, updates: list[ExecutionUpdate]) -> None:
        """Propagate execution updates into the sequence buffer.

        Args:
            updates: Per-request updates from a runner step.
        """
        for update in updates:
            if not self.sequence_buffer.has_request(update.request_id):
                continue
            self.sequence_buffer.append_output_tokens(update.request_id, update.sampled_token_ids)
            if update.num_computed_tokens is not None:
                self.sequence_buffer.set_num_computed_tokens(update.request_id, update.num_computed_tokens)

    def _sampling_histories_for_sequences(
        self,
        sequences: list[ScheduledSequence],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Return full-token and generated-token histories for scheduled rows."""
        token_histories: list[list[int]] = []
        generated_token_histories: list[list[int]] = []
        for sequence in sequences:
            if self.sequence_buffer.has_request(sequence.request_id):
                row = self.sequence_buffer.get_row(sequence.request_id)
                token_histories.append(row.all_token_ids)
                generated_token_histories.append(list(row.output_token_ids))
            else:
                token_histories.append(list(sequence.token_ids))
                generated_token_histories.append([])
        return token_histories, generated_token_histories

    def _apply_penalties_mx(
        self,
        logits: mx.array,
        sampling_params_by_row: list[SamplingParams | None],
        *,
        token_histories: list[list[int]] | None = None,
        generated_token_histories: list[list[int]] | None = None,
    ) -> mx.array:
        """Apply presence/repetition penalties before sampling."""
        if token_histories is None:
            token_histories = [[] for _ in sampling_params_by_row]
        if generated_token_histories is None:
            generated_token_histories = [[] for _ in sampling_params_by_row]
        if len(token_histories) != len(sampling_params_by_row):
            raise ValueError("token histories count must match sampling params count")
        if len(generated_token_histories) != len(sampling_params_by_row):
            raise ValueError("generated token histories count must match sampling params count")

        any_penalty = False
        adjusted_rows: list[mx.array] = []
        for row_index, params in enumerate(sampling_params_by_row):
            row_logits = logits[row_index]
            if params is None:
                adjusted_rows.append(row_logits)
                continue

            repetition_penalty = float(getattr(params, "repetition_penalty", 1.0) or 1.0)
            presence_penalty = float(getattr(params, "presence_penalty", 0.0) or 0.0)

            if repetition_penalty != 1.0:
                row_logits = _apply_repetition_penalty_row_mx(
                    row_logits,
                    token_histories[row_index],
                    repetition_penalty,
                )
                any_penalty = True
            if presence_penalty != 0.0:
                row_logits = _apply_presence_penalty_row_mx(
                    row_logits,
                    generated_token_histories[row_index],
                    presence_penalty,
                )
                any_penalty = True

            adjusted_rows.append(row_logits)

        if not any_penalty:
            return logits
        return mx.stack(adjusted_rows, axis=0)

    def sample_next_token(
        self,
        logits: Any,
        *,
        sampling_params: SamplingParams | None,
        prompt_token_ids: list[int] | None = None,
        generated_token_ids: list[int] | None = None,
    ) -> int:
        """Sample a single token using the runner's RNG state.

        Args:
            logits: Logit scores for one sequence.
            sampling_params: Sampling configuration for the sequence.
            prompt_token_ids: Prompt-history token IDs for repetition penalty.
            generated_token_ids: Generated-history token IDs for penalties.

        Returns:
            The sampled token ID.
        """
        return self._sample_next_tokens_mx(
            logits,
            [sampling_params],
            token_histories=[list(prompt_token_ids or []) + list(generated_token_ids or [])],
            generated_token_histories=[list(generated_token_ids or [])],
        )[0]

    def _next_sample_key(self) -> mx.array:
        """Split and advance the runner-local MLX RNG key."""
        keys = mx.random.split(self._sample_key, num=2)
        self._sample_key = keys[0]
        return keys[1]

    def _get_compiled_sampler(
        self,
        logits_shape: tuple[int, ...],
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Any:
        """Return a cached ``mx.compile`` sampler for one shape/parameter tuple."""
        key = (tuple(int(dim) for dim in logits_shape), do_sample, float(temperature), int(top_k), float(top_p))
        cached = self._compiled_samplers.get(key)
        if cached is not None:
            return cached

        def compiled_sampler(logits: mx.array, rng_key: mx.array) -> mx.array:
            working_logits = logits.astype(mx.float32)
            if not do_sample:
                return mx.argmax(working_logits, axis=-1).astype(mx.int32)

            working_logits = _apply_temperature_mx(working_logits, temperature)
            shortlist_logits, shortlist_indices = _restrict_top_k_mx(working_logits, top_k)
            shortlist_logits = _apply_top_p_mx(
                shortlist_logits,
                top_p,
                assume_sorted=shortlist_indices is not None,
            )
            sampled_positions = mx.random.categorical(
                shortlist_logits,
                axis=-1,
                key=rng_key,
            ).astype(mx.int32)
            if shortlist_indices is None:
                return sampled_positions
            return mx.take_along_axis(
                shortlist_indices,
                sampled_positions[:, None],
                axis=-1,
            )[:, 0].astype(mx.int32)

        cached = mx.compile(compiled_sampler)
        self._compiled_samplers[key] = cached
        return cached

    def _sample_next_tokens_mx(
        self,
        logits: Any,
        sampling_params_by_row: list[SamplingParams | None],
        *,
        token_histories: list[list[int]] | None = None,
        generated_token_histories: list[list[int]] | None = None,
    ) -> list[int]:
        """Sample one token per row from MLX logits without host logits copies."""
        logits = _as_mx_logits(logits)
        if int(logits.shape[0]) != len(sampling_params_by_row):
            raise ValueError("sampling params count must match logits batch size")
        logits = self._apply_penalties_mx(
            logits,
            sampling_params_by_row,
            token_histories=token_histories,
            generated_token_histories=generated_token_histories,
        )

        grouped_rows: dict[tuple[bool, float, int, float], list[int]] = {}
        for row_index, params in enumerate(sampling_params_by_row):
            if params is None:
                key = (False, 1.0, 0, 1.0)
            else:
                do_sample = bool(params.do_sample)
                key = (
                    do_sample,
                    float(params.temperature) if do_sample else 1.0,
                    int(params.top_k) if do_sample else 0,
                    float(params.top_p) if do_sample else 1.0,
                )
            grouped_rows.setdefault(key, []).append(row_index)

        sampled_token_ids = [0] * len(sampling_params_by_row)
        for (do_sample, temperature, top_k, top_p), row_indices in grouped_rows.items():
            row_selector = mx.array(row_indices, dtype=mx.int32)
            group_logits = mx.take(logits, row_selector, axis=0)
            compiled_sampler = self._get_compiled_sampler(
                tuple(int(dim) for dim in group_logits.shape),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            rng_key = self._next_sample_key() if do_sample else self._sample_key
            group_tokens = compiled_sampler(group_logits, rng_key)
            _safe_eval(group_tokens)

            for row_index, token_id in zip(row_indices, group_tokens.tolist(), strict=False):
                sampled_token_ids[row_index] = int(token_id)
        return sampled_token_ids

    def sample_greedy_token_from_output(self, output: Any) -> int:
        """Greedily sample the next token from a raw model output."""
        token_ids = _argmax_token_ids(output)
        if not token_ids:
            raise RuntimeError("Runner produced no greedy token IDs.")
        return int(token_ids[0])

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a full scheduling step: forward pass, sampling, and bookkeeping.

        This is the main entry point satisfying
        :class:`ModelRunnerProtocol`.

        Args:
            request: The :class:`ExecutionRequest` produced by the
                scheduler.

        Returns:
            An :class:`ExecutionResult` with per-sequence updates,
            logits, and timing metadata.

        Raises:
            RuntimeError: If the model produces fewer logit rows than
                scheduled sequences.
        """
        started = time.perf_counter()
        if not request.sequences:
            return ExecutionResult(
                step_id=request.step_id,
                output=None,
                logits=None,
                elapsed_seconds=max(time.perf_counter() - started, 0.0),
            )

        t_fwd = time.perf_counter()
        raw_output = self._forward_step_raw(request)
        t_fwd_done = time.perf_counter()

        logits = _as_mx_logits(raw_output)
        if int(logits.shape[0]) < len(request.sequences):
            raise RuntimeError(
                f"Runner produced {logits.shape[0]} logits rows for {len(request.sequences)} scheduled sequences"
            )
        sampling_params_by_row = [request.sampling_for(sequence.request_id) for sequence in request.sequences]
        token_histories, generated_token_histories = self._sampling_histories_for_sequences(request.sequences)
        sampled_tokens = self._sample_next_tokens_mx(
            logits,
            sampling_params_by_row,
            token_histories=token_histories,
            generated_token_histories=generated_token_histories,
        )

        updates: list[ExecutionUpdate] = []
        for sequence, sampled_token in zip(request.sequences, sampled_tokens, strict=False):
            num_computed_tokens = int(sequence.num_computed_tokens) + len(sequence.token_ids)
            updates.append(
                ExecutionUpdate(
                    request_id=sequence.request_id,
                    row_index=int(sequence.row_index),
                    sampled_token_ids=[sampled_token],
                    num_computed_tokens=num_computed_tokens,
                )
            )

        self._apply_updates_to_sequence_buffer(updates)
        elapsed = max(time.perf_counter() - started, 0.0)
        self._perf_iteration += 1

        num_seqs = len(request.sequences)
        total_tokens = sum(len(s.token_ids) for s in request.sequences)
        num_prefill = sum(1 for s in request.sequences if s.num_computed_tokens == 0)
        num_decode = num_seqs - num_prefill
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        alpha = 0.1
        self._perf_tps_ema = alpha * tps + (1.0 - alpha) * self._perf_tps_ema if self._perf_tps_ema > 0 else tps

        req_bucket = self.select_bucket_size(num_seqs)

        fwd_ms = (t_fwd_done - t_fwd) * 1e3
        sample_ms = (elapsed - (t_fwd_done - started)) * 1e3

        self.log_it(
            "[perf] it=%06d reqs=%d/b%d(prefill=%d,decode=%d) "
            "tok=%d tps=%.0f ema=%.0f runner=%.2fms (fwd=%.1fms sample=%.1fms)",
            self._perf_iteration,
            num_seqs,
            req_bucket,
            num_prefill,
            num_decode,
            total_tokens,
            tps,
            self._perf_tps_ema,
            elapsed * 1e3,
            fwd_ms,
            sample_ms,
        )
        return ExecutionResult(
            step_id=request.step_id,
            updates=updates,
            output=raw_output,
            logits=logits,
            elapsed_seconds=elapsed,
            metadata={
                "mode": request.mode,
                "num_sequences": len(request.sequences),
                "selected_bucket_size": self.select_bucket_size(len(request.sequences)),
            },
        )


__all__ = "ModelRunner"
