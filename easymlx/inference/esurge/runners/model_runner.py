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
import numpy as np

from easymlx.workers.loggers import get_logger

from ..request import EngineRequest
from ..sampling_params import SamplingParams
from .execution_types import ExecutionRequest, ExecutionResult, ExecutionUpdate, ScheduledSequence
from .sequence_buffer import SequenceBuffer

logger = get_logger("eSurgeRunner")


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


def _as_numpy_logits(output: Any) -> np.ndarray:
    """Extract logits from a model output and convert to a 2-D NumPy array.

    Handles raw ``mx.array`` outputs, dict-wrapped outputs, and NumPy
    arrays.  3-D tensors are sliced to the last position along axis 1;
    1-D tensors are expanded.

    Args:
        output: Raw model output (object with ``.logits``, dict, or
            array).

    Returns:
        A ``float`` NumPy array of shape ``(batch_size, vocab_size)``.

    Raises:
        RuntimeError: If the resulting array has an unsupported number
            of dimensions.
    """
    logits = getattr(output, "logits", output)
    if isinstance(logits, dict):
        logits = logits.get("logits", logits)
    if isinstance(logits, mx.array):
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        mx.eval(logits)
        array = np.array(logits)
    else:
        array = np.asarray(logits)
        if array.ndim == 3:
            array = array[:, -1, :]
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return np.expand_dims(array, axis=0)
    raise RuntimeError(f"Unsupported logits shape from runner: {array.shape}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax over the last axis.

    Args:
        logits: 1-D logit array.

    Returns:
        Probability array with the same shape.

    Raises:
        RuntimeError: If the resulting distribution is not finite.
    """
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    denom = np.sum(exp_logits)
    if denom <= 0 or not np.isfinite(denom):
        raise RuntimeError("Sampling distribution is not finite.")
    return exp_logits / denom


def _sample_next_token(
    logits: np.ndarray,
    *,
    sampling_params: SamplingParams | None,
    rng: np.random.Generator,
) -> int:
    """Sample a single next token from logit scores.

    Applies temperature scaling, top-k filtering, and nucleus (top-p)
    filtering according to *sampling_params*.  When ``do_sample`` is
    ``False`` (or *sampling_params* is ``None``), greedy argmax decoding
    is used.

    Args:
        logits: 1-D logit array of shape ``(vocab_size,)``.
        sampling_params: Sampling configuration.  ``None`` triggers
            greedy decoding.
        rng: NumPy random generator for stochastic sampling.

    Returns:
        The sampled token ID.

    Raises:
        ValueError: If ``temperature <= 0`` when ``do_sample=True``.
    """
    if sampling_params is None:
        return int(np.argmax(logits))

    do_sample = bool(sampling_params.do_sample)
    temperature = float(sampling_params.temperature)
    top_k = int(sampling_params.top_k)
    top_p = float(sampling_params.top_p)

    if not do_sample:
        return int(np.argmax(logits))
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")

    logits = np.asarray(logits, dtype=np.float64) / temperature

    if top_k > 0 and top_k < logits.shape[0]:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < threshold, -np.inf, logits)

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]
        sorted_probs = _softmax(sorted_logits)
        cumulative_probs = np.cumsum(sorted_probs)
        keep = cumulative_probs <= top_p
        keep[0] = True
        filtered = np.full_like(logits, -np.inf)
        filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
        logits = filtered

    probs = _softmax(logits)
    return int(rng.choice(logits.shape[0], p=probs))


class ModelRunner:
    """Row-aware execution wrapper over a model or execution callable.

    Manages the lifecycle of active sequences in a :class:`SequenceBuffer`,
    drives model forward passes with KV-cache page views, performs token
    sampling, and tracks performance metrics.

    Attributes:
        model: The model callable (or ``nn.Module.__call__``).
        sequence_buffer: Row-oriented mutable buffer of active sequences.
        kv_caches: List of KV-cache objects (typically
            :class:`PagedKVCache` instances).
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
        self._rng = np.random.default_rng(seed)
        self._page_size = self._infer_page_size()
        self.log_it = logger.info if verbose else logger.debug

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

        logger.debug("ModelRunner initialization complete")

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

        from easymlx.caching import PageCache, PagedKVCache, PageMetadata, build_query_start_loc
        from easymlx.workers.loggers import ProgressLogger

        if not all(isinstance(c, PagedKVCache) for c in self.kv_caches):
            self.log_it("Skipping warmup: KV caches are not PagedKVCache instances")
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
            cache_views = [PageCache(kv_cache, slot_ids, query_lens=query_lens) for kv_cache in self.kv_caches]
            cache_metadata = PageMetadata(query_start_loc=build_query_start_loc(query_lens))

            try:
                out = _call_with_filtered_kwargs(
                    self.model,
                    input_ids=dummy_ids,
                    cache_views=cache_views,
                    cache_metadata=cache_metadata,
                    return_dict=True,
                )
                logits = getattr(out, "logits", out)
                mx.eval(logits)
                self._compiled_cache[key] = True
            except Exception as exc:
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
        """Return an ``mx.compile``'d forward with PageCache captured in closure.

        PageCache is captured in the closure (not passed as arg) so
        ``mx.compile`` only sees ``mx.array`` inputs. We cache compiled fns
        by ``(slot_ids, query_lens)`` since ``cache_metadata`` differs.
        The underlying PagedKVCache arrays are shared and mutated in-place.

        Args:
            slot_ids: Cache slot indices for this batch.
            query_lens: Per-sequence query lengths.

        Returns:
            A compiled callable, or ``None`` if KV caches are not
            configured.
        """
        if not self.kv_caches:
            return self.model.__call__

        return None

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

    def _zero_pages(self, page_ids: list["int"]) -> None:
        """Zero out the key and value caches for the given page IDs.

        Args:
            page_ids: List of page indices to zero.
        """
        if not page_ids or not self.kv_caches:
            return
        for cache in self.kv_caches:
            key_cache = getattr(cache, "key_cache", None)
            value_cache = getattr(cache, "value_cache", None)
            if key_cache is None or value_cache is None:
                continue
            for page_id in page_ids:
                if 0 <= int(page_id) < key_cache.shape[0]:
                    key_cache[int(page_id)] = 0
                    value_cache[int(page_id)] = 0

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
                block_tables[row_index, :count] = np.asarray(page_ids[:count], dtype=np.int32)
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
            from_table = np.array(block_tables[from_index], copy=True)
            from_kv_len = int(kv_lens[from_index])
            to_table = np.array(block_tables[to_index], copy=True)
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
                block_tables[to_index] = np.array(block_tables[from_index], copy=True)
                kv_lens[to_index] = int(kv_lens[from_index])
                block_tables[from_index, :] = -1
                kv_lens[from_index] = 0
        self.sequence_buffer.trim_trailing_holes()
        return moves

    @staticmethod
    def _pad_batch_token_ids(sequences: list[ScheduledSequence]) -> tuple[np.ndarray, list["int"]]:
        """Pad variable-length token ID sequences into a 2-D array.

        Args:
            sequences: Scheduled sequences for this step.

        Returns:
            A ``(batch_array, query_lens)`` tuple where *batch_array*
            has shape ``(num_seqs, max_query_len)`` and *query_lens*
            lists per-sequence lengths.
        """
        query_lens = [len(sequence.token_ids) for sequence in sequences]
        max_query_len = max(query_lens, default=0)
        batch = np.zeros((len(sequences), max_query_len), dtype=np.int32)
        for row_index, sequence in enumerate(sequences):
            if query_lens[row_index] == 0:
                continue
            batch[row_index, : query_lens[row_index]] = np.asarray(sequence.token_ids, dtype=np.int32)
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

    def _forward_step(self, request: ExecutionRequest) -> tuple[Any, np.ndarray]:
        """Execute a single model forward pass for the given execution request.

        Constructs the input tensors, cache views, and metadata, then
        invokes the model callable.

        Args:
            request: The :class:`ExecutionRequest` describing this step.

        Returns:
            A ``(raw_output, logits)`` tuple where *logits* is a 2-D
            NumPy array of shape ``(batch_size, vocab_size)``.
        """
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

        if self.kv_caches:
            batch_token_ids = mx.array(
                np.concatenate([np.asarray(seq.token_ids, dtype=np.int32) for seq in request.sequences])
            )

            if not has_extra:
                compiled_fn = self._get_compiled_forward(slot_ids, query_lens)
                if compiled_fn is not None:
                    try:
                        logits = compiled_fn(batch_token_ids)
                        mx.eval(logits)
                        raw_output = {"logits": logits}
                        return raw_output, _as_numpy_logits(raw_output)
                    except ValueError:
                        logger.debug("Compiled forward failed, falling back to eager")

            from easymlx.caching import PageCache, PageMetadata, build_query_start_loc

            cache_views = [PageCache(kv_cache, slot_ids, query_lens=query_lens) for kv_cache in self.kv_caches]
            cache_metadata = PageMetadata(query_start_loc=build_query_start_loc(query_lens))
        else:
            batch_token_ids, query_lens = self._pad_batch_token_ids(request.sequences)
            if not isinstance(batch_token_ids, mx.array):
                batch_token_ids = mx.array(batch_token_ids)
            cache_views = None
            cache_metadata = None

        raw_output = _call_with_filtered_kwargs(
            self.model,
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
        return raw_output, _as_numpy_logits(raw_output)

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
        raw_output, logits = self._forward_step(request)
        t_fwd_done = time.perf_counter()
        if logits.shape[0] < len(request.sequences):
            raise RuntimeError(
                f"Runner produced {logits.shape[0]} logits rows for {len(request.sequences)} scheduled sequences"
            )

        updates: list[ExecutionUpdate] = []
        for index, sequence in enumerate(request.sequences):
            num_computed_tokens = int(sequence.num_computed_tokens) + len(sequence.token_ids)
            sequence_logits = np.asarray(logits[index], dtype=np.float64)
            sampling_params = request.sampling_for(sequence.request_id)
            sampled_token = _sample_next_token(sequence_logits, sampling_params=sampling_params, rng=self._rng)
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
