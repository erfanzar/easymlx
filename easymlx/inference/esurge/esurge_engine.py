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

"""eSurge inference engine for easymlx (paged attention only).

The runtime stack is:

- request state in `request.py`
- admission and batching in `scheduler/`
- page/cache ownership in `core/`
- row placement and execution in `runners/`
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import numpy as np
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from easymlx.inference.reasoning import ReasoningParserManager, detect_reasoning_parser
from easymlx.inference.tools import ToolParserManager, detect_tool_parser
from easymlx.infra.factory import TaskType, registry
from easymlx.workers.loggers import get_logger

from .config import CacheConfig, Config, SchedulerConfig
from .outputs import CompletionOutput, RequestOutput
from .request import EngineRequest, EngineRequestStatus
from .runners import ExecutionManager, ExecutionRequest, ExecutionResult, ModelRunner, ScheduledSequence, SequenceBuffer
from .sampling_params import SamplingParams
from .scheduler import AsyncScheduler, Scheduler, SchedulerStepOutput
from .utils import (
    normalize_chat_template_messages,
    normalize_chat_template_tools,
    normalize_prompts,
    normalize_stop_sequences,
    to_structured_text_messages,
    truncate_tokens,
)

logger = get_logger("eSurge")


class DistributedControllerProtocol(Protocol):
    """Protocol defining the interface for distributed controller objects.

    Any object conforming to this protocol can be plugged into the eSurge
    engine to enable multi-node lockstep execution.

    Attributes:
        enabled: Whether distributed mode is active.
    """

    enabled: bool

    def start(self) -> None:
        """Initialize the distributed control plane."""
        ...

    def dispatch_step(self, scheduler_output: Any) -> Any:
        """Dispatch a scheduler step to remote workers.

        Args:
            scheduler_output: The scheduler output to broadcast.

        Returns:
            A dispatch token for later verification.
        """
        ...

    def verify_step(self, dispatch: Any, model_output: Any) -> None:
        """Verify that remote workers produced matching outputs.

        Args:
            dispatch: The dispatch token from :meth:`dispatch_step`.
            model_output: The leader's model output to compare against.
        """
        ...

    def shutdown(self) -> None:
        """Shut down the distributed control plane."""
        ...


class MultimodalBatchPreprocessorProtocol(Protocol):
    """Protocol for multimodal batch preprocessors.

    Implementations prepare multimodal inputs (images, audio, etc.) for a
    batch of requests before they are fed to the model runner.
    """

    def prepare_batch(
        self,
        *,
        requests: list[EngineRequest],
        runtime_states: list[Any],
        step_output: SchedulerStepOutput,
    ) -> Any:
        """Prepare multimodal inputs for a scheduled batch.

        Args:
            requests: List of engine requests in the batch.
            runtime_states: Corresponding runtime state objects.
            step_output: The scheduler step output describing the batch.

        Returns:
            Preprocessed multimodal data to pass to the model runner.
        """
        ...


def _apply_stop_strings(
    text: str,
    stop: list["str"] | None,
    *,
    include_stop_str_in_output: bool = False,
) -> tuple[str, str | None]:
    """Truncate text at the earliest occurrence of any stop string.

    Args:
        text: The generated text to scan.
        stop: List of stop strings. ``None`` or empty means no truncation.
        include_stop_str_in_output: If ``True``, the matched stop string
            is included in the returned text.

    Returns:
        A 2-tuple of ``(truncated_text, stop_reason)`` where
        *stop_reason* is ``"stop"`` if truncation occurred, else ``None``.
    """
    if not stop:
        return text, None
    earliest = None
    matched_stop = None
    for stop_string in stop:
        idx = text.find(stop_string)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
            matched_stop = stop_string
    if earliest is None or matched_stop is None:
        return text, None
    cutoff = earliest + len(matched_stop) if include_stop_str_in_output else earliest
    return text[:cutoff], "stop"


def _normalize_eos_token_ids(eos_token_id: int | list["int"] | tuple[int, ...] | None) -> set["int"]:
    """Normalize an EOS token id (or collection thereof) into a set of ints.

    Args:
        eos_token_id: A single token id, a list/tuple/set of ids, or
            ``None``.

    Returns:
        A set of integer EOS token ids (empty if *eos_token_id* is ``None``).
    """
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def _call_with_filtered_kwargs(callable_obj: Any, /, **kwargs: Any) -> Any:
    """Call a callable, filtering kwargs to only those it accepts.

    Inspects the callable's signature and strips out any keyword arguments
    that it does not declare, unless it accepts ``**kwargs``.

    Args:
        callable_obj: The function or method to invoke.
        **kwargs: Keyword arguments to pass (after filtering).

    Returns:
        The return value of the callable.
    """
    signature = inspect.signature(callable_obj)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_kwargs:
        return callable_obj(**kwargs)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return callable_obj(**filtered)


@dataclass(slots=True)
class _RuntimeRequestState:
    """Mutable per-request state tracked by the eSurge background loop.

    Bundles the :class:`EngineRequest` with generation configuration,
    streaming bookkeeping, and parser instances for the lifetime of a
    single generation request.

    Attributes:
        request: The underlying engine request.
        gen_kwargs: Resolved generation keyword arguments.
        eos_token_ids: Set of token ids that signal end-of-sequence.
        pad_token_id: Token id used for padding, or ``None``.
        stop: List of stop strings, or ``None``.
        created_at: Epoch timestamp when this state was created.
        first_token_time: Epoch timestamp of the first generated token.
        error: Exception that caused the request to fail, if any.
        done: Event set when the request is complete.
        update_seq: Monotonically increasing counter of token updates.
        last_emitted_text: Full text emitted in the last streaming yield.
        last_emitted_reasoning: Reasoning text emitted in the last yield.
        last_emitted_tool_calls: Tool calls emitted in the last yield.
        last_emitted_update_seq: ``update_seq`` at the last yield.
        last_emitted_finished: Whether the last yield marked completion.
        tool_parser_instance: Per-request tool parser instance.
        reasoning_parser_instance: Per-request reasoning parser instance.
    """

    request: EngineRequest
    gen_kwargs: dict[str, Any]
    eos_token_ids: set["int"]
    pad_token_id: int | None
    stop: list["str"] | None
    created_at: float = field(default_factory=time.time)
    first_token_time: float | None = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    update_seq: int = 0
    last_emitted_text: str = ""
    last_emitted_reasoning: str = ""
    last_emitted_tool_calls: list[dict[str, Any]] | None = None
    last_emitted_update_seq: int = 0
    last_emitted_finished: bool = False
    tool_parser_instance: Any | None = None
    reasoning_parser_instance: Any | None = None

    @property
    def prompt_token_ids(self) -> list["int"]:
        """Return the prompt token ids from the underlying request."""
        return self.request.prompt_token_ids

    @property
    def generated_token_ids(self) -> list["int"]:
        """Return the generated token ids accumulated so far."""
        return self.request.generated_token_ids

    @property
    def is_done(self) -> bool:
        """Whether the request has reached a terminal state."""
        return self.request.is_finished

    @property
    def request_duration(self) -> float:
        """Elapsed wall-clock time in seconds since the request was created."""
        if self.request.finished_at is None:
            return time.time() - self.created_at
        return self.request.finished_at - self.created_at

    def decoded_text(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Decode the generated token ids into a string.

        Args:
            tokenizer: The tokenizer used for decoding.

        Returns:
            The decoded text with special tokens removed.
        """
        return tokenizer.decode(self.request.generated_token_ids, skip_special_tokens=True)

    def note_generated_tokens(self, token_ids: list["int"]) -> None:
        """Record that new tokens have been generated.

        Sets :attr:`first_token_time` on the first call with non-empty
        *token_ids* and increments :attr:`update_seq`.

        Args:
            token_ids: Newly generated token ids.
        """
        if token_ids and self.first_token_time is None:
            self.first_token_time = time.time()
        self.update_seq += len(token_ids)

    def mark_error(self, error: BaseException) -> None:
        """Transition the request to a failed state.

        Args:
            error: The exception that caused the failure.
        """
        self.error = error
        self.request.mark_failed(str(error))
        self.done.set()

    def mark_canceled(self) -> None:
        """Transition the request to a canceled state."""
        if not self.request.is_finished:
            self.request.mark_canceled()
        self.done.set()

    def finalize(self) -> None:
        """Signal that this request is complete (sets the done event)."""
        self.done.set()


class eSurge:
    """Paged attention inference engine with EasyDeL-like public methods.

    eSurge is the primary inference engine class for easymlx. It manages a
    background scheduling loop, paged KV cache, model runner, and optional
    distributed coordination to serve generation requests with continuous
    batching.

    Args:
        model: The language model instance. Must expose ``init_operations_cache``
            or ``init_paged_cache`` for KV cache initialization.
        tokenizer: A tokenizer instance, a HuggingFace model id string, or
            ``None`` to auto-resolve from the model.
        max_model_len: Maximum total sequence length (prompt + generation).
        max_num_seqs: Maximum number of concurrent sequences.
        max_num_seq_buckets: Optional list of bucket sizes for sequence
            length bucketing.
        max_num_batched_tokens: Maximum tokens processed per step.
        memory_utilization: Fraction of available memory to use for caches.
        page_size: Number of tokens per KV cache page.
        enable_prefix_caching: Whether to enable prefix-aware page sharing.
        reserve_tokens: Tokens reserved for scheduling headroom.
        auto_truncate_prompt: Whether to truncate prompts that exceed the
            context window.
        auto_cap_new_tokens: Whether to cap ``max_new_tokens`` to fit
            within the model length.
        strict_context: If ``True``, raise instead of silently truncating.
        truncate_mode: How to truncate (``"left"``, ``"right"``,
            ``"middle"``).
        prefer_preserve_prompt: Prefer preserving the full prompt when
            truncating.
        decode_truncated_prompt: Whether to decode the truncated prompt
            back to text.
        extra_eos_token_ids: Additional EOS token ids beyond the
            tokenizer's default.
        extra_stops: Additional stop strings applied to all requests.
        tool_parser: Name of the tool parser to use, or ``None`` for
            auto-detection.
        reasoning_parser: Name of the reasoning parser to use, or ``None``
            for auto-detection.
        ignore_stop_strings_in_reasoning: If ``True``, stop strings are
            not applied to reasoning content.
        runner_verbose: Enable verbose logging in the model runner.
        silent_mode: Suppress informational log messages.
        processor: Optional multimodal processor (used as fallback
            tokenizer source).
        config: Optional :class:`Config` override. If provided, scheduler
            and cache configs are taken from it.
        distributed_controller: Optional distributed controller for
            multi-node execution.
        multimodal_preprocessor: Optional preprocessor for multimodal
            inputs.
        seed: Random seed for reproducible sampling.
        **_kwargs: Ignored keyword arguments for forward compatibility.

    Raises:
        ValueError: If the model does not support paged attention or if
            ``max_model_len`` is too small.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        *,
        max_model_len: int = 8192,
        max_num_seqs: int = 4,
        max_num_seq_buckets: list[int] | None = None,
        max_num_batched_tokens: int | None = 1024,
        memory_utilization: float = 0.6,
        page_size: int = 128,
        enable_prefix_caching: bool = True,
        reserve_tokens: int | None = None,
        auto_truncate_prompt: bool = True,
        auto_cap_new_tokens: bool = True,
        strict_context: bool = False,
        truncate_mode: Literal["left", "right", "middle"] = "left",
        prefer_preserve_prompt: bool = True,
        decode_truncated_prompt: bool = True,
        extra_eos_token_ids: list[int] | None = None,
        extra_stops: list[str] | None = None,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        ignore_stop_strings_in_reasoning: bool = True,
        runner_verbose: bool = False,
        silent_mode: bool = False,
        processor: Any | None = None,
        config: Config | None = None,
        distributed_controller: DistributedControllerProtocol | None = None,
        multimodal_preprocessor: MultimodalBatchPreprocessorProtocol | Any | None = None,
        seed: int = 0,
        **_kwargs: Any,
    ):
        scheduler_config = (
            config.scheduler_config
            if config is not None
            else SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seq_buckets=tuple(max_num_seq_buckets) if max_num_seq_buckets else None,
            )
        )
        cache_config = (
            config.cache_config
            if config is not None
            else CacheConfig(
                page_size=page_size,
                enable_prefix_caching=enable_prefix_caching,
            )
        )
        self.config = config or Config(scheduler_config=scheduler_config, cache_config=cache_config)

        if reserve_tokens is None:
            reserve_tokens = max(1, scheduler_config.max_num_seqs)
        if scheduler_config.max_model_len <= reserve_tokens:
            raise ValueError(
                f"max_model_len ({scheduler_config.max_model_len}) must exceed reserve_tokens ({reserve_tokens})"
            )

        self.model = model
        self.max_model_len = int(scheduler_config.max_model_len)
        self.max_num_seqs = int(scheduler_config.max_num_seqs)
        self.reserve_tokens = int(reserve_tokens)
        self.auto_truncate_prompt = bool(auto_truncate_prompt)
        self.auto_cap_new_tokens = bool(auto_cap_new_tokens)
        self.strict_context = bool(strict_context)
        self.truncate_mode = str(truncate_mode)
        self.prefer_preserve_prompt = bool(prefer_preserve_prompt)
        self.decode_truncated_prompt = bool(decode_truncated_prompt)
        self.extra_eos_token_ids = list(extra_eos_token_ids or [])
        self.extra_stops = normalize_stop_sequences(extra_stops)
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.ignore_stop_strings_in_reasoning = bool(ignore_stop_strings_in_reasoning)
        self.memory_utilization = float(memory_utilization)
        self.silent_mode = bool(silent_mode)
        self.runner_verbose = bool(runner_verbose)
        self._info = logger.info if not self.silent_mode else (lambda *a, **kw: None)
        self._log_it = logger.info if self.runner_verbose else logger.debug
        self.tokenizer = self._resolve_tokenizer(tokenizer or processor)
        self._rng_seed = int(seed)
        self._distributed_controller = distributed_controller
        self._multimodal_preprocessor = multimodal_preprocessor

        self._lock = threading.Lock()
        self._work_cv = threading.Condition(self._lock)
        self._shutdown = threading.Event()
        self._has_terminal_states = False
        self._paused = False
        self._background_thread: threading.Thread | None = None
        self._scheduler: Scheduler | AsyncScheduler | None = None
        self._runner: ModelRunner | None = None
        self._execution_manager: ExecutionManager | None = None
        self._runtime_states: dict[str, _RuntimeRequestState] = {}
        self._step_counter = 0
        self._controller_started = False

        model_type = str(
            getattr(self.model, "_model_type", None)
            or getattr(getattr(self.model, "config", None), "model_type", "")
            or ""
        )
        if tool_parser is None:
            tool_parser = detect_tool_parser(model_type=model_type, tokenizer=self.tokenizer)
        if reasoning_parser is None:
            reasoning_parser = detect_reasoning_parser(model_type=model_type, tokenizer=self.tokenizer)
        self.tool_parser_name = tool_parser
        self.reasoning_parser_name = reasoning_parser
        self._tool_parser_class = ToolParserManager.get_tool_parser(tool_parser) if tool_parser else None
        self._reasoning_parser_class = (
            ReasoningParserManager.get_reasoning_parser(reasoning_parser) if reasoning_parser else None
        )

        if tool_parser:
            if self._tool_parser_class is not None:
                self._info("Initialized tool parser: %s", tool_parser)
            else:
                logger.warning("Tool parser '%s' not found, function calling disabled", tool_parser)
        if reasoning_parser:
            if self._reasoning_parser_class is not None:
                self._info("Initialized reasoning parser: %s", reasoning_parser)
            else:
                logger.warning("Reasoning parser '%s' not found, reasoning disabled", reasoning_parser)

        self._validate_paged_interface()
        self._initialize_runtime()

        self._log_startup_summary(model_type, tool_parser, reasoning_parser)

    def _log_startup_summary(self, model_type: str, tool_parser: str | None, reasoning_parser: str | None) -> None:
        """Log a formatted summary of the engine configuration at startup.

        Args:
            model_type: String identifier for the model architecture.
            tool_parser: Name of the configured tool parser, or ``None``.
            reasoning_parser: Name of the configured reasoning parser, or ``None``.
        """
        try:
            config = getattr(self.model, "config", None)
            vocab_size = getattr(config, "vocab_size", 0) or 0
            num_layers = getattr(config, "num_hidden_layers", 0) or 0
            num_attn_heads = max(getattr(config, "num_attention_heads", 1) or 1, 1)
            hidden_size = getattr(config, "hidden_size", 0) or 0
            num_kv_heads = getattr(config, "num_key_value_heads", 0) or 0
            head_dim = hidden_size // num_attn_heads if hidden_size else 0
            layer_types = getattr(config, "layer_types", None)

            if layer_types is not None:
                from collections import Counter

                type_counts = Counter(layer_types)
                n_attn = sum(v for k, v in type_counts.items() if "full" in k or "sliding" in k)
                n_linear = sum(v for k, v in type_counts.items() if "linear" in k)
                n_other = num_layers - n_attn - n_linear
                parts = []
                if n_linear:
                    parts.append(f"{n_linear} linear")
                if n_attn:
                    parts.append(f"{n_attn} full-attention")
                if n_other:
                    parts.append(f"{n_other} other")
                has_recurrent = n_linear > 0
                has_attention = n_attn > 0
                if has_recurrent and has_attention:
                    arch_desc = f"hybrid ({' + '.join(parts)} / {num_layers} layers)"
                elif has_recurrent:
                    arch_desc = f"recurrent ({' + '.join(parts)} / {num_layers} layers)"
                else:
                    arch_desc = f"attention ({num_layers} layers)"
            elif num_layers > 0:
                arch_desc = f"attention ({num_layers} layers)"
            else:
                arch_desc = "unknown"

            algo_str = "attention=paged_attention"

            page_size = self.cache_config.page_size
            num_pages = self.cache_config.num_pages or (self.max_num_seqs * self.max_model_len // max(page_size, 1))
            seq_capacity_k = int((num_pages * page_size) / 1000)

            cache_parts = [
                "type=paged",
                f"pages={num_pages:,} ({page_size} tok/page)",
                f"sequence_capacity={seq_capacity_k:,}K",
            ]

            if num_pages > 8192:
                logger.warning(
                    "Large page allocation: %d pages (%d seqs x %d tokens / %d block_size). "
                    "This may cause slow inference on Apple Silicon. "
                    "Consider lowering max_num_seqs or max_model_len.",
                    num_pages,
                    self.max_num_seqs,
                    self.max_model_len,
                    page_size,
                )

            lines = [
                f"Model         : {model_type or type(self.model).__name__}",
                f"Vocab         : {vocab_size:,} tokens",
                f"Architecture  : {arch_desc}",
                f"KV heads      : {num_kv_heads} x {head_dim}d",
                f"Algorithms    : {algo_str}",
                f"Cache         : {' | '.join(cache_parts)}",
                f"Memory util   : {self.memory_utilization:.0%}",
                f"Runtime       : max_model_len={self.max_model_len:,} | "
                f"max_num_seqs={self.max_num_seqs} | "
                f"max_batched_tokens={self.scheduler_config.max_num_batched_tokens or self.max_model_len:,}",
                f"Tool parser   : {tool_parser or 'none'}",
                f"Reason parser : {reasoning_parser or 'none'}",
            ]
            self._info("\n".join(lines))
        except Exception as exc:
            logger.debug("Could not generate startup summary: %s", exc)

    def _initialize_runtime(self) -> None:
        """Bootstrap the model runner, scheduler, execution manager, and background thread."""
        kv_caches = self._init_paged_cache()
        self._reset_recurrent_states()
        self._info("Initializing paged runtime with %d KV cache layers", len(kv_caches))
        supports_async_overlap = bool(getattr(self.model, "supports_async_overlap", False))
        self._runner = ModelRunner(
            self.model,
            sequence_buffer=SequenceBuffer(max_num_rows=self.max_num_seqs),
            kv_caches=kv_caches,
            max_num_seq_buckets=self.scheduler_config.max_num_seq_buckets,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            memory_utilization=self.memory_utilization,
            supports_async_overlap=supports_async_overlap,
            verbose=self.runner_verbose,
            seed=self._rng_seed,
        )
        self._runner.precompile()
        scheduler_cls: type[Scheduler | AsyncScheduler] = AsyncScheduler if supports_async_overlap else Scheduler
        self._scheduler = scheduler_cls(self.config)
        self._execution_manager = ExecutionManager(self._runner, max_workers=1)
        self._start_distributed_controller()
        self._background_thread = threading.Thread(target=self._run_loop, name="eSurgeWorker", daemon=True)
        self._background_thread.start()
        self._info("eSurge runtime ready")

    def _start_distributed_controller(self) -> None:
        """Start the distributed controller if one is configured."""
        controller = self._distributed_controller
        if controller is None or self._controller_started:
            return
        start = getattr(controller, "start", None)
        if callable(start):
            start()
            self._controller_started = True

    def _stop_distributed_controller(self) -> None:
        """Shut down the distributed controller if it was started."""
        controller = self._distributed_controller
        if controller is None or not self._controller_started:
            return
        shutdown = getattr(controller, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._controller_started = False

    def _resolve_tokenizer(self, tokenizer: str | PreTrainedTokenizerBase | None) -> PreTrainedTokenizerBase:
        """Resolve a tokenizer from a string, instance, or the model's name_or_path.

        Args:
            tokenizer: A tokenizer instance, a HuggingFace model id, or
                ``None`` to auto-resolve.

        Returns:
            A :class:`PreTrainedTokenizerBase` with ``pad_token`` set.

        Raises:
            ValueError: If no tokenizer can be resolved.
        """
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            resolved = tokenizer
        elif isinstance(tokenizer, str):
            resolved = AutoTokenizer.from_pretrained(tokenizer)
        else:
            model_id = getattr(self.model, "name_or_path", None)
            if not model_id:
                raise ValueError("Tokenizer must be provided for easymlx eSurge.")
            resolved = AutoTokenizer.from_pretrained(model_id)

        if resolved.pad_token is None:
            if resolved.eos_token is not None:
                resolved.pad_token = resolved.eos_token
            elif resolved.unk_token is not None:
                resolved.pad_token = resolved.unk_token
        return resolved

    def _validate_paged_interface(self) -> None:
        """Validate the model supports paged attention (required)."""
        has_ops = callable(getattr(self.model, "init_operations_cache", None))
        has_paged = any(callable(getattr(self.model, name, None)) for name in ("init_paged_cache", "init_paged_caches"))
        if not has_ops and not has_paged:
            raise ValueError(
                f"{type(self.model).__name__} does not support paged attention. "
                "Model must provide `init_operations_cache()` or `init_paged_cache()`."
            )

    def _reset_recurrent_states(self) -> None:
        """Reset linear attention / recurrent states on the model.

        Unlike paged KV caches which are managed by the page pool,
        recurrent states (conv_state, recurrent_state) live on the model
        modules themselves. They must be zeroed when requests finish so
        the next request starts clean.
        """
        model = self.model
        layers = None
        for attr in ("layers", "model.layers", "language_model.layers", "model.language_model.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue
        if layers is None:
            return
        for layer in layers:
            for attr_name in ("linear_attn", "recurrent", "ssm"):
                module = getattr(layer, attr_name, None)
                if module is not None and hasattr(module, "reset_state"):
                    module.reset_state(batch_size=1)

    def _init_paged_cache(self) -> list[Any]:
        """Initialize paged KV caches via the model's cache init method.

        Returns:
            A list of KV cache objects (one per layer).

        Raises:
            ValueError: If no suitable init method is found or it returns
                empty results.
        """
        init_ops = getattr(self.model, "init_operations_cache", None)
        if callable(init_ops):
            logger.debug("Initializing KV cache via init_operations_cache")
            result = init_ops(
                batch_size=self.max_num_seqs,
                max_length=self.max_model_len,
                page_size=self.cache_config.page_size,
                dtype=np.float16,
                cache_type="paged",
            )
            if isinstance(result, (list, tuple)):
                cache_list = list(result)
            elif hasattr(result, "views"):
                cache_list = list(result.views)
            elif hasattr(result, "__iter__"):
                cache_list = list(result)
            else:
                cache_list = [result]
            if not cache_list:
                raise ValueError("init_operations_cache returned no caches.")
            return cache_list

        init_method = None
        init_method_name = None
        for name in ("init_paged_cache", "init_paged_caches"):
            candidate = getattr(self.model, name, None)
            if callable(candidate):
                init_method = candidate
                init_method_name = name
                break
        if init_method_name is not None:
            logger.debug("Initializing KV cache via %s", init_method_name)
        if init_method is None:
            raise ValueError(
                "Paged engine requires `init_operations_cache(...)` or `init_paged_cache(...)` on the model."
            )

        caches = _call_with_filtered_kwargs(
            init_method,
            num_seqs=self.max_num_seqs,
            max_seq_len=self.max_model_len,
            page_size=self.cache_config.page_size,
            dtype=np.float16,
        )
        cache_list = list(caches)
        if not cache_list:
            raise ValueError("Paged cache initialization returned no caches.")
        return cache_list

    def _resolve_generation_kwargs(
        self,
        sampling_params: SamplingParams | None,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Merge sampling params and extra kwargs into a single dict.

        Combines values from *sampling_params*, *generate_kwargs*, and the
        engine's :attr:`extra_stops` into a unified generation config.

        Args:
            sampling_params: Optional sampling parameters.
            **generate_kwargs: Additional keyword arguments that override
                values from *sampling_params*.

        Returns:
            A merged dictionary of generation keyword arguments.
        """
        kwargs: dict[str, Any] = {}
        if sampling_params is not None:
            kwargs.update(sampling_params.to_generation_kwargs())
            if sampling_params.extra_args:
                kwargs.update({key: value for key, value in sampling_params.extra_args.items() if value is not None})
            if sampling_params.stop is not None:
                kwargs["stop"] = list(sampling_params.stop)
        kwargs.update({key: value for key, value in generate_kwargs.items() if value is not None})
        if self.extra_stops:
            merged_stops = normalize_stop_sequences(kwargs.get("stop"))
            seen = set(merged_stops)
            for stop in self.extra_stops:
                if stop not in seen:
                    merged_stops.append(stop)
                    seen.add(stop)
            kwargs["stop"] = merged_stops
        return kwargs

    def _new_tool_parser(self) -> Any | None:
        """Create a fresh tool parser instance for a new request.

        Returns:
            A tool parser instance, or ``None`` if no parser is configured.
        """
        return self._tool_parser_class(self.tokenizer) if self._tool_parser_class is not None else None

    def _new_reasoning_parser(self) -> Any | None:
        """Create a fresh reasoning parser instance for a new request.

        Returns:
            A reasoning parser instance, or ``None`` if no parser is
            configured.
        """
        return self._reasoning_parser_class(self.tokenizer) if self._reasoning_parser_class is not None else None

    @staticmethod
    def _delta_text(current: str, previous: str) -> str:
        """Compute the incremental delta between two text snapshots.

        Args:
            current: The current full text.
            previous: The previously emitted text.

        Returns:
            The new text that was not part of *previous*.
        """
        if current.startswith(previous):
            return current[len(previous) :]
        return current

    @staticmethod
    def _delta_tool_calls(
        current: list[dict[str, Any]] | None,
        previous: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Compute the incremental delta between two tool-call lists.

        Args:
            current: The current full list of tool calls.
            previous: The previously emitted tool calls.

        Returns:
            New tool calls not present in *previous*, or ``None`` if there
            is no delta.
        """
        if not current:
            return None
        previous_list = previous or []
        if len(current) >= len(previous_list) and current[: len(previous_list)] == previous_list:
            delta = current[len(previous_list) :]
            return delta or None
        return current

    def _parse_text_output(
        self,
        raw_text: str,
        *,
        stop: list["str"] | None,
        finish_reason: str | None,
        gen_kwargs: dict[str, Any],
        tool_parser: Any | None,
        reasoning_parser: Any | None,
    ) -> dict[str, Any]:
        """Parse raw generated text through stop-string, reasoning, and tool parsers.

        Args:
            raw_text: The decoded generated text.
            stop: List of stop strings to apply.
            finish_reason: Finish reason from the scheduler, or ``None``.
            gen_kwargs: Generation kwargs (checked for
                ``include_stop_str_in_output`` and related flags).
            tool_parser: Tool parser instance, or ``None``.
            reasoning_parser: Reasoning parser instance, or ``None``.

        Returns:
            A dictionary with keys ``"text"``, ``"reasoning_content"``,
            ``"tool_calls"``, ``"stop_reason"``, and ``"finish_reason"``.
        """
        include_stop = bool(gen_kwargs.get("include_stop_str_in_output", False))
        ignore_reasoning_stops = gen_kwargs.get("ignore_stop_strings_in_reasoning")
        if ignore_reasoning_stops is None:
            ignore_reasoning_stops = self.ignore_stop_strings_in_reasoning

        stop_reason = None
        reasoning_content: str | None = None
        visible_text = raw_text

        if reasoning_parser is not None and bool(ignore_reasoning_stops):
            reasoning_content, visible_text = reasoning_parser.extract_reasoning(raw_text)
            visible_text = visible_text or ""
            visible_text, stop_reason = _apply_stop_strings(
                visible_text,
                stop,
                include_stop_str_in_output=include_stop,
            )
        else:
            trimmed_text, stop_reason = _apply_stop_strings(
                raw_text,
                stop,
                include_stop_str_in_output=include_stop,
            )
            if reasoning_parser is not None:
                reasoning_content, visible_text = reasoning_parser.extract_reasoning(trimmed_text)
                visible_text = visible_text or ""
            else:
                visible_text = trimmed_text

        tool_calls = None
        if tool_parser is not None:
            result = tool_parser.extract_tool_calls(visible_text or "", None)
            if result.tools_called:
                tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in result.tool_calls]
                visible_text = result.content or ""
            else:
                visible_text = result.content or visible_text

        effective_finish_reason = finish_reason or stop_reason
        if tool_calls and effective_finish_reason in {None, "eos", "stop", "length"}:
            effective_finish_reason = "tool_calls"

        return {
            "text": visible_text,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
            "finish_reason": effective_finish_reason,
        }

    def _state_to_request_output(
        self,
        state: _RuntimeRequestState,
        *,
        incremental: bool = False,
        consume: bool = False,
    ) -> RequestOutput:
        """Convert a runtime request state into a :class:`RequestOutput`.

        Args:
            state: The per-request runtime state.
            incremental: If ``True``, compute delta fields relative to the
                last emitted output.
            consume: If ``True`` (and *incremental*), update the state's
                last-emitted bookkeeping so subsequent calls produce
                correct deltas.

        Returns:
            A fully populated :class:`RequestOutput`.
        """
        parsed = self._parse_text_output(
            state.decoded_text(self.tokenizer),
            stop=state.stop,
            finish_reason=state.request.finished_reason,
            gen_kwargs=state.gen_kwargs,
            tool_parser=state.tool_parser_instance,
            reasoning_parser=state.reasoning_parser_instance,
        )
        text = str(parsed["text"])
        reasoning_content = parsed["reasoning_content"]
        tool_calls = parsed["tool_calls"]
        finish_reason = parsed["finish_reason"]

        previous_text = state.last_emitted_text if incremental else ""
        previous_reasoning = state.last_emitted_reasoning if incremental else ""
        previous_tool_calls = state.last_emitted_tool_calls if incremental else None
        previous_seq = state.last_emitted_update_seq if incremental else 0
        delta_text = self._delta_text(text, previous_text)
        reasoning_now = reasoning_content or ""
        delta_reasoning = self._delta_text(reasoning_now, previous_reasoning) if reasoning_now else None
        delta_tool_calls = self._delta_tool_calls(tool_calls, previous_tool_calls)

        elapsed = max(state.request_duration, 1e-6)
        num_generated = len(state.generated_token_ids)
        tps = float(num_generated) / elapsed if num_generated else 0.0
        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=list(state.generated_token_ids),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        output = RequestOutput(
            request_id=state.request.request_id,
            prompt=state.request.prompt,
            prompt_token_ids=state.prompt_token_ids,
            outputs=[completion],
            finished=state.is_done,
            metrics={
                "num_generated_tokens": num_generated,
                "num_computed_tokens": int(state.request.num_computed_tokens),
                "num_cached_tokens": int(state.request.num_cached_tokens),
                "time_spent_generating": elapsed,
                "tokens_per_second": tps,
            },
            accumulated_text=text,
            delta_text=delta_text,
            reasoning_content=reasoning_content,
            delta_reasoning_content=delta_reasoning,
            tool_calls=tool_calls,
            delta_tool_calls=delta_tool_calls,
            tokens_per_second=tps,
            num_generated_tokens=num_generated,
            time_spent_generating=elapsed,
            first_token_time=state.first_token_time,
            processing_time=elapsed,
            update_seq=state.update_seq,
            delta_seq=max(0, state.update_seq - previous_seq),
        )

        if consume and incremental:
            state.last_emitted_text = text
            state.last_emitted_reasoning = reasoning_now
            state.last_emitted_tool_calls = list(tool_calls) if tool_calls is not None else None
            state.last_emitted_update_seq = state.update_seq
            state.last_emitted_finished = state.is_done
        return output

    def _prepare_inputs(
        self,
        prompts: list["str"],
        *,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray, list[list["int"]]]:
        """Tokenize, truncate, and pad a list of prompt strings.

        Args:
            prompts: Raw prompt strings.
            max_new_tokens: Maximum generation tokens (used to compute
                the allowed prompt length).

        Returns:
            A 3-tuple of ``(input_ids, attention_mask, prompt_token_ids)``
            where the first two are padded numpy arrays and the third is a
            list of per-prompt unpadded token id lists.
        """
        enc = self.tokenizer(prompts, return_tensors="np", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        prompt_token_ids: list[list["int"]] = []
        truncated_ids: list[list["int"]] = []
        truncated_masks: list[list["int"]] = []

        allowed_prompt = self.max_model_len - self.reserve_tokens - max_new_tokens
        if allowed_prompt <= 0:
            if self.strict_context:
                raise ValueError("No room for prompt tokens after reserve/max_new_tokens constraints.")
            allowed_prompt = max(1, self.max_model_len - self.reserve_tokens)

        for row_ids, row_mask in zip(input_ids, attention_mask, strict=False):
            tokens = row_ids[row_mask.astype(bool)].tolist()
            if self.auto_truncate_prompt and len(tokens) > allowed_prompt:
                tokens, _ = truncate_tokens(tokens, allowed_prompt, mode=self.truncate_mode)
            prompt_token_ids.append(tokens)

        max_len = max((len(tokens) for tokens in prompt_token_ids), default=0)
        for tokens in prompt_token_ids:
            pad_len = max_len - len(tokens)
            truncated_ids.append(tokens + [pad_token_id] * pad_len)
            truncated_masks.append([1] * len(tokens) + [0] * pad_len)

        return (
            np.array(truncated_ids, dtype=np.int32),
            np.array(truncated_masks, dtype=np.int32),
            prompt_token_ids,
        )

    def _sampling_params_from_generation_kwargs(
        self, gen_kwargs: dict[str, Any], *, max_new_tokens: int
    ) -> SamplingParams:
        """Build a :class:`SamplingParams` from a generation kwargs dict.

        Args:
            gen_kwargs: Merged generation keyword arguments.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            A :class:`SamplingParams` populated from *gen_kwargs*.
        """
        known_fields = {
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "do_sample",
            "stop",
            "eos_token_id",
            "pad_token_id",
            "include_stop_str_in_output",
            "ignore_stop_strings_in_reasoning",
            "tools",
            "tool_choice",
            "response_format",
        }
        extra_args = {key: value for key, value in gen_kwargs.items() if key not in known_fields}
        return SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(gen_kwargs.get("temperature", 1.0) or 1.0),
            top_k=int(gen_kwargs.get("top_k", 0) or 0),
            top_p=float(gen_kwargs.get("top_p", 1.0) or 1.0),
            do_sample=bool(gen_kwargs.get("do_sample", False)),
            stop=list(gen_kwargs.get("stop")) if gen_kwargs.get("stop") is not None else None,
            eos_token_id=gen_kwargs.get("eos_token_id"),
            pad_token_id=gen_kwargs.get("pad_token_id"),
            include_stop_str_in_output=bool(gen_kwargs.get("include_stop_str_in_output", False)),
            ignore_stop_strings_in_reasoning=gen_kwargs.get("ignore_stop_strings_in_reasoning"),
            tools=gen_kwargs.get("tools"),
            tool_choice=gen_kwargs.get("tool_choice"),
            response_format=gen_kwargs.get("response_format"),
            extra_args=extra_args,
        )

    def _request_metadata_from_generation_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract request metadata from generation kwargs.

        Args:
            gen_kwargs: Merged generation keyword arguments.

        Returns:
            A metadata dictionary potentially containing ``"cache_group"``
            and ``"multimodal_payload"`` keys.
        """
        metadata: dict[str, Any] = {}
        cache_group = gen_kwargs.get("cache_group")
        if isinstance(cache_group, str) and cache_group.strip():
            metadata["cache_group"] = cache_group
        multimodal_payload = gen_kwargs.get("multimodal_payload", gen_kwargs.get("multimodal_inputs"))
        if multimodal_payload is not None:
            metadata["multimodal_payload"] = multimodal_payload
        return metadata

    def _build_requests(
        self,
        prompts: list["str"],
        gen_kwargs: dict[str, Any],
        max_new_tokens: int,
    ) -> list["_RuntimeRequestState"]:
        """Tokenize prompts and construct runtime request states.

        Args:
            prompts: List of prompt strings.
            gen_kwargs: Merged generation keyword arguments.
            max_new_tokens: Maximum generation length.

        Returns:
            A list of :class:`_RuntimeRequestState` objects ready to be
            enqueued into the scheduler.
        """
        _, _, prompt_token_ids = self._prepare_inputs(prompts, max_new_tokens=max_new_tokens)
        eos_token_ids = _normalize_eos_token_ids(gen_kwargs.get("eos_token_id", self.tokenizer.eos_token_id))
        eos_token_ids.update(int(token_id) for token_id in self.extra_eos_token_ids)

        pad_token_id = gen_kwargs.get("pad_token_id")
        if pad_token_id is None:
            if self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            elif eos_token_ids:
                pad_token_id = min(eos_token_ids)

        requests: list["_RuntimeRequestState"] = []
        request_sampling_params = self._sampling_params_from_generation_kwargs(gen_kwargs, max_new_tokens=max_new_tokens)
        request_metadata = self._request_metadata_from_generation_kwargs(gen_kwargs)
        request_priority = int(gen_kwargs.get("priority", 0) or 0)
        for idx, prompt in enumerate(prompts):
            prompt_ids = list(prompt_token_ids[idx])
            tool_parser_instance = self._new_tool_parser()
            reasoning_parser_instance = self._new_reasoning_parser()
            if reasoning_parser_instance is not None:
                try:
                    reasoning_parser_instance.configure_prompt_context(
                        prompt_text=prompt,
                        prompt_token_ids=prompt_ids,
                    )
                except Exception:
                    pass
            request = EngineRequest(
                request_id=str(uuid.uuid4()),
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                sampling_params=request_sampling_params,
                eos_token_id=int(gen_kwargs["eos_token_id"]) if gen_kwargs.get("eos_token_id") is not None else None,
                client_index=idx,
                priority=request_priority,
                metadata=dict(request_metadata),
            )
            requests.append(
                _RuntimeRequestState(
                    request=request,
                    gen_kwargs=dict(gen_kwargs),
                    eos_token_ids=set(eos_token_ids),
                    pad_token_id=int(pad_token_id) if pad_token_id is not None else None,
                    stop=list(gen_kwargs.get("stop")) if gen_kwargs.get("stop") is not None else None,
                    tool_parser_instance=tool_parser_instance,
                    reasoning_parser_instance=reasoning_parser_instance,
                )
            )
        return requests

    def _prepare_multimodal_batch(self, step_output: SchedulerStepOutput) -> Any | None:
        """Invoke the multimodal preprocessor for the current batch.

        Args:
            step_output: The scheduler step output describing scheduled
                sequences.

        Returns:
            Preprocessed multimodal data, or ``None`` if no preprocessor
            is configured or no matching requests exist.
        """
        preprocessor = self._multimodal_preprocessor
        if preprocessor is None:
            return None
        runtime_states = [
            self._runtime_states[entry.request_id]
            for entry in step_output.scheduled
            if entry.request_id in self._runtime_states
        ]
        if not runtime_states:
            return None
        requests = [state.request for state in runtime_states]
        prepare_batch = getattr(preprocessor, "prepare_batch", None)
        if callable(prepare_batch):
            return _call_with_filtered_kwargs(
                prepare_batch,
                requests=requests,
                runtime_states=runtime_states,
                step_output=step_output,
            )
        if callable(preprocessor):
            return _call_with_filtered_kwargs(
                preprocessor,
                requests=requests,
                runtime_states=runtime_states,
                step_output=step_output,
            )
        return None

    def _build_execution_request(self, step_output: SchedulerStepOutput) -> ExecutionRequest:
        """Translate a scheduler step output into an execution request.

        Handles preemption, row binding, and multimodal preparation.

        Args:
            step_output: The scheduler step describing what to run.

        Returns:
            An :class:`ExecutionRequest` for the model runner.

        Raises:
            RuntimeError: If the runtime is not initialized.
        """
        if self._scheduler is None or self._runner is None:
            raise RuntimeError("Background runtime is not initialized.")

        if step_output.preempted_request_ids:
            for request_id in step_output.preempted_request_ids:
                self._runner.detach_request(request_id)
            self._scheduler.remap_rows(self._runner.compact_rows())

        scheduled_list = step_output.scheduled
        num_scheduled = len(scheduled_list)

        if num_scheduled == 0:
            return ExecutionRequest(step_id=self._step_counter)

        is_pure_decode = num_scheduled > 0 and not scheduled_list[0].is_prefill
        if is_pure_decode and num_scheduled == 1:
            entry = scheduled_list[0]
            state = self._runtime_states.get(entry.request_id)
            if state is not None and not state.request.is_finished:
                self._runner.bind_request(state.request, entry.row_index)
                seq = ScheduledSequence(
                    request_id=entry.request_id,
                    row_index=entry.row_index,
                    token_ids=list(entry.token_ids),
                    num_computed_tokens=state.request.num_computed_tokens,
                    page_ids=entry.page_ids,
                    meta={"is_prefill": False},
                )
                sp = state.request.sampling_params
                return ExecutionRequest(
                    step_id=self._step_counter,
                    mode="decode",
                    sequences=[seq],
                    sampling_by_request={entry.request_id: sp} if sp is not None else {},
                    page_table=self._runner.sequence_buffer.page_table(),
                )

        sequences: list[ScheduledSequence] = []
        sampling_by_request: dict[str, SamplingParams] = {}
        has_prefill = False
        for scheduled in scheduled_list:
            state = self._runtime_states.get(scheduled.request_id)
            if state is None or state.request.is_finished:
                continue
            self._runner.bind_request(state.request, scheduled.row_index)
            if scheduled.is_prefill:
                has_prefill = True
            sequences.append(
                ScheduledSequence(
                    request_id=scheduled.request_id,
                    row_index=scheduled.row_index,
                    token_ids=list(scheduled.token_ids),
                    num_computed_tokens=state.request.num_computed_tokens,
                    page_ids=tuple(scheduled.page_ids),
                    meta={
                        "is_prefill": scheduled.is_prefill,
                        "cache_group": scheduled.cache_group,
                        "prefix_cache_hit": scheduled.prefix_cache_hit,
                    },
                )
            )
            if state.request.sampling_params is not None:
                sampling_by_request[scheduled.request_id] = state.request.sampling_params

        if not sequences:
            return ExecutionRequest(step_id=self._step_counter)

        multimodal = self._prepare_multimodal_batch(step_output) if has_prefill else None
        mode = "prefill" if has_prefill and len(sequences) == num_scheduled else "mixed" if has_prefill else "decode"

        return ExecutionRequest(
            step_id=self._step_counter,
            mode=mode,
            sequences=sequences,
            sampling_by_request=sampling_by_request,
            page_table=self._runner.sequence_buffer.page_table(),
            multimodal=multimodal,
        )

    def _determine_stop_reason(self, state: _RuntimeRequestState, sampled_token_ids: list["int"]) -> str | None:
        """Determine why a request should stop generating.

        Checks for stop strings, tool calls, EOS tokens, and length limits.

        Args:
            state: The runtime state for the request.
            sampled_token_ids: Newly sampled token ids to evaluate.

        Returns:
            A stop reason string (``"stop"``, ``"eos"``, ``"length"``,
            ``"tool_calls"``), or ``None`` if generation should continue.
        """
        if not sampled_token_ids:
            return None

        last_token = int(sampled_token_ids[-1])
        if last_token in state.eos_token_ids:
            return "eos"

        total_generated = len(state.generated_token_ids) + len(sampled_token_ids)
        if total_generated >= state.request.max_new_tokens:
            return "length"

        has_stop_strings = bool(state.stop)
        has_tool_parser = state.tool_parser_instance is not None

        if not has_stop_strings and not has_tool_parser:
            return None

        tail_size = min(total_generated, 64)
        tail_ids = list(state.generated_token_ids[-tail_size:]) + [int(t) for t in sampled_token_ids]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)

        if has_stop_strings:
            for stop_str in state.stop:
                if stop_str in tail_text:
                    return "stop"

        if has_tool_parser:
            candidate_generated = list(state.generated_token_ids) + [int(t) for t in sampled_token_ids]
            raw_text = self.tokenizer.decode(candidate_generated, skip_special_tokens=True)
            parsed = self._parse_text_output(
                raw_text,
                stop=state.stop,
                finish_reason=None,
                gen_kwargs=state.gen_kwargs,
                tool_parser=state.tool_parser_instance,
                reasoning_parser=state.reasoning_parser_instance,
            )
            if parsed["tool_calls"]:
                return "tool_calls"

        return None

    def _reconcile_terminal_states_locked(self) -> None:
        """Detach finished requests from the runner and notify waiters.

        Must be called while holding :attr:`_work_cv`. Compacts runner
        rows and remaps the scheduler after detaching.
        """
        if self._runner is None or self._scheduler is None:
            return
        detached = False
        for state in self._runtime_states.values():
            request = state.request
            if not request.is_finished:
                continue
            if self._runner.sequence_buffer.has_request(request.request_id):
                self._runner.detach_request(request.request_id)
                detached = True
            if request.status == EngineRequestStatus.CANCELED:
                state.mark_canceled()
            else:
                state.finalize()
        if detached:
            self._scheduler.remap_rows(self._runner.compact_rows())
        self._has_terminal_states = False

    def _dispatch_distributed_step(self, step_output: SchedulerStepOutput) -> Any | None:
        """Dispatch a step to remote workers via the distributed controller.

        Args:
            step_output: The scheduler step to broadcast.

        Returns:
            A dispatch token, or ``None`` if distributed mode is inactive.
        """
        controller = self._distributed_controller
        if controller is None or not getattr(controller, "enabled", False):
            return None
        dispatch_step = getattr(controller, "dispatch_step", None)
        if callable(dispatch_step):
            return dispatch_step(step_output)
        return None

    def _verify_distributed_step(self, dispatch: Any, model_output: Any) -> None:
        """Verify worker outputs match the leader's model output.

        Args:
            dispatch: Dispatch token from :meth:`_dispatch_distributed_step`.
            model_output: The leader's model execution result.
        """
        controller = self._distributed_controller
        if controller is None or dispatch is None:
            return
        verify_step = getattr(controller, "verify_step", None)
        if callable(verify_step):
            verify_step(dispatch, model_output)

    def _apply_execution_result_locked(
        self,
        step_output: SchedulerStepOutput,
        execution_request: ExecutionRequest,
        result: ExecutionResult | None,
        execution_error: BaseException | None,
    ) -> None:
        """Apply model execution results back to the scheduler and request states.

        Must be called while holding :attr:`_work_cv`. On error, marks all
        affected requests as failed.

        Args:
            step_output: The original scheduler step output.
            execution_request: The execution request that was run.
            result: The execution result, or ``None`` on error.
            execution_error: The exception raised during execution, or
                ``None`` on success.
        """
        if self._scheduler is None or self._runner is None:
            return

        if execution_error is not None:
            failed_requests = {entry.request_id: str(execution_error) for entry in step_output.scheduled}
            self._scheduler.update_from_model_output(step_output, failed_requests=failed_requests)
            for entry in step_output.scheduled:
                state = self._runtime_states.get(entry.request_id)
                if state is None:
                    continue
                state.mark_error(execution_error)
            self._reconcile_terminal_states_locked()
            return

        if result is None:
            return

        sampled_token_ids = {
            update.request_id: list(update.sampled_token_ids) for update in result.updates if update.sampled_token_ids
        }
        stop_reasons: dict[str, str | None] = {}
        for request_id, token_ids in sampled_token_ids.items():
            state = self._runtime_states.get(request_id)
            if state is None or state.is_done:
                continue
            state.note_generated_tokens(token_ids)
            stop_reasons[request_id] = self._determine_stop_reason(state, token_ids)

        finished_requests = self._scheduler.update_from_model_output(
            step_output,
            sampled_token_ids=sampled_token_ids,
            stop_reasons=stop_reasons,
        )
        for finished in finished_requests:
            state = self._runtime_states.get(finished.request_id)
            if state is None:
                continue
            state.finalize()
        if finished_requests:
            self._has_terminal_states = True
            self._reset_recurrent_states()
            self._reconcile_terminal_states_locked()

    def _run_loop(self) -> None:
        """Background loop that continuously schedules and executes steps.

        Runs in a daemon thread until :attr:`_shutdown` is set. Each
        iteration schedules a batch, builds an execution request, runs
        the model, and applies results.
        """
        while not self._shutdown.is_set():
            with self._work_cv:
                while not self._shutdown.is_set() and (
                    self._paused or self._scheduler is None or not self._scheduler.has_pending_work()
                ):
                    self._reconcile_terminal_states_locked()
                    self._work_cv.wait(timeout=0.001)

                if self._shutdown.is_set():
                    break
                if self._paused or self._scheduler is None:
                    continue

                if self._has_terminal_states:
                    self._reconcile_terminal_states_locked()
                step_output = self._scheduler.schedule()
                if not step_output.scheduled:
                    self._work_cv.wait(timeout=0.001)
                    continue
                self._step_counter += 1
                execution_request = self._build_execution_request(step_output)
                dispatch = self._dispatch_distributed_step(step_output) if self._distributed_controller is not None else None

            execution_error: BaseException | None = None
            result: ExecutionResult | None = None
            try:
                if self._execution_manager is None:
                    raise RuntimeError("Execution manager is not initialized.")
                result = self._execution_manager.execute(execution_request)
                self._verify_distributed_step(dispatch, result.output)
            except BaseException as exc:
                execution_error = exc

            with self._work_cv:
                self._apply_execution_result_locked(step_output, execution_request, result, execution_error)
                self._work_cv.notify_all()

    def _enqueue_states(self, states: list["_RuntimeRequestState"]) -> None:
        """Register runtime states and add their requests to the scheduler.

        Args:
            states: List of request states to enqueue.

        Raises:
            RuntimeError: If the runtime is not initialized.
        """
        if self._scheduler is None:
            raise RuntimeError("Background runtime is not initialized.")
        with self._work_cv:
            for state in states:
                self._runtime_states[state.request.request_id] = state
                self._scheduler.add_request(state.request)
            self._work_cv.notify_all()

    def _generate_with_background(
        self,
        prompts: str | Iterable["str"],
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> list[RequestOutput]:
        """Submit prompts to the background loop and block until complete.

        Args:
            prompts: One or more prompt strings.
            sampling_params: Optional sampling parameters.
            **generate_kwargs: Additional generation keyword arguments.

        Returns:
            A list of :class:`RequestOutput` objects, one per prompt.

        Raises:
            NotImplementedError: If ``n > 1`` sampling is requested.
        """
        if sampling_params is not None and sampling_params.n != 1:
            raise NotImplementedError("n>1 sampling is not supported in easymlx eSurge.")

        prompts_list = normalize_prompts(prompts)
        gen_kwargs = self._resolve_generation_kwargs(sampling_params, **generate_kwargs)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", sampling_params.max_tokens if sampling_params else 16))
        if self.auto_cap_new_tokens:
            max_new_tokens = max(1, min(max_new_tokens, self.max_model_len - self.reserve_tokens))
        gen_kwargs["max_new_tokens"] = max_new_tokens
        if gen_kwargs.get("eos_token_id") is None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if gen_kwargs.get("pad_token_id") is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id or gen_kwargs.get("eos_token_id")

        states = self._build_requests(prompts_list, gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return []

        self._enqueue_states(states)

        outputs: list[RequestOutput] = []
        for state in states:
            state.done.wait()
            if state.error is not None:
                raise state.error
            outputs.append(self._state_to_request_output(state))
        with self._work_cv:
            for state in states:
                self._runtime_states.pop(state.request.request_id, None)
        return outputs

    def initiate(self) -> eSurge:
        """Return ``self`` (no-op, present for API compatibility).

        Returns:
            This engine instance.
        """
        return self

    def generate(
        self,
        prompts: str | Iterable["str"],
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts (blocking).

        Args:
            prompts: A single prompt string or an iterable of prompts.
            sampling_params: Optional sampling parameters.
            **generate_kwargs: Additional keyword arguments forwarded to
                the generation pipeline.

        Returns:
            A list of :class:`RequestOutput` objects, one per prompt.
        """
        prompts_list = normalize_prompts(prompts)
        max_tokens = (
            sampling_params.max_tokens
            if sampling_params and sampling_params.max_tokens
            else generate_kwargs.get("max_new_tokens", 16)
        )
        self._info(
            "generate: %d prompt(s), max_tokens=%s",
            len(prompts_list),
            max_tokens,
        )
        return self._generate_with_background(prompts, sampling_params=sampling_params, **generate_kwargs)

    def _generate_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> RequestOutput | None:
        """Synchronous single-request fast path — no thread, no locks.

        Bypasses the background scheduling loop entirely for single-request
        generation. Runs prefill + decode in a tight Python loop on the
        calling thread, eliminating ~3ms/step of threading overhead.

        Returns ``None`` if the fast path cannot be used (e.g. runner not
        initialized), in which case the caller should fall back to the
        background loop.
        """
        if self._runner is None or self._scheduler is None or self._execution_manager is None:
            return None

        import numpy as np

        gen_kwargs = self._resolve_generation_kwargs(sampling_params, **generate_kwargs)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", sampling_params.max_tokens if sampling_params else 16))
        if self.auto_cap_new_tokens:
            max_new_tokens = max(1, min(max_new_tokens, self.max_model_len - self.reserve_tokens))

        eos_token_id = gen_kwargs.get("eos_token_id", self.tokenizer.eos_token_id)
        eos_set: set[int] = set()
        if isinstance(eos_token_id, int):
            eos_set.add(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            eos_set.update(int(e) for e in eos_token_id)

        stop_strings: list[str] = gen_kwargs.get("stop", []) or []
        if sampling_params and sampling_params.stop:
            stop_strings = list(sampling_params.stop)

        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            return None

        runner = self._runner
        scheduler = self._scheduler

        request_id = f"sync_{id(self)}_{time.time()}"
        from easymlx.inference.esurge.request import EngineRequest
        if sampling_params is not None:
            sp_copy = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                do_sample=sampling_params.do_sample,
                stop=sampling_params.stop,
            )
        else:
            sp_copy = SamplingParams(max_tokens=max_new_tokens)
        engine_request = EngineRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=list(input_ids),
            sampling_params=sp_copy,
        )

        scheduler.add_request(engine_request)
        runner.bind_request(engine_request, 0)

        generated_token_ids: list[int] = []
        finish_reason: str | None = None
        t_start = time.time()
        np.random.default_rng()

        step = 0
        while len(generated_token_ids) < max_new_tokens:
            step_output = scheduler.schedule()
            if not step_output.scheduled:
                break

            entry = step_output.scheduled[0]
            seq = ScheduledSequence(
                request_id=request_id,
                row_index=entry.row_index,
                token_ids=list(entry.token_ids),
                num_computed_tokens=engine_request.num_computed_tokens,
                page_ids=entry.page_ids,
                meta={"is_prefill": entry.is_prefill},
            )
            is_prefill = entry.is_prefill
            mode = "prefill" if is_prefill else "decode"

            exec_request = ExecutionRequest(
                step_id=step,
                mode=mode,
                sequences=[seq],
                sampling_by_request={request_id: sampling_params} if sampling_params else {},
                page_table=runner.sequence_buffer.page_table(),
            )

            result = self._execution_manager.execute(exec_request)

            if result.updates:
                update = result.updates[0]
                new_tokens = list(update.sampled_token_ids) if update.sampled_token_ids else []
                generated_token_ids.extend(new_tokens)
                engine_request.generated_token_ids.extend(new_tokens)
                engine_request.num_computed_tokens = (update.num_computed_tokens or engine_request.num_computed_tokens)
                runner.sequence_buffer.append_output_tokens(request_id, new_tokens)
                runner.sequence_buffer.set_num_computed_tokens(request_id, engine_request.num_computed_tokens)

                sampled_ids = {request_id: new_tokens}
                stop_reason: str | None = None

                if new_tokens and new_tokens[-1] in eos_set:
                    stop_reason = "eos"
                elif len(generated_token_ids) >= max_new_tokens:
                    stop_reason = "length"
                elif stop_strings:
                    tail = self.tokenizer.decode(generated_token_ids[-64:], skip_special_tokens=True)
                    for s in stop_strings:
                        if s in tail:
                            stop_reason = "stop"
                            break

                scheduler.update_from_model_output(
                    step_output,
                    sampled_token_ids=sampled_ids,
                    stop_reasons={request_id: stop_reason},
                )

                if stop_reason is not None:
                    finish_reason = stop_reason
                    break
            else:
                scheduler.update_from_model_output(step_output, sampled_token_ids={})

            step += 1

        self._reset_recurrent_states()
        runner.detach_request(request_id)
        runner.compact_rows()

        elapsed = max(time.time() - t_start, 1e-6)
        num_gen = len(generated_token_ids)
        tps = float(num_gen) / elapsed if num_gen else 0.0

        raw_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        parsed = self._parse_text_output(
            raw_text,
            stop=stop_strings,
            finish_reason=finish_reason,
            gen_kwargs=gen_kwargs,
            tool_parser=self._new_tool_parser() if self.tool_parser_name else None,
            reasoning_parser=self._new_reasoning_parser() if self.reasoning_parser_name else None,
        )

        text = str(parsed["text"])
        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=generated_token_ids,
            finish_reason=parsed["finish_reason"] or finish_reason,
            tool_calls=parsed.get("tool_calls"),
            reasoning_content=parsed.get("reasoning_content"),
        )
        return RequestOutput(
            request_id=request_id,
            outputs=[completion],
            prompt=prompt,
            prompt_token_ids=list(input_ids),
            finished=True,
            accumulated_text=text,
            tokens_per_second=tps,
            num_generated_tokens=num_gen,
            time_spent_generating=elapsed,
            processing_time=elapsed,
            reasoning_content=parsed.get("reasoning_content"),
            tool_calls=parsed.get("tool_calls"),
        )

    def stream(
        self,
        prompt: str | Iterable["str"],
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ):
        """Generate completions with incremental streaming yields.

        Yields :class:`RequestOutput` objects as tokens are generated,
        with ``delta_text`` and ``delta_reasoning_content`` populated for
        incremental consumption.

        Args:
            prompt: A single prompt string or an iterable of prompts.
            sampling_params: Optional sampling parameters.
            **generate_kwargs: Additional keyword arguments forwarded to
                the generation pipeline.

        Yields:
            :class:`RequestOutput` with incremental delta fields.

        Raises:
            NotImplementedError: If ``n > 1`` sampling is requested.
        """
        if sampling_params is not None and sampling_params.n != 1:
            raise NotImplementedError("n>1 sampling is not supported in easymlx eSurge.")

        prompts_list = normalize_prompts(prompt)
        gen_kwargs = self._resolve_generation_kwargs(sampling_params, **generate_kwargs)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", sampling_params.max_tokens if sampling_params else 16))
        if self.auto_cap_new_tokens:
            max_new_tokens = max(1, min(max_new_tokens, self.max_model_len - self.reserve_tokens))
        gen_kwargs["max_new_tokens"] = max_new_tokens
        if gen_kwargs.get("eos_token_id") is None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if gen_kwargs.get("pad_token_id") is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id or gen_kwargs.get("eos_token_id")

        states = self._build_requests(prompts_list, gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return

        self._enqueue_states(states)
        pending = {state.request.request_id: state for state in states}

        while pending:
            with self._work_cv:
                self._work_cv.wait_for(
                    lambda: any(
                        state.error is not None
                        or state.update_seq > state.last_emitted_update_seq
                        or (state.is_done and not state.last_emitted_finished)
                        for state in pending.values()
                    ),
                    timeout=0.05,
                )
                ready = [
                    state
                    for state in pending.values()
                    if state.error is not None
                    or state.update_seq > state.last_emitted_update_seq
                    or (state.is_done and not state.last_emitted_finished)
                ]

            for state in ready:
                if state.error is not None:
                    raise state.error
                yield self._state_to_request_output(state, incremental=True, consume=True)
                if state.is_done and state.last_emitted_finished:
                    pending.pop(state.request.request_id, None)

        with self._work_cv:
            for state in states:
                self._runtime_states.pop(state.request.request_id, None)

    def _format_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render chat messages into a prompt string using the tokenizer template.

        Attempts multiple message normalization strategies, falling back
        to a plain ``"role: content"`` format if all template attempts fail.

        Args:
            messages: List of normalized chat message dicts.
            add_generation_prompt: Whether to append a generation prompt
                marker.
            tools: Optional tool definitions passed to the template.
            chat_template: Optional custom chat template string.
            chat_template_kwargs: Extra kwargs forwarded to
                ``apply_chat_template``.

        Returns:
            A single rendered prompt string.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            logger.warning("Tokenizer has no apply_chat_template; falling back to plain text format")
            return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

        normalized_tools = normalize_chat_template_tools(tools)
        template_messages = normalize_chat_template_messages(messages)
        structured_messages = to_structured_text_messages(template_messages)
        extra = dict(chat_template_kwargs or {})
        if chat_template is not None:
            extra["chat_template"] = chat_template

        attempts = (
            (template_messages, normalized_tools),
            (structured_messages, normalized_tools),
            (structured_messages, None),
        )
        for attempt_idx, (candidate_messages, candidate_tools) in enumerate(attempts):
            try:
                result = self.tokenizer.apply_chat_template(
                    candidate_messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    tools=candidate_tools,
                    **extra,
                )
                logger.debug("Chat template rendered on attempt %d", attempt_idx + 1)
                return result
            except TypeError:
                try:
                    result = self.tokenizer.apply_chat_template(
                        candidate_messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        **extra,
                    )
                    logger.debug("Chat template rendered (without tools) on attempt %d", attempt_idx + 1)
                    return result
                except Exception:
                    logger.debug("Chat template attempt %d failed, trying next", attempt_idx + 1)
                    continue
            except Exception:
                logger.debug("Chat template attempt %d failed, trying next", attempt_idx + 1)
                continue
        logger.warning("All chat template attempts failed; falling back to plain text format")
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

    @overload
    def chat(
        self,
        messages: list[dict[str, Any]] | list[tuple[str, str]],
        sampling_params: SamplingParams | None = ...,
        *,
        stream: Literal[False] = ...,
        add_generation_prompt: bool = ...,
        tools: list[dict[str, Any]] | None = ...,
        chat_template: str | None = ...,
        chat_template_kwargs: dict[str, Any] | None = ...,
        **generate_kwargs: Any,
    ) -> RequestOutput: ...

    @overload
    def chat(
        self,
        messages: list[dict[str, Any]] | list[tuple[str, str]],
        sampling_params: SamplingParams | None = ...,
        *,
        stream: Literal[True],
        add_generation_prompt: bool = ...,
        tools: list[dict[str, Any]] | None = ...,
        chat_template: str | None = ...,
        chat_template_kwargs: dict[str, Any] | None = ...,
        **generate_kwargs: Any,
    ) -> Iterator[RequestOutput]: ...

    def chat(
        self,
        messages: list[dict[str, Any]] | list[tuple[str, str]],
        sampling_params: SamplingParams | None = None,
        *,
        stream: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **generate_kwargs: Any,
    ) -> RequestOutput | Iterator[RequestOutput]:
        """Generate a chat completion from a list of messages.

        Renders the messages into a prompt via the tokenizer's chat
        template, then delegates to :meth:`generate` or :meth:`stream`.

        Args:
            messages: Chat messages as dicts (with ``"role"``/``"content"``)
                or 2-tuples of ``(role, content)``.
            sampling_params: Optional sampling parameters.
            stream: If ``True``, return a streaming iterator.
            add_generation_prompt: Whether to append the generation prompt
                marker.
            tools: Optional tool definitions for function-calling models.
            chat_template: Optional custom chat template override.
            chat_template_kwargs: Extra kwargs for the chat template.
            **generate_kwargs: Additional generation keyword arguments.

        Returns:
            A single :class:`RequestOutput` when ``stream=False``, or an
            iterator of incremental :class:`RequestOutput` objects when
            ``stream=True``.
        """
        max_tokens = (
            sampling_params.max_tokens
            if sampling_params and sampling_params.max_tokens
            else generate_kwargs.get("max_new_tokens", 16)
        )
        self._info(
            "chat: %d message(s), stream=%s, max_tokens=%s",
            len(messages),
            stream,
            max_tokens,
        )
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized_messages.append(dict(message))
            else:
                normalized_messages.append({"role": str(message[0]), "content": str(message[1])})

        prompt = self._format_chat_prompt(
            normalized_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        if tools and "tools" not in generate_kwargs:
            generate_kwargs["tools"] = tools
        if stream:
            return self.stream(prompt, sampling_params=sampling_params, **generate_kwargs)
        return self.generate(prompt, sampling_params=sampling_params, **generate_kwargs)[0]

    def abort_request(self, request_id: str) -> bool:
        """Cancel a pending or in-progress request.

        Args:
            request_id: The unique identifier of the request to abort.

        Returns:
            ``True`` if the request was found and canceled, ``False``
            otherwise.
        """
        with self._work_cv:
            if self._scheduler is None:
                return False
            canceled = self._scheduler.cancel_request(request_id)
            if not canceled:
                return False
            state = self._runtime_states.get(request_id)
            if state is not None:
                state.mark_canceled()
            self._reconcile_terminal_states_locked()
            self._work_cv.notify_all()
            return True

    def pause(self) -> None:
        """Pause the background scheduling loop.

        New requests may still be enqueued but will not be scheduled
        until :meth:`resume` is called.
        """
        with self._work_cv:
            self._paused = True
            if self._scheduler is not None:
                self._scheduler.pause()

    def resume(self) -> None:
        """Resume the background scheduling loop after a :meth:`pause`."""
        with self._work_cv:
            self._paused = False
            if self._scheduler is not None:
                self._scheduler.resume()
            self._work_cv.notify_all()

    def terminate(self) -> None:
        """Terminate the engine (alias for :meth:`close`)."""
        self.close()

    def __call__(
        self,
        prompts: str | Iterable["str"],
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> list[RequestOutput]:
        """Shorthand for :meth:`generate` -- makes the engine callable.

        Args:
            prompts: One or more prompt strings.
            sampling_params: Optional sampling parameters.
            **generate_kwargs: Additional generation keyword arguments.

        Returns:
            A list of :class:`RequestOutput` objects.
        """
        return self.generate(prompts, sampling_params=sampling_params, **generate_kwargs)

    def close(self) -> None:
        """Gracefully shut down the engine.

        Signals the background thread to stop, marks all in-flight
        requests as errored, joins the thread, shuts down the execution
        manager, and stops the distributed controller.
        """
        shutdown = getattr(self, "_shutdown", None)
        if shutdown is None or shutdown.is_set():
            return
        logger.info("eSurge engine shutting down")
        shutdown.set()
        with self._work_cv:
            for state in self._runtime_states.values():
                if not state.request.is_finished:
                    state.mark_error(RuntimeError("Engine is shutting down"))
            self._work_cv.notify_all()
        if self._background_thread is not None:
            self._background_thread.join(timeout=1.0)
        if self._execution_manager is not None:
            self._execution_manager.close()
        self._stop_distributed_controller()
        logger.debug("eSurge engine shutdown complete")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        model_class: type[Any] | None = None,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        converted_cache_dir: str | None = None,
        force_conversion: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        **engine_kwargs: Any,
    ) -> eSurge:
        """Load a model from a HuggingFace repo and wrap it in an eSurge engine.

        Automatically resolves the model class from the easymlx registry
        when *model_class* is ``None``.

        Args:
            pretrained_model_name_or_path: HuggingFace model id or local
                path.
            model_class: Optional explicit model class. If ``None``, the
                class is looked up via the easymlx module registry.
            tokenizer: Tokenizer instance or model id. If ``None``, uses
                the same path as the model.
            revision: Git revision for the model/tokenizer.
            local_files_only: If ``True``, only use locally cached files.
            converted_cache_dir: Directory for cached weight conversions.
            force_conversion: Force re-conversion of weights.
            model_kwargs: Extra kwargs passed to the model's
                ``from_pretrained`` method.
            **engine_kwargs: Additional kwargs forwarded to the
                :class:`eSurge` constructor.

        Returns:
            A fully initialized :class:`eSurge` instance.
        """
        model_kwargs = dict(model_kwargs or {})
        model_kwargs.setdefault("auto_convert_hf", True)
        if model_class is None:
            import easymlx.modules  # noqa: F401

            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                local_files_only=local_files_only,
            )
            registration = registry.get_module_registration(TaskType.CAUSAL_LM, str(config.model_type))
            model_class = registration.module

        model = model_class.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            **model_kwargs,
        )
        return cls(model, tokenizer=tokenizer, **engine_kwargs)

    def __del__(self) -> None:  # pragma: no cover
        """Ensure cleanup on garbage collection."""
        self.close()
