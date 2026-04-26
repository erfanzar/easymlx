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

import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from easymlx.inference.openai_api_modules import ChatCompletionRequest, ChatMessage
from easymlx.inference.parsing import DelegatingParser
from easymlx.inference.reasoning import ReasoningParserManager, detect_reasoning_parser
from easymlx.inference.stream_protocol import compute_stream_delta_text
from easymlx.inference.tools import ToolParserManager, detect_tool_parser
from easymlx.infra.etils import canonical_attention_mechanism
from easymlx.infra.factory import TaskType, registry
from easymlx.utils.hf_composite import resolve_hf_composite_repo
from easymlx.workers.loggers import get_logger

from .config import CacheConfig, Config, SchedulerConfig, SpeculativeConfig
from .engine_types import FinishReason
from .outputs import CompletionOutput, RequestOutput
from .request import EngineRequest, EngineRequestStatus
from .runners import (
    ExecutionManager,
    ExecutionRequest,
    ExecutionResult,
    ModelRunner,
    PendingExecution,
    ScheduledSequence,
    SequenceBuffer,
)
from .sampling_params import SamplingParams
from .scheduler import AsyncScheduler, ScheduledRequest, Scheduler, SchedulerStepOutput
from .utils import (
    normalize_chat_template_messages,
    normalize_chat_template_tools,
    normalize_prompts,
    normalize_stop_sequences,
    to_structured_text_messages,
    truncate_tokens,
)

logger = get_logger("eSurge")
_REPLACEMENT_CHARACTER = "\ufffd"
_DFLASH_ADAPTIVE_MIN_ATTEMPTS = 4
_DFLASH_ADAPTIVE_MIN_ACCEPTANCE_RATE = 0.90


def _strip_unstable_replacement_suffix(text: str) -> str:
    """Drop unresolved trailing replacement characters from decoded text.

    Fast tokenizers can emit ``U+FFFD`` while a multi-byte byte-fallback
    sequence is still incomplete. Streaming those transient markers makes
    later decodes stop sharing the previous emitted prefix, which in turn
    causes duplicated output when consumers append deltas.
    """
    return text.rstrip(_REPLACEMENT_CHARACTER)


def _plain_chat_prompt(messages: list[dict[str, Any]], *, add_generation_prompt: bool = True) -> str:
    """Render chat messages without a tokenizer-owned chat template."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", part.get("content", ""))
                    if text is not None:
                        parts.append(str(text))
                elif part is not None:
                    parts.append(str(part))
            text_content = "\n".join(part for part in parts if part)
        elif content is None:
            text_content = ""
        else:
            text_content = str(content)
        lines.append(f"{role}: {text_content}" if text_content else f"{role}:")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


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
    return text[:cutoff], FinishReason.STOP


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
        tool_parser_request: Request context passed to tool parsers.
        delegating_parser: Stateful parser coordinating reasoning/tool
            streaming phases.
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
    tool_parser_request: ChatCompletionRequest | None = None
    delegating_parser: DelegatingParser | None = None
    parser_previous_text: str = ""
    parser_previous_token_ids: list[int] = field(default_factory=list)

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

    def decoded_text(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        stream_safe: bool = False,
    ) -> str:
        """Decode the generated token ids into a string.

        Args:
            tokenizer: The tokenizer used for decoding.
            stream_safe: Whether to drop trailing transient replacement
                characters that can appear while a byte-fallback sequence
                is still incomplete.

        Returns:
            The decoded text with special tokens removed.
        """
        text = tokenizer.decode(self.request.generated_token_ids, skip_special_tokens=True)
        return _strip_unstable_replacement_suffix(text) if stream_safe else text

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


@dataclass(slots=True)
class _MemoryUtilizationSummary:
    """Estimated memory-utilization state derived from the allocated KV cache.

    Attributes:
        requested_utilization: User-requested memory utilization fraction.
        allocated_utilization: Approximate fraction of device memory used by
            the allocated KV cache.
        estimated_token_capacity: Approximate total cache-token capacity
            implied by the requested memory utilization, derived from the
            current allocation.
        runtime_token_capacity: Total runtime token cap from
            ``max_model_len * max_num_seqs``.
        requested_budget_bytes: Requested cache-memory budget in bytes.
        allocated_cache_bytes: Estimated bytes consumed by the allocated KV
            caches.
        available_memory_bytes: Device memory budget used for the estimate.
        runtime_capped: Whether runtime caps are lower than the requested
            memory-based estimate.
        approximate_sequence_capacity: Whether the requested memory budget is
            tighter than the runtime token cap, making usable per-sequence
            length workload-dependent.
    """

    requested_utilization: float
    allocated_utilization: float | None
    estimated_token_capacity: int | None
    runtime_token_capacity: int
    requested_budget_bytes: int | None
    allocated_cache_bytes: int | None
    available_memory_bytes: int | None
    runtime_capped: bool = False
    approximate_sequence_capacity: bool = False


@dataclass(slots=True)
class _SpeculativeDecodeStats:
    """Counters for one speculative decoding request."""

    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    target_steps: int = 0
    fallback_tokens: int = 0
    fallback_triggered: bool = False
    fallback_reason: str | None = None

    @property
    def acceptance_attempts(self) -> int:
        """Return verified speculative accept/reject attempts."""
        return self.accepted_tokens + self.rejected_tokens

    @property
    def acceptance_rate(self) -> float:
        """Return the current speculative acceptance rate."""
        attempts = self.acceptance_attempts
        return float(self.accepted_tokens) / float(attempts) if attempts else 0.0

    def as_metrics(self, *, enabled: bool, num_speculative_tokens: int, method: str = "draft") -> dict[str, Any]:
        """Serialize speculative counters into the request metrics map."""
        return {
            "speculative_decoding": bool(enabled),
            "speculative_method": str(method),
            "num_speculative_tokens": int(num_speculative_tokens),
            "speculative_draft_tokens": int(self.draft_tokens),
            "speculative_accepted_tokens": int(self.accepted_tokens),
            "speculative_rejected_tokens": int(self.rejected_tokens),
            "speculative_target_steps": int(self.target_steps),
            "speculative_acceptance_rate": self.acceptance_rate,
            "speculative_fallback_triggered": bool(self.fallback_triggered),
            "speculative_fallback_tokens": int(self.fallback_tokens),
            "speculative_fallback_reason": self.fallback_reason,
        }


@dataclass(slots=True)
class _SpeculativeCacheState:
    """Per-request caches for fast draft-model speculative decoding."""

    target_cache_views: list[Any]
    target_decode_state: Any | None
    target_next_logits: mx.array
    draft_cache_views: list[Any]
    draft_decode_state: Any | None
    draft_next_logits: mx.array


@dataclass(slots=True)
class _DFlashSpeculativeCacheState:
    """Per-request caches for cached DFlash speculative decoding."""

    target_cache_views: list[Any]
    target_decode_state: Any | None
    target_next_logits: mx.array
    dflash_cache: list[Any]
    dflash_hidden_states: tuple[mx.array, ...]
    pending_token: int | None = None
    adaptive_disabled: bool = False


@dataclass(slots=True)
class _CachedDraftProposal:
    """Draft tokens plus the temporary cache state after proposing them."""

    tokens: list[int]
    cache_views: list[Any]
    decode_state: Any | None
    next_logits: mx.array


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
        model: str | Any,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        *,
        dtype: Any | None = None,
        max_model_len: int = 4096,
        min_input_pad: int = 16,
        min_token_pad: int | None = None,
        max_num_seqs: int = 1,
        max_num_seq_buckets: list[int] | None = None,
        async_scheduling: bool = True,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float = 0.45,
        page_size: int = 128,
        use_aot_forward: bool = True,
        bind_graphstate_for_aot: bool = False,
        enable_prefix_caching: bool = True,
        auto_shard_model: bool = True,
        sharding_axis_dims: tuple[int, ...] = (1, 1, 1, -1, 1),
        compile_runner: bool = True,
        runner_verbose: bool = False,
        overlap_execution: bool = True,
        sampler_metrics: bool = False,
        data_parallelism_axis: str = "dp",
        esurge_name: str | None = None,
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
        silent_mode: bool = False,
        processor: Any | None = None,
        config: Config | None = None,
        speculative_model: str | Any | None = None,
        num_speculative_tokens: int | None = None,
        speculative_method: Literal["draft", "dflash", "eagle3"] | None = None,
        eagle3_feature_layer_indices: Iterable[int] | None = None,
        distributed_controller: DistributedControllerProtocol | None = None,
        multimodal_preprocessor: MultimodalBatchPreprocessorProtocol | Any | None = None,
        long_prefill_token_threshold: int | None = None,
        distributed_mode: bool = False,
        distributed_role: Literal["auto", "leader", "worker"] = "auto",
        distributed_service_name: str | None = None,
        distributed_world_size: int | None = None,
        distributed_rank: int | None = None,
        distributed_control_port: int = 19666,
        distributed_control_bind_host: str = "0.0.0.0",
        distributed_advertise_addr: str | None = None,
        distributed_auth_token: str | None = None,
        distributed_step_timeout_s: float = 30.0,
        distributed_connect_timeout_s: float = 15.0,
        distributed_verify_sampling_digest: bool = True,
        enable_window_aware_runtime_cap: bool = False,
        destroy_pages_on_pause: bool = True,
        detokenizer_max_states: int | None = None,
        tokenizer_endpoint: str | None = None,
        detokenizer_endpoint: str | None = None,
        worker_startup_timeout: float | None = None,
        max_request_outputs: int | None = 1000,
        idle_reset_seconds: float | None = None,
        idle_reset_min_interval: float = 60.0,
        sampling_params_callback: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        vision_cache_capacity_mb: int = 1024,
        seed: int = 0,
        **_kwargs: Any,
    ):
        del destroy_pages_on_pause, detokenizer_max_states, max_request_outputs
        del idle_reset_seconds, idle_reset_min_interval, sampling_params_callback
        del resolution_buckets, vision_cache_capacity_mb

        deprecated_memory_utilization = _kwargs.pop("memory_utilization", None)
        if deprecated_memory_utilization is not None:
            logger.warning("`memory_utilization` is deprecated for eSurge; use `hbm_utilization` instead.")
            hbm_utilization = float(deprecated_memory_utilization)

        if isinstance(model, str):
            model_class = _kwargs.pop("model_class", None)
            revision = _kwargs.pop("revision", None)
            local_files_only = bool(_kwargs.pop("local_files_only", False))
            converted_cache_dir = _kwargs.pop("converted_cache_dir", None)
            force_conversion = bool(_kwargs.pop("force_conversion", False))
            model_kwargs = dict(_kwargs.pop("model_kwargs", {}) or {})
            model = self._load_pretrained_model(
                model,
                model_class=model_class,
                revision=revision,
                local_files_only=local_files_only,
                converted_cache_dir=converted_cache_dir,
                force_conversion=force_conversion,
                model_kwargs=model_kwargs,
            )

        configured_speculative = config.speculative_config if config is not None else None
        if speculative_model is None and configured_speculative is not None:
            speculative_model = configured_speculative.speculative_model
        if num_speculative_tokens is None and configured_speculative is not None:
            num_speculative_tokens = configured_speculative.num_speculative_tokens
        if speculative_method is None and configured_speculative is not None:
            speculative_method = configured_speculative.speculative_method
        if eagle3_feature_layer_indices is None and configured_speculative is not None:
            eagle3_feature_layer_indices = configured_speculative.eagle3_feature_layer_indices
        if num_speculative_tokens is None:
            num_speculative_tokens = 0
        if speculative_method is None:
            speculative_method = "draft"

        if isinstance(speculative_model, str):
            speculative_model_kwargs = dict(_kwargs.pop("speculative_model_kwargs", {}) or {})
            if self._should_load_dflash_speculative_model(
                speculative_model,
                speculative_method=speculative_method,
                model_kwargs=speculative_model_kwargs,
            ):
                speculative_method = "dflash"
                speculative_model = self._load_dflash_speculative_model(
                    speculative_model,
                    revision=_kwargs.pop("speculative_revision", None),
                    local_files_only=bool(_kwargs.pop("speculative_local_files_only", False)),
                    model_kwargs=speculative_model_kwargs,
                )
            else:
                speculative_model = self._load_pretrained_model(
                    speculative_model,
                    model_class=_kwargs.pop("speculative_model_class", None),
                    revision=_kwargs.pop("speculative_revision", None),
                    local_files_only=bool(_kwargs.pop("speculative_local_files_only", False)),
                    converted_cache_dir=_kwargs.pop("speculative_converted_cache_dir", None),
                    force_conversion=bool(_kwargs.pop("speculative_force_conversion", False)),
                    model_kwargs=speculative_model_kwargs,
                )

        distributed_active = bool(
            distributed_mode
            or (distributed_controller is not None and getattr(distributed_controller, "enabled", False))
        )
        if distributed_active and overlap_execution:
            raise ValueError(
                "`overlap_execution=True` is not supported with distributed lockstep mode. "
                "Use overlap_execution=False for multi-host serving."
            )

        unsupported_options = {
            "use_aot_forward": use_aot_forward is not True,
            "bind_graphstate_for_aot": bool(bind_graphstate_for_aot),
            "auto_shard_model": auto_shard_model is not True,
            "sharding_axis_dims": tuple(sharding_axis_dims) != (1, 1, 1, -1, 1),
            "sampler_metrics": bool(sampler_metrics),
            "data_parallelism_axis": data_parallelism_axis != "dp",
            "enable_window_aware_runtime_cap": bool(enable_window_aware_runtime_cap),
            "distributed_mode": bool(distributed_mode),
            "tokenizer_endpoint": tokenizer_endpoint is not None,
            "detokenizer_endpoint": detokenizer_endpoint is not None,
            "worker_startup_timeout": worker_startup_timeout is not None,
        }
        unsupported_enabled = [name for name, enabled in unsupported_options.items() if enabled]
        if unsupported_enabled:
            logger.warning(
                "Ignoring EasyDeL/JAX-only eSurge options in MLX runtime: %s",
                ", ".join(sorted(unsupported_enabled)),
            )

        scheduler_config = (
            config.scheduler_config
            if config is not None
            else SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seq_buckets=tuple(max_num_seq_buckets) if max_num_seq_buckets else None,
                async_scheduling=bool(async_scheduling),
                long_prefill_token_threshold=long_prefill_token_threshold,
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
        speculative_config = SpeculativeConfig(
            num_speculative_tokens=max(int(num_speculative_tokens), 0),
            speculative_model=speculative_model,
            speculative_method=speculative_method,
            eagle3_feature_layer_indices=(
                tuple(int(idx) for idx in eagle3_feature_layer_indices)
                if eagle3_feature_layer_indices is not None
                else None
            ),
        )
        if config is not None:
            config.speculative_config = speculative_config
        self.config = config or Config(
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            speculative_config=speculative_config,
        )

        if reserve_tokens is None:
            reserve_tokens = max(1, int(max_model_len) // 10)
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
        self.hbm_utilization = float(hbm_utilization)
        self.memory_utilization = float(hbm_utilization)
        self.effective_memory_utilization = float(self.memory_utilization)
        self.estimated_memory_token_capacity: int | None = None
        self.runtime_dtype = self._infer_runtime_dtype(dtype)
        self.kv_cache_dtype = self.runtime_dtype
        self.silent_mode = bool(silent_mode)
        self.runner_verbose = bool(runner_verbose)
        self.min_input_pad = max(int(min_input_pad), 1)
        self.min_token_pad = max(int(min_token_pad), 1) if min_token_pad is not None else None
        self.async_scheduling = bool(async_scheduling)
        self.overlap_execution = bool(overlap_execution)
        self.compile_runner = bool(compile_runner)
        self.esurge_name = esurge_name
        self.distributed_mode = bool(distributed_mode)
        self.distributed_role = distributed_role
        self.distributed_service_name = distributed_service_name
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        self.distributed_control_port = int(distributed_control_port)
        self.distributed_control_bind_host = str(distributed_control_bind_host)
        self.distributed_advertise_addr = distributed_advertise_addr
        self.distributed_auth_token = distributed_auth_token
        self.distributed_step_timeout_s = float(distributed_step_timeout_s)
        self.distributed_connect_timeout_s = float(distributed_connect_timeout_s)
        self.distributed_verify_sampling_digest = bool(distributed_verify_sampling_digest)
        self._info = logger.info if not self.silent_mode else (lambda *a, **kw: None)
        self._log_it = logger.info if self.runner_verbose else logger.debug
        self.tokenizer = self._resolve_tokenizer(tokenizer or processor)
        self._rng_seed = int(seed)
        self.speculative_config = speculative_config
        self._speculative_model = speculative_model
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
        self._memory_utilization_summary: _MemoryUtilizationSummary | None = None

        requested_tool_parser = tool_parser
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
        self._tool_parser_autodetected = requested_tool_parser is None and tool_parser is not None
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

            algo_str = "attention=unified_attention"

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
                self._format_memory_utilization_line(),
                f"Runtime       : max_model_len={self.max_model_len:,} | "
                f"max_num_seqs={self.max_num_seqs} | "
                f"max_batched_tokens={self.scheduler_config.max_num_batched_tokens or self.max_model_len:,}",
                f"Tool parser   : {tool_parser or 'none'}",
                f"Reason parser : {reasoning_parser or 'none'}",
                "Speculative   : "
                + (
                    f"method={self.speculative_config.speculative_method} | "
                    f"model={type(self._speculative_model).__name__} | "
                    f"tokens={self.speculative_config.num_speculative_tokens}"
                    if self._speculative_model is not None and self.speculative_config.num_speculative_tokens > 0
                    else "none"
                ),
            ]
            memory_note = self._format_memory_utilization_note()
            if memory_note is not None:
                lines.insert(7, memory_note)
            self._info("\n".join(lines))
        except Exception as exc:
            logger.debug("Could not generate startup summary: %s", exc)

    def _initialize_runtime(self) -> None:
        """Bootstrap the model runner, scheduler, execution manager, and background thread."""
        kv_caches = self._init_paged_cache()
        self._memory_utilization_summary = self._build_memory_utilization_summary(kv_caches)
        if self._memory_utilization_summary is not None:
            if self._memory_utilization_summary.allocated_utilization is not None:
                self.effective_memory_utilization = self._memory_utilization_summary.allocated_utilization
            if self._memory_utilization_summary.estimated_token_capacity is not None:
                self.estimated_memory_token_capacity = self._memory_utilization_summary.estimated_token_capacity
        self._reset_recurrent_states()
        self._info("Initializing paged runtime with %d KV cache layers", len(kv_caches))
        supports_async_overlap = bool(
            self.async_scheduling
            and self.overlap_execution
            and not (
                self._distributed_controller is not None and getattr(self._distributed_controller, "enabled", False)
            )
        )
        self._runner = ModelRunner(
            self.model,
            sequence_buffer=SequenceBuffer(max_num_rows=self.max_num_seqs),
            kv_caches=kv_caches,
            max_num_seq_buckets=self.scheduler_config.max_num_seq_buckets,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            memory_utilization=self.memory_utilization,
            min_input_pad=self.min_input_pad,
            min_token_pad=self.min_token_pad,
            supports_async_overlap=supports_async_overlap,
            verbose=self.runner_verbose,
            seed=self._rng_seed,
            use_compiled_forward=self.compile_runner,
        )
        if self.compile_runner:
            self._runner.precompile()
            self._reset_recurrent_states()
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

    @staticmethod
    def _load_pretrained_model(
        pretrained_model_name_or_path: str,
        *,
        model_class: type[Any] | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        converted_cache_dir: str | None = None,
        force_conversion: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Load an EasyMLX causal-LM model for constructor string inputs."""
        model_kwargs = dict(model_kwargs or {})
        model_kwargs.setdefault("auto_convert_hf", True)
        resolved_repo = resolve_hf_composite_repo(
            pretrained_model_name_or_path,
            task_type=TaskType.CAUSAL_LM,
            revision=revision,
            local_files_only=local_files_only,
        )
        model_subfolder = resolved_repo.model_subfolder
        if model_class is None:

            config_kwargs: dict[str, Any] = {
                "revision": revision,
                "local_files_only": local_files_only,
            }
            if model_subfolder is not None:
                config_kwargs["subfolder"] = model_subfolder
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
            )
            model_type = str(config.model_type)
            if model_type == "qwen3_5" and hasattr(config, "text_config"):
                model_type = "qwen3_5_text"
            registration = registry.get_module_registration(TaskType.CAUSAL_LM, model_type)
            model_class = registration.module

        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            subfolder=model_subfolder,
            **model_kwargs,
        )

    @staticmethod
    def _should_load_dflash_speculative_model(
        pretrained_model_name_or_path: str,
        *,
        speculative_method: str | None,
        model_kwargs: dict[str, Any],
    ) -> bool:
        """Return whether a speculative checkpoint should use the DFlash loader."""
        if speculative_method == "dflash":
            return True
        kind = model_kwargs.pop("speculative_kind", None) or model_kwargs.pop("draft_kind", None)
        if isinstance(kind, str) and kind.strip().lower() == "dflash":
            return True
        return "dflash" in str(pretrained_model_name_or_path).lower()

    @staticmethod
    def _load_dflash_speculative_model(
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Load a DFlash draft model instead of a standard causal LM."""
        from .dflash import load_dflash_draft_model

        model_kwargs = dict(model_kwargs or {})
        ignored = {
            key: model_kwargs.pop(key)
            for key in ("auto_convert_hf", "strict", "copy_support_files")
            if key in model_kwargs
        }
        if ignored:
            logger.debug("Ignoring standard model kwargs for DFlash draft load: %s", ", ".join(sorted(ignored)))
        return load_dflash_draft_model(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            **model_kwargs,
        )

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
            model_id = getattr(self.model, "tokenizer_name_or_path", None) or getattr(self.model, "name_or_path", None)
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

    @staticmethod
    def _normalize_mx_floating_dtype(dtype: Any) -> mx.Dtype | None:
        """Normalize common dtype spellings to an MLX floating dtype."""
        if dtype is None:
            return None
        if isinstance(dtype, mx.Dtype):
            try:
                return dtype if mx.issubdtype(dtype, mx.floating) else None
            except TypeError:
                return None
        text = str(dtype).lower().replace("torch.", "").replace("mlx.core.", "").replace("numpy.", "")
        if "bfloat16" in text or "bf16" in text:
            return mx.bfloat16
        if "float16" in text or "fp16" in text or text.endswith("half") or " half" in text:
            return mx.float16
        if "float32" in text or "fp32" in text or text.endswith("float") or " float" in text:
            return mx.float32
        return None

    @classmethod
    def _first_declared_dtype(cls, owner: Any, attr_names: tuple[str, ...]) -> mx.Dtype | None:
        """Return the first normalized dtype declared on *owner*."""
        for attr_name in attr_names:
            value = getattr(owner, attr_name, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    pass
            normalized = cls._normalize_mx_floating_dtype(value)
            if normalized is not None:
                return normalized
        return None

    @staticmethod
    def _iter_runtime_subconfigs(config: Any) -> tuple[Any, ...]:
        """Return text/runtime sub-configs that better describe activation dtype."""
        subconfigs: list[Any] = []
        for attr_name in ("text_config", "language_config", "llm_config", "decoder_config", "model_config"):
            subconfig = getattr(config, attr_name, None)
            if subconfig is not None and subconfig is not config:
                subconfigs.append(subconfig)
        return tuple(subconfigs)

    def _infer_runtime_dtype(self, explicit_dtype: Any | None) -> mx.Dtype:
        """Infer the floating dtype used by runtime KV tensors."""
        normalized = self._normalize_mx_floating_dtype(explicit_dtype)
        if normalized is not None:
            return normalized

        runtime_attr_names = ("runtime_dtype", "activation_dtype", "compute_dtype")
        normalized = self._first_declared_dtype(self.model, runtime_attr_names)
        if normalized is not None:
            return normalized

        config = getattr(self.model, "config", None)
        if config is not None:
            normalized = self._first_declared_dtype(config, runtime_attr_names)
            if normalized is not None:
                return normalized

            for subconfig in self._iter_runtime_subconfigs(config):
                normalized = self._first_declared_dtype(subconfig, runtime_attr_names)
                if normalized is not None:
                    return normalized

            for subconfig in self._iter_runtime_subconfigs(config):
                normalized = self._first_declared_dtype(subconfig, ("mlx_dtype", "dtype", "torch_dtype"))
                if normalized is not None:
                    return normalized

        parameter_dtype_counts: dict[mx.Dtype, int] = {}
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                for _name, value in tree_flatten(parameters()):
                    value_dtype = getattr(value, "dtype", None)
                    normalized = self._normalize_mx_floating_dtype(value_dtype)
                    if normalized is not None:
                        parameter_dtype_counts[normalized] = parameter_dtype_counts.get(normalized, 0) + 1
            except Exception as exc:
                logger.debug("Could not infer cache dtype from model parameters: %s", exc)
        low_precision_counts = {
            dtype: count for dtype, count in parameter_dtype_counts.items() if dtype in (mx.bfloat16, mx.float16)
        }
        if low_precision_counts:
            preference = {mx.bfloat16: 1, mx.float16: 0}
            return max(low_precision_counts, key=lambda dtype: (low_precision_counts[dtype], preference[dtype]))

        if config is not None:
            normalized = self._first_declared_dtype(config, ("mlx_dtype", "dtype", "torch_dtype"))
            if normalized is not None:
                return normalized

        if parameter_dtype_counts.get(mx.float32):
            return mx.float32

        normalized = self._normalize_mx_floating_dtype(getattr(self.model, "dtype", None))
        return normalized or mx.float16

    def _reset_recurrent_states(self, model: Any | None = None) -> None:
        """Reset linear attention / recurrent states on the model.

        Unlike paged KV caches which are managed by the page pool,
        recurrent states (conv_state, recurrent_state) live on the model
        modules themselves. They must be zeroed when requests finish so
        the next request starts clean.
        """
        model = self.model if model is None else model
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
        batch_size = max(int(getattr(self, "max_num_seqs", 1) or 1), 1)
        for layer in layers:
            for attr_name in ("linear_attn", "recurrent", "ssm"):
                module = getattr(layer, attr_name, None)
                if module is not None and hasattr(module, "reset_state"):
                    module.reset_state(batch_size=batch_size)

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
                dtype=self.kv_cache_dtype,
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
            self._disable_unused_packed_page_cache(cache_list)
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
            dtype=self.kv_cache_dtype,
        )
        cache_list = list(caches)
        if not cache_list:
            raise ValueError("Paged cache initialization returned no caches.")
        self._disable_unused_packed_page_cache(cache_list)
        return cache_list

    def _uses_page_attention_backend(self) -> bool:
        """Return whether the model is configured for packed page-attention kernels."""
        configs: list[Any] = []
        config = getattr(self.model, "config", None)
        if config is not None:
            configs.append(config)
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                configs.append(text_config)
        model_config = getattr(self.model, "model_config", None)
        if model_config is not None:
            configs.append(model_config)

        for cfg in configs:
            mechanism = canonical_attention_mechanism(getattr(cfg, "attn_mechanism", None))
            if mechanism == "page_attention":
                return True
        return False

    def _disable_unused_packed_page_cache(self, cache_list: list[Any]) -> None:
        """Skip packed page-cache maintenance when unified attention is selected."""
        if self._uses_page_attention_backend():
            return
        for cache in cache_list:
            if getattr(cache, "cache_dtype_is_turboquant", False):
                continue
            if hasattr(cache, "page_key_cache"):
                cache.page_key_cache = None
            if hasattr(cache, "page_value_cache"):
                cache.page_value_cache = None
            if hasattr(cache, "page_vec_size"):
                cache.page_vec_size = 0

    @staticmethod
    def _sum_nbytes(value: Any, seen: set[int]) -> int:
        """Recursively sum ``nbytes`` for arrays nested inside *value*."""
        if value is None:
            return 0
        if isinstance(value, (str, bytes, bytearray, int, float, bool)):
            return 0

        value_id = id(value)
        if value_id in seen:
            return 0

        if isinstance(value, dict):
            seen.add(value_id)
            return sum(eSurge._sum_nbytes(item, seen) for item in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(value_id)
            return sum(eSurge._sum_nbytes(item, seen) for item in value)

        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            seen.add(value_id)
            return int(nbytes)

        total = 0
        slots = getattr(type(value), "__slots__", ())
        if slots:
            seen.add(value_id)
            for name in slots:
                if isinstance(name, str) and hasattr(value, name):
                    total += eSurge._sum_nbytes(getattr(value, name), seen)
            return total

        try:
            attrs = vars(value)
        except TypeError:
            return 0

        seen.add(value_id)
        return sum(eSurge._sum_nbytes(item, seen) for item in attrs.values())

    @staticmethod
    def _available_device_memory_bytes() -> int | None:
        """Return the device-memory budget used for cache-utilization estimates."""
        try:
            device_info = mx.device_info()
        except Exception:
            try:
                device_info = mx.metal.device_info()
            except Exception:
                return None

        recommended = int(device_info.get("max_recommended_working_set_size") or 0)
        total = int(device_info.get("memory_size") or 0)
        candidates = [value for value in (recommended, total) if value > 0]
        if not candidates:
            return None
        return min(candidates)

    def _build_memory_utilization_summary(self, kv_caches: list[Any]) -> _MemoryUtilizationSummary | None:
        """Estimate requested versus allocated cache utilization.

        The estimate is derived from the actual KV-cache allocation and the
        current device-memory budget. It is intentionally approximate and is
        used for startup reporting only.
        """
        available_memory_bytes = self._available_device_memory_bytes()
        if available_memory_bytes is None or available_memory_bytes <= 0:
            return None

        allocated_cache_bytes = self._sum_nbytes(kv_caches, seen=set())
        runtime_token_capacity = max(self.max_model_len * self.max_num_seqs, 0)
        if allocated_cache_bytes <= 0 or runtime_token_capacity <= 0:
            return _MemoryUtilizationSummary(
                requested_utilization=self.memory_utilization,
                allocated_utilization=None,
                estimated_token_capacity=None,
                runtime_token_capacity=runtime_token_capacity,
                requested_budget_bytes=None,
                allocated_cache_bytes=allocated_cache_bytes or None,
                available_memory_bytes=available_memory_bytes,
            )

        requested_budget_bytes = max(int(available_memory_bytes * self.memory_utilization), 0)
        allocated_utilization = allocated_cache_bytes / available_memory_bytes
        estimated_token_capacity = None
        if requested_budget_bytes > 0:
            estimated_token_capacity = max(
                int((runtime_token_capacity * requested_budget_bytes) / allocated_cache_bytes),
                0,
            )

        runtime_capped = estimated_token_capacity is not None and estimated_token_capacity >= runtime_token_capacity
        approximate_sequence_capacity = (
            estimated_token_capacity is not None and estimated_token_capacity < runtime_token_capacity
        )

        return _MemoryUtilizationSummary(
            requested_utilization=self.memory_utilization,
            allocated_utilization=allocated_utilization,
            estimated_token_capacity=estimated_token_capacity,
            runtime_token_capacity=runtime_token_capacity,
            requested_budget_bytes=requested_budget_bytes,
            allocated_cache_bytes=allocated_cache_bytes,
            available_memory_bytes=available_memory_bytes,
            runtime_capped=runtime_capped,
            approximate_sequence_capacity=approximate_sequence_capacity,
        )

    def _format_memory_utilization_line(self) -> str:
        """Return the startup-summary line for memory utilization."""
        summary = self._memory_utilization_summary
        if summary is None:
            return f"Memory util   : requested={self.memory_utilization:.0%}"

        parts = [f"requested={summary.requested_utilization:.0%}"]
        if summary.allocated_utilization is not None:
            parts.append(f"allocated~{summary.allocated_utilization:.0%}")
        if summary.estimated_token_capacity is not None:
            parts.append(f"est_capacity~{summary.estimated_token_capacity:,} tok")
        return f"Memory util   : {' | '.join(parts)}"

    def _format_memory_utilization_note(self) -> str | None:
        """Return an explanatory memory-summary note when the estimate is useful."""
        summary = self._memory_utilization_summary
        if summary is None or summary.estimated_token_capacity is None:
            return None

        runtime_cap = summary.runtime_token_capacity
        if summary.runtime_capped:
            return (
                "Memory note   : runtime cap "
                f"{runtime_cap:,} tok ({self.max_model_len:,} x {self.max_num_seqs}) "
                f"is below the ~{summary.estimated_token_capacity:,} tok implied by requested "
                "memory_utilization, so the effective allocation is reduced to the runtime cap."
            )

        if summary.approximate_sequence_capacity:
            return (
                "Memory note   : requested memory_utilization implies ~"
                f"{summary.estimated_token_capacity:,} total cache tok, below the runtime cap of "
                f"{runtime_cap:,}. This is not an exact per-sequence limit: usable sequence "
                "length depends on concurrent sequence lengths and page rounding."
            )
        return None

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

    @staticmethod
    def _tool_choice_is_none(tool_choice: Any) -> bool:
        """Return whether the request explicitly disabled tool calling."""
        return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"

    def _should_enable_tool_parser_for_request(self, gen_kwargs: dict[str, Any]) -> bool:
        """Whether a request should instantiate / use a tool parser."""
        if self._tool_parser_class is None:
            return False
        if self._tool_choice_is_none(gen_kwargs.get("tool_choice")):
            return False
        if not self._tool_parser_autodetected:
            return True
        return bool(gen_kwargs.get("tools"))

    def _build_tool_parser_request(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatCompletionRequest | None:
        """Build the parser-side chat request used for schema-aware tool parsing."""
        if tools is None and tool_choice is None:
            return None

        normalized_tools: list[dict[str, Any]] | None = None
        if tools is not None:
            normalized_tools = []
            for tool in tools:
                if hasattr(tool, "model_dump"):
                    raw_tool = tool.model_dump(exclude_none=True)
                elif isinstance(tool, dict):
                    raw_tool = dict(tool)
                else:
                    continue

                function_payload = raw_tool.get("function")
                if isinstance(function_payload, dict):
                    normalized_tools.append(
                        {
                            "type": str(raw_tool.get("type") or "function"),
                            "function": function_payload,
                        }
                    )
                    continue

                if isinstance(raw_tool.get("name"), str):
                    normalized_tools.append(
                        {
                            "type": "function",
                            "function": raw_tool,
                        }
                    )

        return ChatCompletionRequest(
            model=str(self.esurge_name or "easymlx"),
            messages=[ChatMessage(role="user", content=prompt)],
            tools=normalized_tools,
            tool_choice=tool_choice,
        )

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
        tool_parser_request: ChatCompletionRequest | None = None,
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
        if reasoning_content is not None and visible_text:
            visible_text = visible_text.lstrip()
        if tool_parser is not None:
            if tool_parser_request is None:
                tool_parser_request = ChatCompletionRequest(
                    model=str(self.esurge_name or "easymlx"),
                    messages=[ChatMessage(role="user", content="")],
                )
            result = tool_parser.extract_tool_calls(visible_text or "", tool_parser_request)
            if result.tools_called:
                tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in result.tool_calls]
                visible_text = result.content or ""
            else:
                visible_text = result.content or visible_text

        effective_finish_reason = finish_reason or stop_reason
        if tool_calls and effective_finish_reason in {None, FinishReason.EOS, FinishReason.STOP, FinishReason.LENGTH}:
            effective_finish_reason = FinishReason.TOOL_CALLS

        return {
            "text": visible_text,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason,
            "finish_reason": effective_finish_reason,
        }

    @staticmethod
    def _dump_parser_payload(payload: Any) -> Any:
        """Convert parser Pydantic objects into plain response payloads."""
        if payload is None:
            return None
        if isinstance(payload, list):
            return [eSurge._dump_parser_payload(item) for item in payload]
        if hasattr(payload, "model_dump"):
            return payload.model_dump(exclude_none=True)
        return payload

    def _parse_state_output(
        self,
        state: _RuntimeRequestState,
        *,
        raw_text: str,
        incremental: bool,
    ) -> dict[str, Any]:
        """Parse a state snapshot with EasyDeL-style streaming delegation when useful."""
        if state.delegating_parser is None or not incremental or (state.is_done and not state.parser_previous_text):
            return self._parse_text_output(
                raw_text,
                stop=state.stop,
                finish_reason=state.request.finished_reason,
                gen_kwargs=state.gen_kwargs,
                tool_parser=state.tool_parser_instance,
                reasoning_parser=state.reasoning_parser_instance,
                tool_parser_request=state.tool_parser_request,
            )

        include_stop = bool(state.gen_kwargs.get("include_stop_str_in_output", False))
        ignore_reasoning_stops = state.gen_kwargs.get("ignore_stop_strings_in_reasoning")
        if ignore_reasoning_stops is None:
            ignore_reasoning_stops = self.ignore_stop_strings_in_reasoning

        parser_input = raw_text
        stop_reason = None
        if not (state.reasoning_parser_instance is not None and bool(ignore_reasoning_stops)):
            parser_input, stop_reason = _apply_stop_strings(
                raw_text,
                state.stop,
                include_stop_str_in_output=include_stop,
            )

        token_ids = list(state.generated_token_ids)
        if state.is_done:
            parser_result = state.delegating_parser.process_final(parser_input, token_ids)
        else:
            previous_parser_text = state.parser_previous_text
            delta_text = compute_stream_delta_text(parser_input, previous_parser_text, "")
            parser_result = state.delegating_parser.process_delta(
                parser_input,
                delta_text,
                token_ids,
                previous_parser_text,
                state.parser_previous_token_ids,
            )

        state.parser_previous_text = parser_input
        state.parser_previous_token_ids = token_ids
        parsed = parser_result.to_dict()

        visible_text = str(parsed["accumulated_content"] or "")
        reasoning_content = str(parsed["accumulated_reasoning"] or "") or None
        if reasoning_content is not None and visible_text:
            visible_text = visible_text.lstrip()
            parsed["accumulated_content"] = visible_text
            if incremental:
                parsed["delta_content"] = compute_stream_delta_text(
                    visible_text,
                    state.last_emitted_text,
                    str(parsed.get("delta_content") or "").lstrip(),
                )
        if state.reasoning_parser_instance is not None and bool(ignore_reasoning_stops):
            trimmed_text, visible_stop_reason = _apply_stop_strings(
                visible_text,
                state.stop,
                include_stop_str_in_output=include_stop,
            )
            if visible_stop_reason is not None:
                stop_reason = visible_stop_reason
                previous_visible = state.last_emitted_text if incremental else ""
                parsed["accumulated_content"] = trimmed_text
                parsed["delta_content"] = compute_stream_delta_text(
                    trimmed_text,
                    previous_visible,
                    str(parsed.get("delta_content") or ""),
                )
                visible_text = trimmed_text

        tool_calls = self._dump_parser_payload(parsed.get("tool_calls"))
        delta_tool_calls = self._dump_parser_payload(parsed.get("delta_tool_calls"))
        delta_reasoning = parsed.get("delta_reasoning")
        delta_content = str(parsed.get("delta_content") or "")

        effective_finish_reason = state.request.finished_reason or stop_reason
        if tool_calls and effective_finish_reason in {None, FinishReason.EOS, FinishReason.STOP, FinishReason.LENGTH}:
            effective_finish_reason = FinishReason.TOOL_CALLS

        return {
            "text": visible_text,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "delta_text": delta_content,
            "delta_reasoning_content": delta_reasoning,
            "delta_tool_calls": delta_tool_calls,
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
        previous_text = state.last_emitted_text if incremental else ""
        previous_reasoning = state.last_emitted_reasoning if incremental else ""
        previous_tool_calls = state.last_emitted_tool_calls if incremental else None
        previous_seq = state.last_emitted_update_seq if incremental else 0
        parsed = self._parse_state_output(
            state,
            raw_text=state.decoded_text(self.tokenizer, stream_safe=incremental),
            incremental=incremental,
        )
        text = str(parsed["text"])
        reasoning_content = parsed["reasoning_content"]
        tool_calls = parsed["tool_calls"]
        finish_reason = parsed["finish_reason"]
        delta_text = (
            str(parsed["delta_text"])
            if incremental and "delta_text" in parsed
            else self._delta_text(text, previous_text)
        )
        reasoning_now = reasoning_content or ""
        delta_reasoning = (
            parsed.get("delta_reasoning_content")
            if incremental and "delta_reasoning_content" in parsed
            else (self._delta_text(reasoning_now, previous_reasoning) if reasoning_now else None)
        )
        delta_tool_calls = (
            parsed.get("delta_tool_calls")
            if incremental and "delta_tool_calls" in parsed
            else self._delta_tool_calls(tool_calls, previous_tool_calls)
        )

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
    ) -> tuple[mx.array, mx.array, list[list["int"]]]:
        """Tokenize, truncate, and pad a list of prompt strings.

        Args:
            prompts: Raw prompt strings.
            max_new_tokens: Maximum generation tokens (used to compute
                the allowed prompt length).

        Returns:
            A 3-tuple of ``(input_ids, attention_mask, prompt_token_ids)``
            where the first two are padded MLX arrays and the third is a
            list of per-prompt unpadded token id lists.
        """
        enc = self.tokenizer(prompts, padding=True)
        input_ids = [list(map(int, row_ids)) for row_ids in enc["input_ids"]]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = [[1] * len(row_ids) for row_ids in input_ids]
        else:
            attention_mask = [list(map(int, row_mask)) for row_mask in attention_mask]

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
            tokens = [int(token_id) for token_id, keep in zip(row_ids, row_mask, strict=False) if int(keep) != 0]
            if self.auto_truncate_prompt and len(tokens) > allowed_prompt:
                tokens, _ = truncate_tokens(tokens, allowed_prompt, mode=self.truncate_mode)
            prompt_token_ids.append(tokens)

        max_len = max((len(tokens) for tokens in prompt_token_ids), default=0)
        for tokens in prompt_token_ids:
            pad_len = max_len - len(tokens)
            truncated_ids.append(tokens + [pad_token_id] * pad_len)
            truncated_masks.append([1] * len(tokens) + [0] * pad_len)

        return (
            mx.array(truncated_ids, dtype=mx.int32),
            mx.array(truncated_masks, dtype=mx.int32),
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
            "presence_penalty",
            "repetition_penalty",
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
            presence_penalty=float(gen_kwargs.get("presence_penalty", 0.0) or 0.0),
            repetition_penalty=float(gen_kwargs.get("repetition_penalty", 1.0) or 1.0),
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
        requested_id = gen_kwargs.get("request_id")
        enable_tool_parser = self._should_enable_tool_parser_for_request(gen_kwargs)
        for idx, prompt in enumerate(prompts):
            prompt_ids = list(prompt_token_ids[idx])
            tool_parser_instance = self._new_tool_parser() if enable_tool_parser else None
            reasoning_parser_instance = self._new_reasoning_parser()
            tool_parser_request = (
                self._build_tool_parser_request(
                    prompt=prompt,
                    tools=gen_kwargs.get("tools"),
                    tool_choice=gen_kwargs.get("tool_choice"),
                )
                if tool_parser_instance is not None
                else None
            )
            delegating_parser = (
                DelegatingParser(
                    reasoning_parser=reasoning_parser_instance,
                    tool_parser=tool_parser_instance,
                    tool_request=tool_parser_request,
                )
                if reasoning_parser_instance is not None or tool_parser_instance is not None
                else None
            )
            if reasoning_parser_instance is not None:
                try:
                    reasoning_parser_instance.configure_prompt_context(
                        prompt_text=prompt,
                        prompt_token_ids=prompt_ids,
                    )
                except Exception:
                    pass
            request = EngineRequest(
                request_id=str(requested_id) if requested_id is not None and len(prompts) == 1 else str(uuid.uuid4()),
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
                    tool_parser_request=tool_parser_request,
                    delegating_parser=delegating_parser,
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
            return FinishReason.EOS

        total_generated = len(state.generated_token_ids) + len(sampled_token_ids)
        if total_generated >= state.request.max_new_tokens:
            return FinishReason.LENGTH

        has_stop_strings = bool(state.stop)
        has_tool_parser = state.tool_parser_instance is not None

        if not has_stop_strings and not has_tool_parser:
            return None

        tail_size = min(total_generated, 64)
        tail_ids = list(state.generated_token_ids[-tail_size:]) + [int(t) for t in sampled_token_ids]
        tail_text = _strip_unstable_replacement_suffix(self.tokenizer.decode(tail_ids, skip_special_tokens=True))

        if has_stop_strings:
            for stop_str in state.stop:
                if stop_str in tail_text:
                    return FinishReason.STOP

        if has_tool_parser:
            candidate_generated = list(state.generated_token_ids) + [int(t) for t in sampled_token_ids]
            raw_text = _strip_unstable_replacement_suffix(
                self.tokenizer.decode(candidate_generated, skip_special_tokens=True)
            )
            parsed = self._parse_text_output(
                raw_text,
                stop=state.stop,
                finish_reason=None,
                gen_kwargs=state.gen_kwargs,
                tool_parser=state.tool_parser_instance,
                reasoning_parser=state.reasoning_parser_instance,
                tool_parser_request=state.tool_parser_request,
            )
            if parsed["tool_calls"]:
                return FinishReason.TOOL_CALLS

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
        the model, and applies results.  When overlap execution is
        enabled, one runner future may be pending while the scheduler
        speculatively admits unrelated waiting work for the following
        step.
        """
        pending_execution: tuple[PendingExecution, SchedulerStepOutput, ExecutionRequest, Any | None] | None = None
        prefetched_schedule: SchedulerStepOutput | None = None
        overlap_enabled = bool(self.async_scheduling and self.overlap_execution)

        def _fail_all_locked(error: BaseException) -> None:
            if self._scheduler is not None:
                self._scheduler.reset()
            for state in self._runtime_states.values():
                if not state.request.is_finished:
                    state.mark_error(error)
            self._has_terminal_states = True
            self._work_cv.notify_all()

        def _try_prefetch_locked(previous_step: SchedulerStepOutput) -> SchedulerStepOutput | None:
            if self._scheduler is None or self._paused or not self._scheduler.has_pending_work():
                return None
            blocked = {entry.request_id for entry in previous_step.scheduled}
            if not blocked:
                return None
            output = self._scheduler.schedule(
                blocked_request_ids=blocked,
                allow_preemption=False,
            )
            if not output.scheduled:
                return None
            return output

        def _safe_apply_locked(
            step_output: SchedulerStepOutput,
            execution_request: ExecutionRequest,
            result: ExecutionResult | None,
            execution_error: BaseException | None,
        ) -> None:
            try:
                self._apply_execution_result_locked(step_output, execution_request, result, execution_error)
            except BaseException as exc:
                _fail_all_locked(exc)

        while not self._shutdown.is_set():
            if pending_execution is not None:
                pending, step_output, execution_request, dispatch = pending_execution
                if overlap_enabled and prefetched_schedule is None:
                    with self._work_cv:
                        if not self._shutdown.is_set():
                            prefetched_schedule = _try_prefetch_locked(step_output)

                execution_error: BaseException | None = None
                result: ExecutionResult | None = None
                try:
                    if self._execution_manager is None:
                        raise RuntimeError("Execution manager is not initialized.")
                    result = self._execution_manager.collect(pending)
                    self._verify_distributed_step(dispatch, result.output)
                except BaseException as exc:
                    execution_error = exc

                with self._work_cv:
                    if execution_error is not None and prefetched_schedule is not None:
                        _fail_all_locked(execution_error)
                        prefetched_schedule = None
                    else:
                        _safe_apply_locked(
                            step_output,
                            execution_request,
                            result,
                            execution_error,
                        )
                    pending_execution = None
                    self._work_cv.notify_all()
                continue

            with self._work_cv:
                while not self._shutdown.is_set() and (
                    prefetched_schedule is None
                    and (self._paused or self._scheduler is None or not self._scheduler.has_pending_work())
                ):
                    self._reconcile_terminal_states_locked()
                    self._work_cv.wait(timeout=0.05)

                if self._shutdown.is_set():
                    break
                if self._paused or self._scheduler is None:
                    continue

                if self._has_terminal_states:
                    self._reconcile_terminal_states_locked()
                if prefetched_schedule is not None:
                    step_output = prefetched_schedule
                    prefetched_schedule = None
                else:
                    step_output = self._scheduler.schedule()
                if not step_output.scheduled:
                    self._work_cv.wait(timeout=0.001)
                    continue
                self._step_counter += 1
                execution_request = self._build_execution_request(step_output)
                dispatch = (
                    self._dispatch_distributed_step(step_output) if self._distributed_controller is not None else None
                )

            execution_error: BaseException | None = None
            result: ExecutionResult | None = None
            try:
                if self._execution_manager is None:
                    raise RuntimeError("Execution manager is not initialized.")
                if overlap_enabled:
                    pending_execution = (
                        self._execution_manager.execute_async(execution_request),
                        step_output,
                        execution_request,
                        dispatch,
                    )
                    continue
                result = self._execution_manager.execute(execution_request)
                self._verify_distributed_step(dispatch, result.output)
            except BaseException as exc:
                execution_error = exc

            with self._work_cv:
                _safe_apply_locked(step_output, execution_request, result, execution_error)
                self._work_cv.notify_all()

        if pending_execution is not None:
            pending, step_output, execution_request, dispatch = pending_execution
            execution_error: BaseException | None = None
            result: ExecutionResult | None = None
            try:
                if self._execution_manager is None:
                    raise RuntimeError("Execution manager is not initialized.")
                result = self._execution_manager.collect(pending)
                self._verify_distributed_step(dispatch, result.output)
            except BaseException as exc:
                execution_error = exc
            with self._work_cv:
                _safe_apply_locked(step_output, execution_request, result, execution_error)
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
        if len(prompts_list) == 1 and not self._paused:
            output = self._generate_speculative_sync(
                prompts_list[0],
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            if output is not None:
                return [output]
            output = self._generate_sync(
                prompts_list[0],
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            if output is not None:
                return [output]
        return self._generate_with_background(prompts_list, sampling_params=sampling_params, **generate_kwargs)

    def _sync_validate_preconditions(
        self,
        sampling_params: SamplingParams | None,
    ) -> tuple[ModelRunner, "Scheduler"] | None:
        """Validate that the sync fast path can be used.

        Returns:
            A ``(runner, scheduler)`` tuple if preconditions are met,
            ``None`` otherwise.
        """
        if sampling_params is not None and sampling_params.n != 1:
            raise NotImplementedError("n>1 sampling is not supported in easymlx eSurge.")
        runner = self._runner
        scheduler = self._scheduler
        if runner is None or scheduler is None:
            return None
        return runner, scheduler

    def _sync_prepare_kwargs(
        self,
        sampling_params: SamplingParams | None,
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], int, set[int]]:
        """Build generation kwargs, max_new_tokens, and EOS set for sync paths.

        Returns:
            A 3-tuple of ``(gen_kwargs, max_new_tokens, eos_token_ids)``.
        """
        gen_kwargs = self._resolve_generation_kwargs(sampling_params, **generate_kwargs)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", sampling_params.max_tokens if sampling_params else 16))
        if self.auto_cap_new_tokens:
            max_new_tokens = max(1, min(max_new_tokens, self.max_model_len - self.reserve_tokens))
        gen_kwargs["max_new_tokens"] = max_new_tokens
        if gen_kwargs.get("eos_token_id") is None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if gen_kwargs.get("pad_token_id") is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id or gen_kwargs.get("eos_token_id")
        eos_set = _normalize_eos_token_ids(gen_kwargs.get("eos_token_id"))
        return gen_kwargs, max_new_tokens, eos_set

    def _sync_execute_step(
        self,
        runner: "ModelRunner",
        execution_request: ExecutionRequest,
        step_output: "SchedulerStepOutput",
    ) -> tuple[Any, Any]:
        """Run one forward step, optionally dispatching to distributed workers."""
        dispatch = self._dispatch_distributed_step(step_output) if self._distributed_controller is not None else None
        raw_output, logits = runner._forward_step(execution_request)
        self._verify_distributed_step(dispatch, raw_output)
        return raw_output, logits

    def _sync_sample_token(
        self,
        raw_output: Any,
        logits: Any,
        runner: "ModelRunner",
        greedy_sampling: bool,
        sampling_params: SamplingParams | None,
        request: EngineRequest,
    ) -> int:
        """Sample a single token from logits."""
        del raw_output, greedy_sampling
        return runner.sample_next_token(
            logits,
            sampling_params=sampling_params,
            prompt_token_ids=request.prompt_token_ids,
            generated_token_ids=request.generated_token_ids,
        )

    def _sync_check_stop_reason(
        self,
        token_id: int,
        state: "_RuntimeRequestState",
        request: EngineRequest,
    ) -> str | None:
        """Check whether generation should stop after a token.

        Only decodes the tail when stop strings are configured.
        """
        if token_id in state.eos_token_ids:
            return FinishReason.EOS
        if len(request.generated_token_ids) >= request.max_new_tokens:
            return FinishReason.LENGTH
        if state.stop:
            tail_text = _strip_unstable_replacement_suffix(
                self.tokenizer.decode(request.generated_token_ids[-64:], skip_special_tokens=True)
            )
            for stop_str in state.stop:
                if stop_str in tail_text:
                    return FinishReason.STOP
        return None

    def _sync_decode_step_output(
        self,
        token_id: int,
        row_index: int,
        request: EngineRequest,
        scheduler: "Scheduler",
    ) -> "SchedulerStepOutput":
        """Build a single-token decode SchedulerStepOutput."""
        scheduled = ScheduledRequest(
            request_id=request.request_id,
            row_index=row_index,
            token_ids=[int(token_id)],
            is_prefill=False,
            num_tokens=1,
            page_ids=list(request.cache_state.page_ids),
            cache_group=request.cache_state.cache_group,
            prefix_cache_hit=request.cache_state.prefix_cache_hit,
        )
        return SchedulerStepOutput(
            scheduled=[scheduled],
            num_scheduled=1,
            num_running=1,
            token_budget_remaining=max(int(getattr(scheduler, "max_num_scheduled_tokens", 1)) - 1, 0),
            decode_only=True,
        )

    def _sync_finalize(
        self,
        *,
        paused_for_sync: bool,
        runtime_state_added: bool,
        request_id: str,
        runner: "ModelRunner",
        scheduler: "Scheduler",
    ) -> None:
        """Shared cleanup for sync generate and stream paths."""
        if runtime_state_added:
            with self._work_cv:
                self._runtime_states.pop(request_id, None)
        if runner.sequence_buffer.has_request(request_id):
            runner.detach_request(request_id)
        runner.compact_rows()
        self._reset_recurrent_states()
        if paused_for_sync:
            with self._work_cv:
                self._paused = False
                scheduler.resume()
                self._work_cv.notify_all()

    def _speculative_decoding_enabled(self) -> bool:
        """Return whether a draft model is configured for speculative decoding."""
        return bool(self._speculative_model is not None and self.speculative_config.num_speculative_tokens > 0)

    def _can_use_speculative_sync_path(
        self,
        state: _RuntimeRequestState,
        gen_kwargs: dict[str, Any],
    ) -> bool:
        """Return whether the current request can use exact speculative decode."""
        if not self._speculative_decoding_enabled():
            return False
        if self._distributed_controller is not None and getattr(self._distributed_controller, "enabled", False):
            return False
        if gen_kwargs.get("multimodal_payload") is not None or gen_kwargs.get("multimodal_inputs") is not None:
            return False
        if not state.prompt_token_ids:
            return False
        params = state.request.sampling_params
        if params is not None and bool(params.do_sample):
            return False
        return True

    def _forward_full_context_output(self, model: Any, token_ids: list[int], **kwargs: Any) -> Any:
        """Run a model without paged cache and return the raw output."""
        if not token_ids:
            raise RuntimeError("Speculative decoding requires at least one context token.")
        input_ids = mx.array([[int(token_id) for token_id in token_ids]], dtype=mx.int32)
        return _call_with_filtered_kwargs(
            model,
            input_ids=input_ids,
            return_dict=True,
            **kwargs,
        )

    @staticmethod
    def _full_context_logits_from_output(output: Any) -> mx.array:
        """Extract per-position logits from a full-context model call."""
        logits = getattr(output, "logits", output)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits)
        if not isinstance(logits, mx.array):
            logits = mx.array(logits)
        if logits.ndim == 3:
            if int(logits.shape[0]) != 1:
                raise RuntimeError(f"Expected batch size 1 for speculative logits, got {logits.shape}")
            logits = logits[0]
        if logits.ndim == 1:
            logits = mx.expand_dims(logits, axis=0)
        if logits.ndim != 2:
            raise RuntimeError(f"Unsupported speculative logits shape: {logits.shape}")
        return logits

    def _forward_full_context_logits(self, model: Any, token_ids: list[int]) -> mx.array:
        """Run a model without paged cache and return per-position logits."""
        output = self._forward_full_context_output(model, token_ids)
        return self._full_context_logits_from_output(output)

    @staticmethod
    def _extract_hidden_states_from_output(output: Any) -> Any | None:
        """Extract hidden-state features from common output shapes."""
        if output is None:
            return None
        if isinstance(output, dict):
            for key in ("hidden_states", "eagle3_hidden_states", "eagle3_features", "features"):
                if output.get(key) is not None:
                    return output[key]
            return None
        for attr in ("hidden_states", "eagle3_hidden_states", "eagle3_features", "features"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        return output if isinstance(output, mx.array) else None

    @staticmethod
    def _normalize_hidden_state_features(hidden_states: Any | None) -> tuple[mx.array, ...] | None:
        """Normalize hidden-state features to a tuple of MLX arrays."""
        if hidden_states is None:
            return None
        if isinstance(hidden_states, mx.array):
            return (hidden_states,)
        if isinstance(hidden_states, dict):
            extracted = eSurge._extract_hidden_states_from_output(hidden_states)
            return eSurge._normalize_hidden_state_features(extracted)
        if isinstance(hidden_states, (list, tuple)):
            arrays = tuple(mx.array(value) for value in hidden_states if value is not None)
            return arrays or None
        return (mx.array(hidden_states),)

    @staticmethod
    def _concat_eagle3_features(hidden_states: tuple[mx.array, ...] | None) -> mx.array | None:
        """Concatenate EAGLE3 hidden-state features along the channel axis."""
        if not hidden_states:
            return None
        if len(hidden_states) == 1:
            return hidden_states[0]
        return mx.concatenate(list(hidden_states), axis=-1)

    def _target_eagle3_hidden_states(self, token_ids: list[int]) -> tuple[mx.array, ...] | None:
        """Fetch EAGLE3 target hidden-state features from the target model."""
        input_ids = mx.array([[int(token_id) for token_id in token_ids]], dtype=mx.int32)
        layer_indices = self.speculative_config.eagle3_feature_layer_indices
        for method_name in ("eagle3_hidden_states", "get_eagle3_hidden_states"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                output = _call_with_filtered_kwargs(
                    method,
                    input_ids=input_ids,
                    token_ids=list(token_ids),
                    feature_layer_indices=layer_indices,
                    layer_indices=layer_indices,
                )
                extracted = self._extract_hidden_states_from_output(output)
                if extracted is None and isinstance(output, (list, tuple, mx.array)):
                    extracted = output
                return self._normalize_hidden_state_features(extracted)
        return None

    def _target_logits_and_eagle3_features(
        self,
        token_ids: list[int],
        *,
        stats: _SpeculativeDecodeStats,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Run the target model and collect logits plus EAGLE3 features."""
        output = self._forward_full_context_output(
            self.model,
            token_ids,
            output_hidden_states=True,
            eagle3_feature_layer_indices=self.speculative_config.eagle3_feature_layer_indices,
        )
        stats.target_steps += 1
        logits = self._full_context_logits_from_output(output)
        hidden_states = self._normalize_hidden_state_features(self._extract_hidden_states_from_output(output))
        if hidden_states is None:
            hidden_states = self._target_eagle3_hidden_states(token_ids)
        if hidden_states is None:
            raise RuntimeError(
                "EAGLE3 speculative decoding requires target hidden-state features. "
                "Expose `eagle3_hidden_states(...)` on the target model or return `hidden_states` "
                "from a full-context call with `output_hidden_states=True`."
            )
        return logits, hidden_states

    @staticmethod
    def _coerce_speculative_token_list(tokens: Any, *, max_tokens: int) -> list[int]:
        """Normalize proposer output to a bounded Python token list."""
        if tokens is None:
            return []
        if isinstance(tokens, dict):
            for key in ("token_ids", "tokens", "draft_tokens", "proposal"):
                if key in tokens:
                    return eSurge._coerce_speculative_token_list(tokens[key], max_tokens=max_tokens)
            return []
        for attr in ("token_ids", "tokens", "draft_tokens", "proposal"):
            value = getattr(tokens, attr, None)
            if value is not None:
                return eSurge._coerce_speculative_token_list(value, max_tokens=max_tokens)
        if isinstance(tokens, mx.array):
            value = tokens.astype(mx.int64).tolist()
        else:
            value = tokens
        if isinstance(value, int):
            flat = [value]
        else:
            flat = []

            def visit(item: Any) -> None:
                if isinstance(item, (list, tuple)):
                    for child in item:
                        visit(child)
                elif item is not None:
                    flat.append(int(item))

            visit(value)
        return [int(token_id) for token_id in flat[:max_tokens]]

    def _call_eagle3_proposer(
        self,
        *,
        state: _RuntimeRequestState,
        token_ids: list[int],
        target_logits: mx.array,
        hidden_states: tuple[mx.array, ...],
        max_tokens: int,
    ) -> list[int]:
        """Call an EAGLE3 adapter and return a linear proposal."""
        proposer = self._speculative_model
        if proposer is None:
            return []

        input_ids = mx.array([[int(token_id) for token_id in token_ids]], dtype=mx.int32)
        features = self._concat_eagle3_features(hidden_states)
        kwargs = {
            "input_ids": input_ids,
            "token_ids": list(token_ids),
            "generated_token_ids": list(state.generated_token_ids),
            "hidden_states": hidden_states,
            "eagle3_features": features,
            "features": features,
            "target_logits": target_logits,
            "logits": target_logits,
            "max_tokens": int(max_tokens),
            "num_tokens": int(max_tokens),
            "sampling_params": state.request.sampling_params,
            "tokenizer": self.tokenizer,
            "target_model": self.model,
            "eos_token_ids": set(state.eos_token_ids),
            "lm_head": getattr(self.model, "lm_head", None),
        }

        for method_name in (
            "propose_eagle3",
            "eagle3_propose",
            "propose_tokens",
            "draft_tokens",
            "propose",
        ):
            method = getattr(proposer, method_name, None)
            if callable(method):
                return self._coerce_speculative_token_list(
                    _call_with_filtered_kwargs(method, **kwargs),
                    max_tokens=max_tokens,
                )

        raise RuntimeError(
            "EAGLE3 speculative_model must expose a proposal method such as `propose_eagle3(...)` or `propose(...)`."
        )

    def _eagle3_speculative_tokens(
        self,
        state: _RuntimeRequestState,
        *,
        max_tokens: int,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Generate an EAGLE3 draft proposal from target hidden features."""
        if max_tokens <= 0:
            return []
        context = list(state.request.all_token_ids)
        target_logits, hidden_states = self._target_logits_and_eagle3_features(context, stats=stats)
        draft_tokens = self._call_eagle3_proposer(
            state=state,
            token_ids=context,
            target_logits=target_logits,
            hidden_states=hidden_states,
            max_tokens=max_tokens,
        )
        stats.draft_tokens += len(draft_tokens)
        return draft_tokens

    def _target_logits_and_dflash_features(
        self,
        token_ids: list[int],
        *,
        stats: _SpeculativeDecodeStats,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Run the target model and collect the hidden features required by DFlash."""
        proposer = self._speculative_model
        layer_indices = getattr(proposer, "target_feature_layer_indices", None)
        output = self._forward_full_context_output(
            self.model,
            token_ids,
            output_hidden_states=True,
            eagle3_feature_layer_indices=layer_indices,
        )
        stats.target_steps += 1
        logits = self._full_context_logits_from_output(output)
        hidden_states = self._normalize_hidden_state_features(self._extract_hidden_states_from_output(output))
        if hidden_states is None:
            hidden_states = self._target_eagle3_hidden_states(token_ids)
        if hidden_states is None:
            raise RuntimeError(
                "DFlash speculative decoding requires target hidden-state features. "
                "The target model must expose `output_hidden_states=True` or `eagle3_hidden_states(...)`."
            )
        return logits, hidden_states

    def _dflash_speculative_tokens(
        self,
        state: _RuntimeRequestState,
        *,
        max_tokens: int,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Generate a DFlash proposal seeded by the target model's next token."""
        proposer = self._speculative_model
        if proposer is None or max_tokens <= 0:
            return []

        context = list(state.request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        target_logits, hidden_states = self._target_logits_and_dflash_features(context, stats=stats)
        first_token = self._sample_speculative_token(
            target_logits[-1],
            state,
            token_history=context,
            generated_history=generated_history,
        )

        draft_tokens = [int(first_token)]
        if int(first_token) in state.eos_token_ids or max_tokens == 1:
            stats.draft_tokens += len(draft_tokens)
            return draft_tokens

        logits_method = getattr(proposer, "dflash_logits", None)
        if not callable(logits_method):
            raise RuntimeError("DFlash speculative_model must expose `dflash_logits(...)`.")

        draft_logits = _call_with_filtered_kwargs(
            logits_method,
            first_token=int(first_token),
            hidden_states=hidden_states,
            max_tokens=int(max_tokens) - 1,
            target_model=self.model,
        )
        if not isinstance(draft_logits, mx.array):
            draft_logits = mx.array(draft_logits)
        if draft_logits.ndim == 3:
            draft_logits = draft_logits.reshape((-1, int(draft_logits.shape[-1])))

        context.append(int(first_token))
        generated_history.append(int(first_token))
        for row in draft_logits:
            if len(draft_tokens) >= max_tokens:
                break
            token_id = self._sample_speculative_token(
                row,
                state,
                token_history=context,
                generated_history=generated_history,
            )
            draft_tokens.append(int(token_id))
            context.append(int(token_id))
            generated_history.append(int(token_id))
            if int(token_id) in state.eos_token_ids:
                break

        stats.draft_tokens += len(draft_tokens)
        return draft_tokens

    def _init_speculative_cache_views(self, model: Any) -> list[Any] | None:
        """Allocate a single-slot paged cache for speculative decoding."""
        init_ops = getattr(model, "init_operations_cache", None)
        if not callable(init_ops):
            return None
        result = init_ops(
            batch_size=1,
            max_length=self.max_model_len,
            page_size=self.cache_config.page_size,
            dtype=self.kv_cache_dtype,
            cache_type="paged",
        )
        if hasattr(result, "views"):
            cache_views = list(result.views)
        elif isinstance(result, (list, tuple)):
            cache_views = list(result)
        elif hasattr(result, "__iter__"):
            cache_views = list(result)
        else:
            cache_views = [result]
        if not cache_views:
            return None
        self._disable_unused_packed_page_cache(cache_views)
        return cache_views

    @staticmethod
    def _speculative_page_metadata(num_tokens: int, *, single_token: bool) -> Any:
        """Build PageMetadata for a single speculative cache slot."""
        from easymlx.caching import PageMetadata, build_query_start_loc

        return PageMetadata(
            query_start_loc=build_query_start_loc([int(num_tokens)]),
            is_single_token_decode=bool(single_token),
            slot_ids=(0,),
        )

    @staticmethod
    def _last_logits_row(logits: mx.array) -> mx.array:
        """Return the last next-token logits row from common logits shapes."""
        if not isinstance(logits, mx.array):
            logits = mx.array(logits)
        if logits.ndim == 3:
            if int(logits.shape[0]) != 1:
                raise RuntimeError(f"Expected batch size 1 for cached speculative logits, got {logits.shape}")
            return logits[0, -1]
        if logits.ndim == 2:
            return logits[-1]
        if logits.ndim == 1:
            return logits
        raise RuntimeError(f"Unsupported cached speculative logits shape: {logits.shape}")

    @staticmethod
    def _clone_decode_state(state: Any) -> Any:
        """Clone nested decode-state containers enough to avoid aliasing lists."""
        if state is None:
            return None
        if isinstance(state, dict):
            return {key: eSurge._clone_decode_state(value) for key, value in state.items()}
        if isinstance(state, list):
            return [eSurge._clone_decode_state(value) for value in state]
        if isinstance(state, tuple):
            return tuple(eSurge._clone_decode_state(value) for value in state)
        return state

    @staticmethod
    def _clone_page_cache_view(cache: Any) -> Any:
        """Clone mutable cache metadata while sharing KV storage arrays."""
        attrs = {
            "key_cache": cache.key_cache,
            "value_cache": cache.value_cache,
            "block_tables": cache.block_tables,
            "kv_lens": cache.kv_lens + 0,
            "block_size": cache.block_size,
            "num_kv_heads": cache.num_kv_heads,
            "head_dim": cache.head_dim,
        }
        for optional in (
            "page_key_cache",
            "page_value_cache",
            "page_vec_size",
            "cache_dtype_is_fp8",
            "k_scales",
            "v_scales",
        ):
            if hasattr(cache, optional):
                attrs[optional] = getattr(cache, optional)
        return type(cache)(**attrs)

    @classmethod
    def _clone_cache_views_for_speculative(cls, cache_views: list[Any]) -> list[Any]:
        """Clone a list of page-cache views for speculative verification."""
        cloned: list[Any] = []
        for cache in cache_views:
            if cache is None:
                cloned.append(None)
                continue
            required = ("key_cache", "value_cache", "block_tables", "kv_lens", "block_size", "num_kv_heads", "head_dim")
            if all(hasattr(cache, attr) for attr in required):
                cloned.append(cls._clone_page_cache_view(cache))
                continue
            raise RuntimeError(f"Unsupported cache type for fast speculative decoding: {type(cache).__name__}")
        return cloned

    @staticmethod
    def _adopt_cache_view_storage(cache_views: list[Any], source_views: list[Any]) -> None:
        """Promote cache storage arrays written through temporary views."""
        for cache, source in zip(cache_views, source_views, strict=False):
            if cache is None or source is None:
                continue
            for attr in ("key_cache", "value_cache", "page_key_cache", "page_value_cache", "k_scales", "v_scales"):
                if hasattr(cache, attr) and hasattr(source, attr):
                    setattr(cache, attr, getattr(source, attr))

    @staticmethod
    def _cache_slot_length(cache_views: list[Any]) -> int:
        """Return the current token length for speculative slot 0."""
        length = 0
        for cache in cache_views:
            kv_lens = getattr(cache, "kv_lens", None)
            if kv_lens is not None:
                length = max(length, int(kv_lens[0].item()))
        return length

    @staticmethod
    def _set_cache_slot_length(cache_views: list[Any], length: int) -> None:
        """Set speculative slot 0 length across all page-cache views."""
        for cache in cache_views:
            kv_lens = getattr(cache, "kv_lens", None)
            if kv_lens is not None:
                kv_lens[0] = int(length)

    def _model_get_decode_state(self, model: Any) -> Any | None:
        get_state = getattr(model, "get_decode_state", None)
        if not callable(get_state):
            return None
        return self._clone_decode_state(get_state())

    def _model_set_decode_state(self, model: Any, decode_state: Any | None) -> None:
        if decode_state is None:
            return
        set_state = getattr(model, "set_decode_state", None)
        if callable(set_state):
            set_state(decode_state)

    def _cached_forward_logits(
        self,
        model: Any,
        token_ids: list[int],
        cache_views: list[Any],
        *,
        decode_state: Any | None = None,
        prefer_decode_state: bool = True,
    ) -> tuple[mx.array, Any | None]:
        """Forward tokens through a speculative cache and return logits."""
        if not token_ids:
            raise RuntimeError("Cached speculative forward requires at least one token.")

        input_ids = mx.array([[int(token_id) for token_id in token_ids]], dtype=mx.int32)
        metadata = self._speculative_page_metadata(len(token_ids), single_token=len(token_ids) == 1)
        decode_fn = getattr(model, "decode_step_with_state", None)
        can_use_decode_state = len(token_ids) == 1 or bool(getattr(model, "supports_multitoken_decode_state", False))
        if prefer_decode_state and can_use_decode_state and decode_state is not None and callable(decode_fn):
            logits, new_decode_state = _call_with_filtered_kwargs(
                decode_fn,
                input_ids=input_ids,
                cache_views=cache_views,
                cache_metadata=metadata,
                decode_state=decode_state,
            )
            return self._full_context_logits_from_output(logits), self._clone_decode_state(new_decode_state)

        output = _call_with_filtered_kwargs(
            model,
            input_ids=input_ids,
            cache_views=cache_views,
            cache_metadata=metadata,
            return_dict=True,
        )
        return self._full_context_logits_from_output(output), self._model_get_decode_state(model)

    def _cached_forward_logits_and_hidden(
        self,
        model: Any,
        token_ids: list[int],
        cache_views: list[Any],
        decode_state: Any | None,
        *,
        feature_layer_indices: tuple[int, ...] | list[int] | None,
    ) -> tuple[mx.array, Any | None, tuple[mx.array, ...]]:
        """Forward cached tokens and return logits, decode state, and hidden features."""
        if not token_ids:
            raise RuntimeError("Cached speculative forward requires at least one token.")

        method = getattr(model, "decode_step_with_state_and_hidden", None)
        if not callable(method):
            raise RuntimeError(
                f"{type(model).__name__} does not expose decode_step_with_state_and_hidden(), "
                "which cached DFlash requires."
            )

        input_ids = mx.array([[int(token_id) for token_id in token_ids]], dtype=mx.int32)
        metadata = self._speculative_page_metadata(len(token_ids), single_token=len(token_ids) == 1)
        output = _call_with_filtered_kwargs(
            method,
            input_ids=input_ids,
            cache_views=cache_views,
            cache_metadata=metadata,
            decode_state=decode_state,
            feature_layer_indices=feature_layer_indices,
            layer_indices=feature_layer_indices,
            output_hidden_states=True,
        )

        logits: Any
        new_decode_state: Any | None
        hidden_states: Any | None
        if isinstance(output, tuple):
            if len(output) < 3:
                raise RuntimeError(
                    "decode_step_with_state_and_hidden() must return (logits, decode_state, hidden_states)."
                )
            logits, new_decode_state, hidden_states = output[:3]
        elif isinstance(output, dict):
            logits = output.get("logits")
            new_decode_state = output.get("decode_state", output.get("state"))
            hidden_states = self._extract_hidden_states_from_output(output)
        else:
            logits = getattr(output, "logits", output)
            new_decode_state = getattr(output, "decode_state", getattr(output, "state", None))
            hidden_states = self._extract_hidden_states_from_output(output)

        normalized_hidden = self._normalize_hidden_state_features(hidden_states)
        if normalized_hidden is None:
            raise RuntimeError("Cached DFlash target forward did not return hidden-state features.")
        return self._full_context_logits_from_output(logits), new_decode_state, normalized_hidden

    @staticmethod
    def _append_hidden_feature_tuples(
        left: tuple[mx.array, ...] | None,
        right: tuple[mx.array, ...] | None,
    ) -> tuple[mx.array, ...]:
        """Concatenate hidden-feature tuples along their token axis."""
        if not left:
            return tuple(right or ())
        if not right:
            return tuple(left)
        if len(left) != len(right):
            raise RuntimeError(f"Hidden feature count mismatch: {len(left)} != {len(right)}")
        merged: list[mx.array] = []
        for left_item, right_item in zip(left, right, strict=True):
            axis = 1 if left_item.ndim >= 3 else 0
            merged.append(mx.concatenate([left_item, right_item], axis=axis))
        return tuple(merged)

    def _prefill_speculative_cache(
        self,
        model: Any,
        token_ids: list[int],
        cache_views: list[Any],
    ) -> tuple[Any | None, mx.array]:
        """Populate a speculative cache with the prompt/context."""
        self._reset_recurrent_states(model)
        logits, decode_state = self._cached_forward_logits(
            model,
            token_ids,
            cache_views,
            decode_state=None,
            prefer_decode_state=False,
        )
        return decode_state, self._last_logits_row(logits)

    def _initialize_dflash_cache_state(self, state: _RuntimeRequestState) -> _DFlashSpeculativeCacheState | None:
        """Initialize cached target and DFlash state for one request."""
        proposer = self._speculative_model
        make_cache = getattr(proposer, "make_cache", None)
        if proposer is None or not callable(make_cache):
            return None
        try:
            target_cache_views = self._init_speculative_cache_views(self.model)
            if target_cache_views is None:
                return None

            context = list(state.request.all_token_ids)
            layer_indices = getattr(proposer, "target_feature_layer_indices", None)
            self._reset_recurrent_states(self.model)
            output = self._forward_full_context_output(
                self.model,
                context,
                output_hidden_states=True,
                eagle3_feature_layer_indices=layer_indices,
            )
            hidden_states = self._normalize_hidden_state_features(self._extract_hidden_states_from_output(output))
            if hidden_states is None:
                hidden_states = self._target_eagle3_hidden_states(context)
            if hidden_states is None:
                return None
            target_decode_state, target_next_logits = self._prefill_speculative_cache(
                self.model,
                context,
                target_cache_views,
            )
            return _DFlashSpeculativeCacheState(
                target_cache_views=target_cache_views,
                target_decode_state=target_decode_state,
                target_next_logits=target_next_logits,
                dflash_cache=list(make_cache()),
                dflash_hidden_states=hidden_states,
            )
        except Exception as exc:
            logger.debug("Cached DFlash speculative path unavailable: %s", exc)
            return None

    def _initialize_speculative_cache_state(
        self,
        state: _RuntimeRequestState,
    ) -> _SpeculativeCacheState | _DFlashSpeculativeCacheState | None:
        """Initialize fast cached state for draft-model speculative decoding."""
        if self.speculative_config.speculative_method == "dflash":
            return self._initialize_dflash_cache_state(state)
        if self.speculative_config.speculative_method != "draft":
            return None
        draft_model = self._speculative_model
        if draft_model is None:
            return None
        try:
            target_cache_views = self._init_speculative_cache_views(self.model)
            draft_cache_views = self._init_speculative_cache_views(draft_model)
            if target_cache_views is None or draft_cache_views is None:
                return None

            context = list(state.request.all_token_ids)
            target_decode_state, target_next_logits = self._prefill_speculative_cache(
                self.model,
                context,
                target_cache_views,
            )
            draft_decode_state, draft_next_logits = self._prefill_speculative_cache(
                draft_model,
                context,
                draft_cache_views,
            )
        except Exception as exc:
            logger.debug("Fast speculative cache path unavailable: %s", exc)
            return None
        return _SpeculativeCacheState(
            target_cache_views=target_cache_views,
            target_decode_state=target_decode_state,
            target_next_logits=target_next_logits,
            draft_cache_views=draft_cache_views,
            draft_decode_state=draft_decode_state,
            draft_next_logits=draft_next_logits,
        )

    def _advance_cached_model_one(
        self,
        model: Any,
        token_id: int,
        cache_views: list[Any],
        decode_state: Any | None,
    ) -> tuple[Any | None, mx.array]:
        """Append one token to a cached model and return updated next logits."""
        logits, decode_state = self._cached_forward_logits(
            model,
            [int(token_id)],
            cache_views,
            decode_state=decode_state,
            prefer_decode_state=True,
        )
        return decode_state, self._last_logits_row(logits)

    def _advance_cached_model_tokens(
        self,
        model: Any,
        token_ids: list[int],
        cache_views: list[Any],
        decode_state: Any | None,
    ) -> tuple[Any | None, mx.array]:
        """Append one or more tokens to a cached model."""
        if not token_ids:
            raise RuntimeError("Cached speculative advance requires at least one token.")
        if len(token_ids) == 1:
            return self._advance_cached_model_one(model, int(token_ids[0]), cache_views, decode_state)

        self._model_set_decode_state(model, decode_state)
        logits, new_decode_state = self._cached_forward_logits(
            model,
            [int(token_id) for token_id in token_ids],
            cache_views,
            decode_state=None,
            prefer_decode_state=False,
        )
        return new_decode_state, self._last_logits_row(logits)

    def _advance_draft_cache_tokens(
        self,
        cache_state: _SpeculativeCacheState,
        token_ids: list[int],
    ) -> None:
        """Advance the real draft cache to match accepted output tokens."""
        draft_model = self._speculative_model
        if draft_model is None or not token_ids:
            return
        cache_state.draft_decode_state, cache_state.draft_next_logits = self._advance_cached_model_tokens(
            draft_model,
            [int(token_id) for token_id in token_ids],
            cache_state.draft_cache_views,
            cache_state.draft_decode_state,
        )

    def _draft_speculative_tokens_cached(
        self,
        state: _RuntimeRequestState,
        cache_state: _SpeculativeCacheState,
        *,
        max_tokens: int,
        stats: _SpeculativeDecodeStats,
    ) -> _CachedDraftProposal:
        """Generate draft proposals using a temporary cached draft state."""
        draft_model = self._speculative_model
        if draft_model is None or max_tokens <= 0:
            return _CachedDraftProposal(
                tokens=[],
                cache_views=cache_state.draft_cache_views,
                decode_state=cache_state.draft_decode_state,
                next_logits=cache_state.draft_next_logits,
            )

        temp_cache_views = self._clone_cache_views_for_speculative(cache_state.draft_cache_views)
        temp_decode_state = self._clone_decode_state(cache_state.draft_decode_state)
        next_logits = cache_state.draft_next_logits
        token_history = list(state.request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        draft_tokens: list[int] = []
        for _proposal_index in range(max_tokens):
            token_id = self._sample_speculative_token(
                next_logits,
                state,
                token_history=token_history,
                generated_history=generated_history,
            )
            draft_tokens.append(int(token_id))
            token_history.append(int(token_id))
            generated_history.append(int(token_id))
            if int(token_id) in state.eos_token_ids:
                break
            temp_decode_state, next_logits = self._advance_cached_model_one(
                draft_model,
                int(token_id),
                temp_cache_views,
                temp_decode_state,
            )
        stats.draft_tokens += len(draft_tokens)
        return _CachedDraftProposal(
            tokens=draft_tokens,
            cache_views=temp_cache_views,
            decode_state=temp_decode_state,
            next_logits=next_logits,
        )

    def _adopt_draft_proposal_cache(
        self,
        cache_state: _SpeculativeCacheState,
        proposal: _CachedDraftProposal,
    ) -> None:
        """Promote an accepted draft proposal's temporary cache metadata."""
        if not proposal.tokens:
            return
        self._adopt_cache_view_storage(cache_state.draft_cache_views, proposal.cache_views)
        self._set_cache_slot_length(
            cache_state.draft_cache_views,
            self._cache_slot_length(proposal.cache_views),
        )
        cache_state.draft_decode_state = self._clone_decode_state(proposal.decode_state)
        cache_state.draft_next_logits = proposal.next_logits

    def _commit_target_tokens_cached(
        self,
        cache_state: _SpeculativeCacheState,
        token_ids: list[int],
        *,
        stats: _SpeculativeDecodeStats,
    ) -> None:
        """Advance the real target cache through verified output tokens."""
        if not token_ids:
            return
        cache_state.target_decode_state, cache_state.target_next_logits = self._advance_cached_model_tokens(
            self.model,
            [int(token_id) for token_id in token_ids],
            cache_state.target_cache_views,
            cache_state.target_decode_state,
        )
        stats.target_steps += 1

    def _advance_dflash_target_tokens_cached(
        self,
        cache_state: _DFlashSpeculativeCacheState,
        token_ids: list[int],
        *,
        stats: _SpeculativeDecodeStats,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Advance the real target cache and capture features for DFlash."""
        logits, target_decode_state, hidden_states = self._cached_forward_logits_and_hidden(
            self.model,
            [int(token_id) for token_id in token_ids],
            cache_state.target_cache_views,
            cache_state.target_decode_state,
            feature_layer_indices=getattr(self._speculative_model, "target_feature_layer_indices", None),
        )
        cache_state.target_decode_state = target_decode_state
        cache_state.target_next_logits = self._last_logits_row(logits)
        stats.target_steps += 1
        return logits, hidden_states

    def _sample_dflash_tail_tokens_cached(
        self,
        state: _RuntimeRequestState,
        *,
        first_token: int,
        max_tail_tokens: int,
        cache_state: _DFlashSpeculativeCacheState,
        stats: _SpeculativeDecodeStats,
        first_token_in_history: bool = False,
    ) -> list[int]:
        """Run the stateful DFlash drafter and sample continuation tokens."""
        proposer = self._speculative_model
        if proposer is None or max_tail_tokens <= 0:
            return []
        logits_method = getattr(proposer, "dflash_logits", None)
        if not callable(logits_method):
            raise RuntimeError("DFlash speculative_model must expose `dflash_logits(...)`.")

        draft_logits = _call_with_filtered_kwargs(
            logits_method,
            first_token=int(first_token),
            hidden_states=cache_state.dflash_hidden_states,
            max_tokens=int(max_tail_tokens),
            target_model=self.model,
            cache=cache_state.dflash_cache,
        )
        if not isinstance(draft_logits, mx.array):
            draft_logits = mx.array(draft_logits)
        if draft_logits.ndim == 3:
            draft_logits = draft_logits.reshape((-1, int(draft_logits.shape[-1])))

        token_history = list(state.request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        if not first_token_in_history:
            token_history.append(int(first_token))
            generated_history.append(int(first_token))
        draft_tail: list[int] = []
        for row in draft_logits:
            if len(draft_tail) >= max_tail_tokens:
                break
            token_id = self._sample_speculative_token(
                row,
                state,
                token_history=token_history,
                generated_history=generated_history,
            )
            draft_tail.append(int(token_id))
            token_history.append(int(token_id))
            generated_history.append(int(token_id))
            if int(token_id) in state.eos_token_ids:
                break

        stats.draft_tokens += (0 if first_token_in_history else 1) + len(draft_tail)
        return draft_tail

    @staticmethod
    def _should_disable_dflash_speculation(stats: _SpeculativeDecodeStats) -> bool:
        """Return whether cached DFlash should fall back to target-only decode."""
        if stats.fallback_triggered:
            return True
        if stats.acceptance_attempts < _DFLASH_ADAPTIVE_MIN_ATTEMPTS:
            return False
        if stats.rejected_tokens <= 0:
            return False
        return stats.acceptance_rate < _DFLASH_ADAPTIVE_MIN_ACCEPTANCE_RATE

    def _dflash_target_decode_token_cached(
        self,
        state: _RuntimeRequestState,
        cache_state: _DFlashSpeculativeCacheState,
        *,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Decode one token from the target cache after disabling DFlash."""
        request = state.request
        if request.remaining_generation_budget <= 0:
            return []

        if cache_state.pending_token is not None:
            pending_token = int(cache_state.pending_token)
            cache_state.pending_token = None
            self._commit_target_tokens_cached(cache_state, [pending_token], stats=stats)

        token_history = list(request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        token_id = self._sample_speculative_token(
            cache_state.target_next_logits,
            state,
            token_history=token_history,
            generated_history=generated_history,
        )
        self._commit_target_tokens_cached(cache_state, [int(token_id)], stats=stats)
        stats.fallback_tokens += 1
        return [int(token_id)]

    def _dflash_speculative_decode_tokens_cached(
        self,
        state: _RuntimeRequestState,
        cache_state: _DFlashSpeculativeCacheState,
        *,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Run one cached DFlash speculative verification round."""
        request = state.request
        remaining = request.remaining_generation_budget
        if remaining <= 0:
            return []

        if cache_state.adaptive_disabled or self._should_disable_dflash_speculation(stats):
            if not cache_state.adaptive_disabled:
                cache_state.adaptive_disabled = True
                stats.fallback_triggered = True
                stats.fallback_reason = (
                    f"acceptance_rate={stats.acceptance_rate:.3f}<{_DFLASH_ADAPTIVE_MIN_ACCEPTANCE_RATE:.2f}"
                )
                logger.debug(
                    "Disabling DFlash speculation for request %s after %d attempts: %.3f acceptance",
                    request.request_id,
                    stats.acceptance_attempts,
                    stats.acceptance_rate,
                )
            return self._dflash_target_decode_token_cached(state, cache_state, stats=stats)

        block_size = max(int(self.speculative_config.num_speculative_tokens), 1)
        pending_token = cache_state.pending_token
        first_token_in_history = pending_token is not None
        if first_token_in_history:
            first_token = int(pending_token)
            cache_state.pending_token = None
        else:
            first_token = self._sample_speculative_token(
                cache_state.target_next_logits,
                state,
                token_history=list(request.all_token_ids),
                generated_history=list(state.generated_token_ids),
            )

        output_tokens: list[int] = []
        token_history = list(request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        if not first_token_in_history:
            output_tokens.append(int(first_token))
            token_history.append(int(first_token))
            generated_history.append(int(first_token))
            stats.accepted_tokens += 1

            if not state.generated_token_ids:
                _, hidden_states = self._advance_dflash_target_tokens_cached(
                    cache_state,
                    [int(first_token)],
                    stats=stats,
                )
                cache_state.dflash_hidden_states = hidden_states
                stats.draft_tokens += 1
                return output_tokens[:remaining]

        if int(first_token) in state.eos_token_ids:
            if not first_token_in_history:
                _, hidden_states = self._advance_dflash_target_tokens_cached(
                    cache_state,
                    [int(first_token)],
                    stats=stats,
                )
                cache_state.dflash_hidden_states = hidden_states
                stats.draft_tokens += 1
            return output_tokens[:remaining]

        max_tail_tokens = min(block_size - 1, remaining if first_token_in_history else max(remaining - 1, 0))
        if max_tail_tokens <= 0:
            _, hidden_states = self._advance_dflash_target_tokens_cached(
                cache_state,
                [int(first_token)],
                stats=stats,
            )
            cache_state.dflash_hidden_states = hidden_states
            if first_token_in_history and remaining > 0:
                bonus_token = self._sample_speculative_token(
                    cache_state.target_next_logits,
                    state,
                    token_history=token_history,
                    generated_history=generated_history,
                )
                cache_state.pending_token = int(bonus_token)
                output_tokens.append(int(bonus_token))
            else:
                stats.draft_tokens += 1
            return output_tokens[:remaining]

        draft_tail = self._sample_dflash_tail_tokens_cached(
            state,
            first_token=int(first_token),
            max_tail_tokens=max_tail_tokens,
            cache_state=cache_state,
            stats=stats,
            first_token_in_history=first_token_in_history,
        )
        draft_tokens = [int(first_token), *draft_tail]

        temp_cache_views = self._clone_cache_views_for_speculative(cache_state.target_cache_views)
        temp_target_state = self._clone_decode_state(cache_state.target_decode_state)
        verified_logits, verified_target_state, verified_hidden_states = self._cached_forward_logits_and_hidden(
            self.model,
            draft_tokens,
            temp_cache_views,
            temp_target_state,
            feature_layer_indices=getattr(self._speculative_model, "target_feature_layer_indices", None),
        )
        stats.target_steps += 1

        rejected = False
        accepted_tail = 0
        correction_token: int | None = None

        for tail_index, draft_token in enumerate(draft_tail):
            logits_index = tail_index
            if logits_index >= int(verified_logits.shape[0]):
                break
            target_token = self._sample_speculative_token(
                verified_logits[logits_index],
                state,
                token_history=token_history,
                generated_history=generated_history,
            )
            if int(target_token) == int(draft_token):
                output_tokens.append(int(draft_token))
                accepted_tail += 1
                stats.accepted_tokens += 1
                token_history.append(int(draft_token))
                generated_history.append(int(draft_token))
                if int(draft_token) in state.eos_token_ids or len(output_tokens) >= remaining:
                    break
                continue

            correction_token = int(target_token)
            output_tokens.append(correction_token)
            stats.rejected_tokens += 1
            rejected = True
            break

        all_tail_accepted = not rejected and accepted_tail == len(draft_tail)
        if all_tail_accepted:
            self._adopt_cache_view_storage(cache_state.target_cache_views, temp_cache_views)
            self._set_cache_slot_length(cache_state.target_cache_views, self._cache_slot_length(temp_cache_views))
            cache_state.target_decode_state = self._clone_decode_state(verified_target_state)
            cache_state.target_next_logits = self._last_logits_row(verified_logits)
            cache_state.dflash_hidden_states = verified_hidden_states
            if len(output_tokens) < remaining and (not output_tokens or output_tokens[-1] not in state.eos_token_ids):
                bonus_token = self._sample_speculative_token(
                    cache_state.target_next_logits,
                    state,
                    token_history=token_history,
                    generated_history=generated_history,
                )
                cache_state.pending_token = int(bonus_token)
                output_tokens.append(int(bonus_token))
            return output_tokens[:remaining]

        accepted_input_tokens = [int(first_token), *draft_tail[:accepted_tail]]
        _, hidden_states = self._advance_dflash_target_tokens_cached(
            cache_state,
            accepted_input_tokens,
            stats=stats,
        )
        cache_state.dflash_hidden_states = hidden_states
        cache_state.pending_token = correction_token
        return output_tokens[:remaining]

    def _speculative_decode_tokens_cached(
        self,
        state: _RuntimeRequestState,
        cache_state: _SpeculativeCacheState | _DFlashSpeculativeCacheState,
        *,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Run one cached speculative verification round."""
        if isinstance(cache_state, _DFlashSpeculativeCacheState):
            return self._dflash_speculative_decode_tokens_cached(state, cache_state, stats=stats)

        request = state.request
        remaining = request.remaining_generation_budget
        if remaining <= 0:
            return []

        proposal_len = min(self.speculative_config.num_speculative_tokens, remaining)
        draft_proposal = self._draft_speculative_tokens_cached(
            state,
            cache_state,
            max_tokens=proposal_len,
            stats=stats,
        )
        draft_tokens = draft_proposal.tokens
        if not draft_tokens:
            target_token = self._sample_speculative_token(
                cache_state.target_next_logits,
                state,
                token_history=list(request.all_token_ids),
                generated_history=list(state.generated_token_ids),
            )
            self._commit_target_tokens_cached(cache_state, [target_token], stats=stats)
            self._advance_draft_cache_tokens(cache_state, [target_token])
            return [int(target_token)]

        accepted_or_corrected: list[int] = []
        token_history = list(request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        first_draft_token = int(draft_tokens[0])
        first_target_token = self._sample_speculative_token(
            cache_state.target_next_logits,
            state,
            token_history=token_history,
            generated_history=generated_history,
        )
        if int(first_target_token) != first_draft_token:
            stats.rejected_tokens += 1
            correction_token = int(first_target_token)
            self._commit_target_tokens_cached(cache_state, [correction_token], stats=stats)
            self._advance_draft_cache_tokens(cache_state, [correction_token])
            return [correction_token]

        accepted_or_corrected.append(first_draft_token)
        stats.accepted_tokens += 1
        token_history.append(first_draft_token)
        generated_history.append(first_draft_token)
        if first_draft_token in state.eos_token_ids or len(accepted_or_corrected) >= remaining:
            self._commit_target_tokens_cached(cache_state, accepted_or_corrected, stats=stats)
            self._advance_draft_cache_tokens(cache_state, accepted_or_corrected)
            return accepted_or_corrected

        temp_cache_views = self._clone_cache_views_for_speculative(cache_state.target_cache_views)
        temp_target_state = self._clone_decode_state(cache_state.target_decode_state)
        verified_logits, verified_target_state = self._cached_forward_logits(
            self.model,
            [int(token_id) for token_id in draft_tokens],
            temp_cache_views,
            decode_state=temp_target_state,
            prefer_decode_state=True,
        )
        stats.target_steps += 1

        accepted_count = 1
        rejected = False
        for draft_index, draft_token in enumerate(draft_tokens[1:], start=1):
            logits_index = draft_index - 1
            if logits_index >= int(verified_logits.shape[0]):
                break
            target_token = self._sample_speculative_token(
                verified_logits[logits_index],
                state,
                token_history=token_history,
                generated_history=generated_history,
            )
            if int(target_token) == int(draft_token):
                accepted_or_corrected.append(int(draft_token))
                accepted_count += 1
                stats.accepted_tokens += 1
                token_history.append(int(draft_token))
                generated_history.append(int(draft_token))
                if int(draft_token) in state.eos_token_ids or len(accepted_or_corrected) >= remaining:
                    break
                continue

            accepted_or_corrected.append(int(target_token))
            stats.rejected_tokens += 1
            rejected = True
            break

        if not accepted_or_corrected:
            return []

        all_draft_tokens_accepted = not rejected and accepted_count == len(draft_tokens)
        if all_draft_tokens_accepted:
            self._adopt_cache_view_storage(cache_state.target_cache_views, temp_cache_views)
            self._set_cache_slot_length(cache_state.target_cache_views, self._cache_slot_length(temp_cache_views))
            cache_state.target_decode_state = self._clone_decode_state(verified_target_state)
            cache_state.target_next_logits = self._last_logits_row(verified_logits)
            if len(accepted_or_corrected) < remaining and accepted_or_corrected[-1] not in state.eos_token_ids:
                bonus_token = self._sample_speculative_token(
                    cache_state.target_next_logits,
                    state,
                    token_history=token_history,
                    generated_history=generated_history,
                )
                accepted_or_corrected.append(int(bonus_token))
                self._commit_target_tokens_cached(cache_state, [int(bonus_token)], stats=stats)
                self._adopt_draft_proposal_cache(cache_state, draft_proposal)
                self._advance_draft_cache_tokens(cache_state, [int(bonus_token)])
            else:
                self._adopt_draft_proposal_cache(cache_state, draft_proposal)
            return accepted_or_corrected

        self._commit_target_tokens_cached(cache_state, accepted_or_corrected, stats=stats)
        self._advance_draft_cache_tokens(cache_state, accepted_or_corrected)
        return accepted_or_corrected

    def _sample_speculative_token(
        self,
        logits_row: mx.array,
        state: _RuntimeRequestState,
        *,
        token_history: list[int],
        generated_history: list[int],
    ) -> int:
        """Sample one token from a logits row using the existing runner sampler."""
        if logits_row.ndim != 1:
            logits_row = logits_row.reshape((-1,))
        runner = self._runner
        if runner is not None:
            sampled = runner._sample_next_tokens_mx(  # type: ignore[attr-defined]
                mx.expand_dims(logits_row, axis=0),
                [state.request.sampling_params],
                token_histories=[list(token_history)],
                generated_token_histories=[list(generated_history)],
            )
            return int(sampled[0])
        token_id = mx.argmax(logits_row, axis=-1).astype(mx.int64)
        try:
            mx.eval(token_id)
        except RuntimeError:
            pass
        value = token_id.tolist()
        return int(value[0] if isinstance(value, list) else value)

    def _draft_speculative_tokens(
        self,
        state: _RuntimeRequestState,
        *,
        max_tokens: int,
        stats: _SpeculativeDecodeStats,
    ) -> list[int]:
        """Generate a draft token proposal with the configured draft model."""
        if self.speculative_config.speculative_method == "eagle3":
            return self._eagle3_speculative_tokens(state, max_tokens=max_tokens, stats=stats)
        if self.speculative_config.speculative_method == "dflash":
            return self._dflash_speculative_tokens(state, max_tokens=max_tokens, stats=stats)

        draft_model = self._speculative_model
        if draft_model is None or max_tokens <= 0:
            return []

        context = list(state.request.all_token_ids)
        generated_history = list(state.generated_token_ids)
        draft_tokens: list[int] = []
        for _ in range(max_tokens):
            logits = self._forward_full_context_logits(draft_model, context)
            token_id = self._sample_speculative_token(
                logits[-1],
                state,
                token_history=context,
                generated_history=generated_history,
            )
            draft_tokens.append(token_id)
            context.append(token_id)
            generated_history.append(token_id)
            if token_id in state.eos_token_ids:
                break
        stats.draft_tokens += len(draft_tokens)
        return draft_tokens

    def _target_speculative_next_token(
        self,
        state: _RuntimeRequestState,
        *,
        stats: _SpeculativeDecodeStats,
    ) -> int:
        """Generate one token from the target model with a full-context call."""
        context = list(state.request.all_token_ids)
        logits = self._forward_full_context_logits(self.model, context)
        stats.target_steps += 1
        return self._sample_speculative_token(
            logits[-1],
            state,
            token_history=context,
            generated_history=list(state.generated_token_ids),
        )

    def _speculative_decode_tokens(
        self,
        state: _RuntimeRequestState,
        *,
        stats: _SpeculativeDecodeStats,
        cache_state: _SpeculativeCacheState | _DFlashSpeculativeCacheState | None = None,
    ) -> list[int] | None:
        """Run one speculative verification round."""
        if cache_state is not None:
            return self._speculative_decode_tokens_cached(state, cache_state, stats=stats)

        request = state.request
        remaining = request.remaining_generation_budget
        if remaining <= 0:
            return []

        context = list(request.all_token_ids)
        proposal_len = min(self.speculative_config.num_speculative_tokens, remaining)
        draft_tokens = self._draft_speculative_tokens(state, max_tokens=proposal_len, stats=stats)
        if not draft_tokens:
            return [self._target_speculative_next_token(state, stats=stats)]

        try:
            target_logits = self._forward_full_context_logits(self.model, context + draft_tokens)
        except RuntimeError:
            return None if not request.generated_token_ids else [self._target_speculative_next_token(state, stats=stats)]
        stats.target_steps += 1

        start_index = len(context) - 1
        if int(target_logits.shape[0]) <= start_index:
            return None if not request.generated_token_ids else [self._target_speculative_next_token(state, stats=stats)]

        accepted_or_corrected: list[int] = []
        token_history = list(context)
        generated_history = list(state.generated_token_ids)
        for offset, draft_token in enumerate(draft_tokens):
            logits_index = start_index + offset
            if logits_index >= int(target_logits.shape[0]):
                break
            target_token = self._sample_speculative_token(
                target_logits[logits_index],
                state,
                token_history=token_history,
                generated_history=generated_history,
            )
            if target_token == int(draft_token):
                accepted_or_corrected.append(int(draft_token))
                stats.accepted_tokens += 1
                token_history.append(int(draft_token))
                generated_history.append(int(draft_token))
                if int(draft_token) in state.eos_token_ids or len(accepted_or_corrected) >= remaining:
                    return accepted_or_corrected
                continue

            accepted_or_corrected.append(int(target_token))
            stats.rejected_tokens += 1
            return accepted_or_corrected

        if accepted_or_corrected and len(accepted_or_corrected) < remaining:
            bonus_index = start_index + len(draft_tokens)
            if bonus_index < int(target_logits.shape[0]):
                bonus_token = self._sample_speculative_token(
                    target_logits[bonus_index],
                    state,
                    token_history=token_history,
                    generated_history=generated_history,
                )
                accepted_or_corrected.append(int(bonus_token))
        return accepted_or_corrected

    def _append_speculative_token(self, state: _RuntimeRequestState, token_id: int) -> str | None:
        """Append a verified token and return a finish reason if it stops."""
        request = state.request
        request.append_generated_token(int(token_id))
        request.num_computed_tokens = max(request.num_computed_tokens, request.total_tokens)
        request.num_cached_tokens = max(request.num_cached_tokens, request.num_computed_tokens)
        state.note_generated_tokens([int(token_id)])

        finish_reason = self._sync_check_stop_reason(int(token_id), state, request)
        if finish_reason is None and state.tool_parser_instance is not None:
            parsed = self._parse_text_output(
                state.decoded_text(self.tokenizer, stream_safe=True),
                stop=state.stop,
                finish_reason=None,
                gen_kwargs=state.gen_kwargs,
                tool_parser=state.tool_parser_instance,
                reasoning_parser=state.reasoning_parser_instance,
                tool_parser_request=state.tool_parser_request,
            )
            if parsed["tool_calls"]:
                finish_reason = FinishReason.TOOL_CALLS
        if finish_reason is not None:
            request.mark_finished(finish_reason)
            state.finalize()
        return finish_reason

    def _add_speculative_metrics(
        self,
        output: RequestOutput,
        stats: _SpeculativeDecodeStats,
    ) -> RequestOutput:
        """Attach speculative decoding counters to a request output."""
        metrics = dict(output.metrics or {})
        metrics.update(
            stats.as_metrics(
                enabled=True,
                num_speculative_tokens=self.speculative_config.num_speculative_tokens,
                method=self.speculative_config.speculative_method,
            )
        )
        output.metrics = metrics
        return output

    def _generate_speculative_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> RequestOutput | None:
        """Single-request exact greedy speculative decoding path."""
        preconditions = self._sync_validate_preconditions(sampling_params)
        if preconditions is None:
            return None
        _runner, scheduler = preconditions

        gen_kwargs, max_new_tokens, _ = self._sync_prepare_kwargs(sampling_params, **generate_kwargs)
        states = self._build_requests([prompt], gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return None

        state = states[0]
        request = state.request
        if not self._can_use_speculative_sync_path(state, gen_kwargs):
            return None

        paused_for_sync = False
        stats = _SpeculativeDecodeStats()
        try:
            with self._work_cv:
                if self._shutdown.is_set() or self._paused or self._runtime_states or scheduler.has_pending_work():
                    return None
                self._paused = True
                scheduler.pause()
                paused_for_sync = True

            if request.max_new_tokens <= 0:
                request.mark_finished(FinishReason.LENGTH)
                state.finalize()
                return self._add_speculative_metrics(self._state_to_request_output(state), stats)

            cache_state = self._initialize_speculative_cache_state(state)
            while not request.is_finished:
                token_ids = self._speculative_decode_tokens(state, stats=stats, cache_state=cache_state)
                if token_ids is None:
                    return (
                        None
                        if not request.generated_token_ids
                        else self._add_speculative_metrics(
                            self._state_to_request_output(state),
                            stats,
                        )
                    )
                if not token_ids:
                    request.mark_finished(FinishReason.LENGTH)
                    state.finalize()
                    break
                for token_id in token_ids:
                    self._append_speculative_token(state, token_id)
                    if request.is_finished:
                        break
                if request.remaining_generation_budget <= 0 and not request.is_finished:
                    request.mark_finished(FinishReason.LENGTH)
                    state.finalize()

            return self._add_speculative_metrics(self._state_to_request_output(state), stats)
        except Exception as exc:
            if not request.generated_token_ids:
                logger.debug("Speculative decoding unavailable for request %s: %s", request.request_id, exc)
                return None
            if not request.is_finished:
                state.mark_error(exc)
            raise
        finally:
            self._reset_recurrent_states()
            if self._speculative_model is not None:
                self._reset_recurrent_states(self._speculative_model)
            if paused_for_sync:
                with self._work_cv:
                    self._paused = False
                    scheduler.resume()
                    self._work_cv.notify_all()

    def _stream_speculative_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> Iterator[RequestOutput] | None:
        """Single-request exact greedy speculative decoding path for streams."""
        preconditions = self._sync_validate_preconditions(sampling_params)
        if preconditions is None:
            return None
        _runner, scheduler = preconditions

        gen_kwargs, max_new_tokens, _ = self._sync_prepare_kwargs(sampling_params, **generate_kwargs)
        states = self._build_requests([prompt], gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return None

        state = states[0]
        request = state.request
        if not self._can_use_speculative_sync_path(state, gen_kwargs):
            return None

        def _iterator() -> Iterator[RequestOutput]:
            paused_for_sync = False
            stats = _SpeculativeDecodeStats()
            try:
                with self._work_cv:
                    if self._shutdown.is_set() or self._paused or self._runtime_states or scheduler.has_pending_work():
                        return
                    self._paused = True
                    scheduler.pause()
                    paused_for_sync = True

                if request.max_new_tokens <= 0:
                    request.mark_finished(FinishReason.LENGTH)
                    state.finalize()
                    yield self._add_speculative_metrics(
                        self._state_to_request_output(state, incremental=True, consume=True),
                        stats,
                    )
                    return

                cache_state = self._initialize_speculative_cache_state(state)
                while not request.is_finished:
                    token_ids = self._speculative_decode_tokens(state, stats=stats, cache_state=cache_state)
                    if token_ids is None:
                        if request.generated_token_ids:
                            yield self._add_speculative_metrics(
                                self._state_to_request_output(state, incremental=True, consume=True),
                                stats,
                            )
                        return
                    if not token_ids:
                        request.mark_finished(FinishReason.LENGTH)
                        state.finalize()
                        yield self._add_speculative_metrics(
                            self._state_to_request_output(state, incremental=True, consume=True),
                            stats,
                        )
                        break

                    for token_id in token_ids:
                        self._append_speculative_token(state, token_id)
                        if request.is_finished:
                            break
                    if request.remaining_generation_budget <= 0 and not request.is_finished:
                        request.mark_finished(FinishReason.LENGTH)
                        state.finalize()

                    yield self._add_speculative_metrics(
                        self._state_to_request_output(state, incremental=True, consume=True),
                        stats,
                    )
            except GeneratorExit:
                raise
            except Exception as exc:
                if not request.generated_token_ids:
                    logger.debug("Speculative streaming unavailable for request %s: %s", request.request_id, exc)
                    return
                if not request.is_finished:
                    state.mark_error(exc)
                raise
            finally:
                self._reset_recurrent_states()
                if self._speculative_model is not None:
                    self._reset_recurrent_states(self._speculative_model)
                if paused_for_sync:
                    with self._work_cv:
                        self._paused = False
                        scheduler.resume()
                        self._work_cv.notify_all()

        return _iterator()

    def _generate_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> RequestOutput | None:
        """Synchronous single-request fast path with no per-step worker overhead.

        Bypasses the background scheduling loop entirely for single-request
        generation. Runs prefill + decode in a tight Python loop on the
        calling thread, eliminating ~3ms/step of threading overhead.

        Returns ``None`` if the fast path cannot be used (e.g. runner not
        initialized), in which case the caller should fall back to the
        background loop.
        """
        preconditions = self._sync_validate_preconditions(sampling_params)
        if preconditions is None:
            return None
        runner, scheduler = preconditions

        gen_kwargs, max_new_tokens, _ = self._sync_prepare_kwargs(sampling_params, **generate_kwargs)

        states = self._build_requests([prompt], gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return None

        state = states[0]
        request = state.request
        request_id = request.request_id
        if request.max_new_tokens <= 0:
            request.mark_finished(FinishReason.LENGTH)
            return self._state_to_request_output(state)

        greedy_sampling = request.sampling_params is None or not bool(request.sampling_params.do_sample)

        paused_for_sync = False
        runtime_state_added = False
        request_added = False
        row_index: int | None = None
        sequence: ScheduledSequence | None = None
        execution_request: ExecutionRequest | None = None
        finish_reason: str | None = None
        pending_decode_token: int | None = None

        try:
            with self._work_cv:
                if self._shutdown.is_set() or self._paused or self._runtime_states or scheduler.has_pending_work():
                    return None
                self._paused = True
                scheduler.pause()
                paused_for_sync = True

            with self._work_cv:
                self._runtime_states[request_id] = state
                runtime_state_added = True
            scheduler.add_request(request)
            request_added = True

            while pending_decode_token is None and finish_reason is None:
                step_output = scheduler.schedule()
                if not step_output.scheduled:
                    if request.remaining_generation_budget <= 0:
                        finish_reason = FinishReason.LENGTH
                        break
                    raise RuntimeError("Scheduler produced no work for synchronous generation.")

                entry = step_output.scheduled[0]
                row_index = int(entry.row_index)
                if sequence is None:
                    runner.bind_request(request, row_index)
                    sequence = ScheduledSequence(
                        request_id=request_id,
                        row_index=row_index,
                        token_ids=[int(token_id) for token_id in entry.token_ids],
                        num_computed_tokens=int(request.num_computed_tokens),
                        page_ids=tuple(int(page_id) for page_id in entry.page_ids),
                        meta={"is_prefill": entry.is_prefill},
                    )
                    execution_request = ExecutionRequest(
                        step_id=0,
                        mode="prefill" if entry.is_prefill else "decode",
                        sequences=[sequence],
                    )
                else:
                    sequence.token_ids[:] = [int(token_id) for token_id in entry.token_ids]
                    object.__setattr__(sequence, "num_computed_tokens", int(request.num_computed_tokens))
                    if tuple(int(page_id) for page_id in entry.page_ids) != sequence.page_ids:
                        object.__setattr__(sequence, "page_ids", tuple(int(page_id) for page_id in entry.page_ids))
                    execution_request.mode = "prefill" if entry.is_prefill else "decode"

                assert execution_request is not None
                self._step_counter += 1
                execution_request.step_id = self._step_counter
                execution_request.multimodal = self._prepare_multimodal_batch(step_output) if entry.is_prefill else None

                raw_output, logits = self._sync_execute_step(runner, execution_request, step_output)

                request.consume_computed_tokens(entry.num_tokens)
                runner.sequence_buffer.set_num_computed_tokens(request_id, request.num_computed_tokens)

                if entry.is_prefill and request.remaining_prefill_tokens > 0:
                    continue

                sampled_token = self._sync_sample_token(
                    raw_output, logits, runner, greedy_sampling, request.sampling_params, request
                )
                request.append_generated_token(sampled_token)
                runner.sequence_buffer.append_output_tokens(request_id, [sampled_token])
                state.note_generated_tokens([sampled_token])
                finish_reason = self._sync_check_stop_reason(sampled_token, state, request)
                if finish_reason is None:
                    pending_decode_token = sampled_token

            while pending_decode_token is not None and finish_reason is None:
                if sequence is None or execution_request is None or row_index is None:
                    raise RuntimeError("Synchronous decode state was not initialized.")

                decode_step = self._sync_decode_step_output(pending_decode_token, row_index, request, scheduler)
                sequence.token_ids[:] = [int(pending_decode_token)]
                object.__setattr__(sequence, "num_computed_tokens", int(request.num_computed_tokens))
                execution_request.mode = "decode"
                execution_request.multimodal = None
                self._step_counter += 1
                execution_request.step_id = self._step_counter

                raw_output, logits = self._sync_execute_step(runner, execution_request, decode_step)

                request.consume_computed_tokens(1)
                runner.sequence_buffer.set_num_computed_tokens(request_id, request.num_computed_tokens)

                sampled_token = self._sync_sample_token(
                    raw_output, logits, runner, greedy_sampling, request.sampling_params, request
                )
                request.append_generated_token(sampled_token)
                runner.sequence_buffer.append_output_tokens(request_id, [sampled_token])
                state.note_generated_tokens([sampled_token])
                finish_reason = self._sync_check_stop_reason(sampled_token, state, request)
                pending_decode_token = None if finish_reason is not None else sampled_token

            if finish_reason is None:
                if request.remaining_generation_budget <= 0:
                    finish_reason = FinishReason.LENGTH
                elif request.finished_reason is not None:
                    finish_reason = request.finished_reason
                else:
                    raise RuntimeError("Synchronous generation exited without a finish reason.")

            if request_added and not request.is_finished:
                scheduler._finalize_request(request, reason=finish_reason)
            return self._state_to_request_output(state)
        except BaseException as exc:
            if request_added and not request.is_finished:
                request.mark_failed(str(exc))
                scheduler._finalize_request(request, reason=FinishReason.ERROR)
            raise
        finally:
            self._sync_finalize(
                paused_for_sync=paused_for_sync,
                runtime_state_added=runtime_state_added,
                request_id=request_id,
                runner=runner,
                scheduler=scheduler,
            )

    def _stream_sync(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        **generate_kwargs: Any,
    ) -> Iterator[RequestOutput] | None:
        """Synchronous single-request streaming fast path.

        Uses the same direct-runner path as :meth:`_generate_sync`, but
        yields incremental :class:`RequestOutput` updates after each
        token. Returns ``None`` when the fast path cannot be used.
        """
        preconditions = self._sync_validate_preconditions(sampling_params)
        if preconditions is None:
            return None
        runner, scheduler = preconditions

        gen_kwargs, max_new_tokens, _ = self._sync_prepare_kwargs(sampling_params, **generate_kwargs)

        states = self._build_requests([prompt], gen_kwargs, max_new_tokens=max_new_tokens)
        if not states:
            return None

        state = states[0]
        request = state.request
        request_id = request.request_id

        greedy_sampling = request.sampling_params is None or not bool(request.sampling_params.do_sample)

        def _check_tool_call_stop() -> str | None:
            if state.tool_parser_instance is None:
                return None
            parsed = self._parse_text_output(
                state.decoded_text(self.tokenizer, stream_safe=True),
                stop=state.stop,
                finish_reason=None,
                gen_kwargs=state.gen_kwargs,
                tool_parser=state.tool_parser_instance,
                reasoning_parser=state.reasoning_parser_instance,
                tool_parser_request=state.tool_parser_request,
            )
            return FinishReason.TOOL_CALLS if parsed["tool_calls"] else None

        def _iterator() -> Iterator[RequestOutput]:
            paused_for_sync = False
            runtime_state_added = False
            request_added = False
            row_index: int | None = None
            sequence: ScheduledSequence | None = None
            execution_request: ExecutionRequest | None = None
            pending_decode_token: int | None = None

            try:
                if request.max_new_tokens <= 0:
                    request.mark_finished(FinishReason.LENGTH)
                    yield self._state_to_request_output(state, incremental=True, consume=True)
                    return

                with self._work_cv:
                    if self._shutdown.is_set() or self._paused or self._runtime_states or scheduler.has_pending_work():
                        return
                    self._paused = True
                    scheduler.pause()
                    paused_for_sync = True

                with self._work_cv:
                    self._runtime_states[request_id] = state
                    runtime_state_added = True
                scheduler.add_request(request)
                request_added = True

                while True:
                    if request.is_finished:
                        if not state.last_emitted_finished:
                            yield self._state_to_request_output(state, incremental=True, consume=True)
                        break

                    if pending_decode_token is None:
                        step_output = scheduler.schedule()
                        if not step_output.scheduled:
                            if request.remaining_generation_budget <= 0:
                                scheduler._finalize_request(request, reason=FinishReason.LENGTH)
                                continue
                            raise RuntimeError("Scheduler produced no work for synchronous streaming generation.")

                        entry = step_output.scheduled[0]
                        row_index = int(entry.row_index)
                        if sequence is None:
                            runner.bind_request(request, row_index)
                            sequence = ScheduledSequence(
                                request_id=request_id,
                                row_index=row_index,
                                token_ids=[int(token_id) for token_id in entry.token_ids],
                                num_computed_tokens=int(request.num_computed_tokens),
                                page_ids=tuple(int(page_id) for page_id in entry.page_ids),
                                meta={"is_prefill": entry.is_prefill},
                            )
                            execution_request = ExecutionRequest(
                                step_id=0,
                                mode="prefill" if entry.is_prefill else "decode",
                                sequences=[sequence],
                            )
                        else:
                            sequence.token_ids[:] = [int(token_id) for token_id in entry.token_ids]
                            object.__setattr__(sequence, "num_computed_tokens", int(request.num_computed_tokens))
                            if tuple(int(page_id) for page_id in entry.page_ids) != sequence.page_ids:
                                object.__setattr__(
                                    sequence, "page_ids", tuple(int(page_id) for page_id in entry.page_ids)
                                )
                            execution_request.mode = "prefill" if entry.is_prefill else "decode"

                        assert execution_request is not None
                        self._step_counter += 1
                        execution_request.step_id = self._step_counter
                        execution_request.multimodal = (
                            self._prepare_multimodal_batch(step_output) if entry.is_prefill else None
                        )

                        raw_output, logits = self._sync_execute_step(runner, execution_request, step_output)

                        if request.is_finished:
                            continue

                        request.consume_computed_tokens(entry.num_tokens)
                        runner.sequence_buffer.set_num_computed_tokens(request_id, request.num_computed_tokens)

                        if entry.is_prefill and request.remaining_prefill_tokens > 0:
                            continue

                        sampled_token = self._sync_sample_token(
                            raw_output, logits, runner, greedy_sampling, request.sampling_params, request
                        )
                    else:
                        if sequence is None or execution_request is None or row_index is None:
                            raise RuntimeError("Synchronous decode state was not initialized.")

                        decode_step = self._sync_decode_step_output(pending_decode_token, row_index, request, scheduler)
                        sequence.token_ids[:] = [int(pending_decode_token)]
                        object.__setattr__(sequence, "num_computed_tokens", int(request.num_computed_tokens))
                        execution_request.mode = "decode"
                        execution_request.multimodal = None
                        self._step_counter += 1
                        execution_request.step_id = self._step_counter

                        raw_output, logits = self._sync_execute_step(runner, execution_request, decode_step)

                        if request.is_finished:
                            continue

                        request.consume_computed_tokens(1)
                        runner.sequence_buffer.set_num_computed_tokens(request_id, request.num_computed_tokens)
                        sampled_token = self._sync_sample_token(
                            raw_output, logits, runner, greedy_sampling, request.sampling_params, request
                        )

                    if request.is_finished:
                        continue

                    request.append_generated_token(sampled_token)
                    runner.sequence_buffer.append_output_tokens(request_id, [sampled_token])
                    state.note_generated_tokens([sampled_token])

                    stop_reason = self._sync_check_stop_reason(sampled_token, state, request)
                    if stop_reason is None:
                        stop_reason = _check_tool_call_stop()

                    if stop_reason is not None:
                        scheduler._finalize_request(request, reason=stop_reason)
                        pending_decode_token = None
                    else:
                        pending_decode_token = sampled_token

                    yield self._state_to_request_output(state, incremental=True, consume=True)

            except GeneratorExit:
                raise
            except BaseException as exc:
                if request_added and not request.is_finished:
                    request.mark_failed(str(exc))
                    scheduler._finalize_request(request, reason=FinishReason.ERROR)
                raise
            finally:
                if request_added and not request.is_finished:
                    scheduler._finalize_request(request, reason=FinishReason.CANCELED)
                self._sync_finalize(
                    paused_for_sync=paused_for_sync,
                    runtime_state_added=runtime_state_added,
                    request_id=request_id,
                    runner=runner,
                    scheduler=scheduler,
                )

        return _iterator()

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

        if len(prompts_list) == 1 and not self._paused:
            speculative_iterator = self._stream_speculative_sync(
                prompts_list[0],
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            if speculative_iterator is not None:
                yield from speculative_iterator
                return
            iterator = self._stream_sync(
                prompts_list[0],
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            if iterator is not None:
                yield from iterator
                return

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
        fallback_prompt = _plain_chat_prompt(messages, add_generation_prompt=add_generation_prompt)
        if not hasattr(self.tokenizer, "apply_chat_template"):
            logger.debug("Tokenizer has no apply_chat_template; using plain chat prompt fallback")
            return fallback_prompt

        normalized_tools = normalize_chat_template_tools(tools)
        template_messages = normalize_chat_template_messages(messages)
        structured_messages = to_structured_text_messages(template_messages)
        extra = dict(chat_template_kwargs or {})
        if chat_template is not None:
            extra["chat_template"] = chat_template
        has_template = bool(extra.get("chat_template") or getattr(self.tokenizer, "chat_template", None))
        if not has_template:
            logger.debug("Tokenizer has no chat_template; using plain chat prompt fallback")
            return fallback_prompt

        attempts = (
            (template_messages, normalized_tools),
            (structured_messages, normalized_tools),
            (structured_messages, None),
        )
        last_error: Exception | None = None
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
            except TypeError as exc:
                last_error = exc
                try:
                    result = self.tokenizer.apply_chat_template(
                        candidate_messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        **extra,
                    )
                    logger.debug("Chat template rendered (without tools) on attempt %d", attempt_idx + 1)
                    return result
                except Exception as retry_exc:
                    last_error = retry_exc
                    logger.debug(
                        "Chat template attempt %d failed without tools, trying next: %s",
                        attempt_idx + 1,
                        retry_exc,
                    )
                    continue
            except Exception as exc:
                last_error = exc
                logger.debug("Chat template attempt %d failed, trying next: %s", attempt_idx + 1, exc)
                continue
        logger.debug("All chat template attempts failed; using plain chat prompt fallback: %s", last_error)
        return fallback_prompt

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
        model = cls._load_pretrained_model(
            pretrained_model_name_or_path,
            model_class=model_class,
            revision=revision,
            local_files_only=local_files_only,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            model_kwargs=model_kwargs,
        )
        return cls(model, tokenizer=tokenizer, **engine_kwargs)

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        self.close()
