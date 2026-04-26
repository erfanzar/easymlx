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

"""FastAPI + OpenAI-compatible layer for easymlx eSurge.

This module implements the ``eSurgeApiServer`` class, a full-featured
OpenAI-compatible API server built on FastAPI. It exposes endpoints for:

- Text completions (``/v1/completions``)
- Chat completions (``/v1/chat/completions``)
- Responses API (``/v1/responses``)
- Model listing and info (``/v1/models``)
- Admin key management (``/v1/admin/keys``)
- Health and metrics endpoints

The server wraps one or more ``eSurge`` engine instances and supports
streaming, engine-owned tool calling/reasoning extraction, and rate-limited
API key authentication.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from easymlx.workers.esurge.admin_state import AdminState
from easymlx.workers.esurge.auth_endpoints import AuthEndpointsMixin

from ..esurge_engine import eSurge
from ..metrics import MetricsCollector
from ..outputs import RequestOutput
from ..sampling_params import SamplingParams
from .api_models import (
    AdminAuditEntry,
    AdminAuditLogResponse,
    AdminKeyCreateRequest,
    AdminKeyResponse,
    AdminKeyStatsResponse,
    AdminKeyUpdateRequest,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatStreamChoice,
    Choice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    FunctionCall,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ResponseOutputText,
    ResponsesOutput,
    ResponsesRequest,
    ResponsesResponse,
    ToolCall,
    Usage,
)


class eSurgeApiServer(AuthEndpointsMixin):
    """OpenAI-style FastAPI wrapper around one or more easymlx eSurge engines.

    Provides a complete REST API that is compatible with OpenAI client libraries.
    Supports multiple models, streaming responses, engine-owned
    tool/function calling, admin API key management with rate limiting,
    and server metrics collection.

    Attributes:
        engines: Dictionary mapping model names to ``eSurge`` engine instances.
        app: The FastAPI application instance.
        status: Current server status string.
        admin_state: Admin state managing API keys, audit logs, and quotas.
        require_api_key: Whether API key authentication is required.
        admin_api_key: Optional master admin API key.
        metrics: Metrics collector for request tracking.
    """

    def __init__(
        self,
        engines: dict[str, eSurge],
        *,
        title: str | None = None,
        tools: dict[str, Any] | list[Any] | None = None,
        require_api_key: bool = False,
        admin_api_key: str | None = None,
        enable_response_store: bool = True,
        default_store_responses: bool = True,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
        response_store_dir: str | None = None,
        response_store: Any | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        """Initialize the API server with engines and configuration.

        Args:
            engines: Dictionary mapping model names to ``eSurge`` engine
                instances to serve.
            title: Optional title for the FastAPI application.
            tools: Deprecated server-side registry input. Ignored; tools are
                request metadata consumed by the engine.
            require_api_key: Whether to require API key authentication.
            admin_api_key: Optional master admin API key for admin endpoints.
            enable_response_store: Deprecated and ignored; response
                persistence is not owned by this server.
            default_store_responses: Deprecated and ignored.
            max_stored_responses: Deprecated and ignored.
            max_stored_conversations: Deprecated and ignored.
            response_store_dir: Deprecated and ignored.
            response_store: Deprecated and ignored.
            metrics_collector: Optional pre-configured metrics collector.
        """
        del tools, default_store_responses, max_stored_responses, max_stored_conversations
        del response_store_dir, response_store
        del enable_response_store
        self.engines = dict(engines)
        self.app = FastAPI(title=title or "easymlx eSurge API")
        self.status = "ready"
        self.admin_state = AdminState()
        self.require_api_key = bool(require_api_key)
        self.admin_api_key = admin_api_key
        self.enable_response_store = False
        self.default_store_responses = False
        self.response_store = None
        self.metrics = metrics_collector or MetricsCollector()
        self._register_routes()

    def register_tool(
        self,
        name: str,
        handler: Any,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        """Reject server-owned tool registration.

        eSurge owns tool-call parsing and the API server does not execute
        tools.
        """
        del name, handler, description, parameters, strict
        raise RuntimeError("Server-owned tool registration was removed; pass tools on inference requests.")

    def _register_routes(self) -> None:
        """Register all HTTP route handlers with the FastAPI application."""
        self.app.get("/health")(self._health)
        self.app.get("/metrics")(self._metrics)
        self.app.get("/v1/models")(self._list_models)
        self.app.get("/v1/models/{model_id}")(self._get_model)
        self.app.post("/v1/completions")(self._complete)
        self.app.post("/v1/chat/completions")(self._chat_complete)
        self.app.post("/v1/responses")(self._responses)
        self.app.post("/v1/admin/keys")(self._create_admin_key)
        self.app.get("/v1/admin/keys/stats")(self._admin_key_stats)
        self.app.get("/v1/admin/audit-logs")(self._admin_audit_logs)
        self.app.get("/v1/admin/keys")(self._list_admin_keys)
        self.app.get("/v1/admin/keys/{key_id}")(self._get_admin_key)
        self.app.patch("/v1/admin/keys/{key_id}")(self._update_admin_key)
        self.app.delete("/v1/admin/keys/{key_id}/revoke")(self._revoke_admin_key)
        self.app.post("/v1/admin/keys/{key_id}/suspend")(self._suspend_admin_key)
        self.app.post("/v1/admin/keys/{key_id}/reactivate")(self._reactivate_admin_key)
        self.app.delete("/v1/admin/keys/{key_id}")(self._delete_admin_key)
        self.app.post("/v1/admin/keys/{key_id}/rotate")(self._rotate_admin_key)

    def _get_engine(self, model: str) -> eSurge:
        """Look up an engine by model name.

        Args:
            model: The model name to look up.

        Returns:
            The ``eSurge`` engine instance for the given model.

        Raises:
            HTTPException: 404 if the model is not found.
        """
        engine = self.engines.get(model)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"Model {model!r} not found")
        return engine

    @staticmethod
    def _merge_tools(
        tools: list[dict[str, Any]] | None,
        functions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Merge tools and legacy functions into a unified tool list.

        Prefers ``tools`` if provided. Falls back to converting ``functions``
        to OpenAI tool format.

        Args:
            tools: OpenAI-format tool definitions, or ``None``.
            functions: Legacy function definitions, or ``None``.

        Returns:
            A list of OpenAI-format tool dicts, or ``None`` if both inputs
            are empty/``None``.
        """
        if tools:
            return tools
        if not functions:
            return None
        normalized: list[dict[str, Any]] = []
        for function in functions:
            if not isinstance(function, dict):
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": dict(function),
                }
            )
        return normalized or None

    def _build_sampling_params(
        self,
        *,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list["str"] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        functions: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> SamplingParams:
        """Build a ``SamplingParams`` object from request parameters.

        Args:
            max_tokens: Maximum tokens to generate (defaults to 16 if ``None``).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            presence_penalty: Additive penalty for generated-token repeats.
            repetition_penalty: Multiplicative penalty for prompt/output repeats.
            stop: Stop sequence(s).
            tools: Tool definitions for function calling.
            functions: Legacy function definitions.
            tool_choice: Tool selection strategy.

        Returns:
            A configured ``SamplingParams`` instance.
        """
        merged_tools = self._merge_tools(tools, functions)
        return SamplingParams(
            max_tokens=max_tokens or 16,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            presence_penalty=float(presence_penalty),
            repetition_penalty=float(repetition_penalty),
            stop=[stop] if isinstance(stop, str) else stop,
            do_sample=bool(float(temperature) > 0 and (float(temperature) != 0.0 or top_k or top_p < 1.0)),
            tools=merged_tools,
            tool_choice=tool_choice,
        )

    @staticmethod
    def _usage_counts(output: RequestOutput) -> tuple[int, int]:
        """Return prompt and completion token counts for a request output."""
        metrics = output.metrics or {}
        prompt_tokens = int(metrics.get("num_computed_tokens", len(output.prompt_token_ids)))
        completion_tokens = int(
            metrics.get("num_generated_tokens", output.num_generated_tokens or len(output.outputs[0].token_ids))
        )
        return prompt_tokens, completion_tokens

    @classmethod
    def _build_usage(cls, output: RequestOutput) -> Usage:
        """Build a ``Usage`` object from a request output's metrics.

        Args:
            output: The completed request output containing metrics.

        Returns:
            A ``Usage`` instance with prompt, completion, and total token counts.
        """
        prompt_tokens, completion_tokens = cls._usage_counts(output)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _record_stream_progress(self, started_at: float, output: RequestOutput | None) -> None:
        """Record live generated-token progress for a streaming response."""
        if output is None:
            return
        prompt_tokens, completion_tokens = self._usage_counts(output)
        self.metrics.record_request_progress(
            started_at,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _finish_metrics(
        self,
        started_at: float,
        *,
        success: bool,
        output: RequestOutput | None = None,
        raw_request: Request | None = None,
    ) -> None:
        """Record request completion in the metrics collector and API key usage.

        Args:
            started_at: Timestamp from ``metrics.start_request()``.
            success: Whether the request completed successfully.
            output: The request output (used to extract token counts).
            raw_request: The raw FastAPI request (used for API key tracking).
        """
        prompt_tokens = 0
        completion_tokens = 0
        if output is not None:
            usage = self._build_usage(output)
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        self.metrics.finish_request(
            started_at,
            success=success,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if success and output is not None:
            self._record_api_key_usage(
                raw_request,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    @staticmethod
    def _normalize_conversation_id(value: Any) -> str | None:
        """Extract a conversation ID from a string or dict value.

        Args:
            value: Raw conversation field from the request. Can be a string,
                a dict with ``"id"``/``"conversation_id"``/``"conversation"``
                keys, or ``None``.

        Returns:
            A stripped non-empty string conversation ID, or ``None``.
        """
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            conversation_id = value.get("id") or value.get("conversation_id") or value.get("conversation")
            if isinstance(conversation_id, str):
                return conversation_id.strip() or None
        return None

    @staticmethod
    def _responses_reasoning_summary_requested(payload: dict[str, Any]) -> bool:
        """Determine whether the client requested reasoning summaries.

        Checks the ``include`` and ``reasoning`` fields in the Responses API
        payload to decide whether reasoning content should be included.

        Args:
            payload: The raw request payload dictionary.

        Returns:
            ``True`` if reasoning summaries should be included in the response.
        """
        include = payload.get("include")
        if isinstance(include, list):
            for entry in include:
                if isinstance(entry, str) and entry.strip().lower().startswith("reasoning"):
                    return True

        reasoning = payload.get("reasoning")
        if isinstance(reasoning, bool):
            return reasoning
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, bool):
                return summary
            if isinstance(summary, str):
                return summary.strip().lower() not in {"", "none", "off", "disabled", "false", "0", "null"}
            return summary is not None
        return True

    @staticmethod
    def _responses_payload_to_messages(
        payload: dict[str, Any],
        *,
        include_instructions: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert a Responses API payload into a list of chat messages.

        Handles multiple input formats: ``messages`` list, ``input`` as a
        string, ``input`` as a message list, and ``input`` as structured
        items (``message``, ``function_call_output``, ``input_text``, etc.).

        Args:
            payload: The raw Responses API request payload.
            include_instructions: Whether to prepend ``instructions`` as a
                system message.

        Returns:
            A list of chat message dictionaries suitable for passing to
            the engine's ``chat()`` method.
        """
        messages: list[dict[str, Any]] = []

        if include_instructions:
            instructions = payload.get("instructions")
            if isinstance(instructions, str) and instructions.strip():
                messages.append({"role": "system", "content": instructions.strip()})

        if isinstance(payload.get("messages"), list):
            for message in payload["messages"]:
                if isinstance(message, dict):
                    messages.append(dict(message))
            return messages

        input_value = payload.get("input")
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        if isinstance(input_value, list):
            if all(isinstance(item, dict) and "role" in item for item in input_value):
                return [dict(item) for item in input_value]
            converted: list[dict[str, Any]] = []
            for item in input_value:
                if not isinstance(item, dict):
                    converted.append({"role": "user", "content": item})
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "message":
                    role = item.get("role")
                    converted.append(
                        {
                            "role": role if isinstance(role, str) and role.strip() else "user",
                            "content": item.get("content", ""),
                        }
                    )
                    continue
                if item_type == "function_call_output":
                    call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id")
                    output_value = item.get("output", item.get("content", ""))
                    if isinstance(output_value, (dict, list)):
                        output_text = json.dumps(output_value, ensure_ascii=False)
                    else:
                        output_text = "" if output_value is None else str(output_value)
                    tool_message: dict[str, Any] = {"role": "tool", "content": output_text}
                    if call_id is not None:
                        tool_message["tool_call_id"] = str(call_id)
                    converted.append(tool_message)
                    continue
                if item_type in {"input_text", "output_text", "text"}:
                    converted.append({"role": "user", "content": str(item.get("text", item.get("content", "")))})
                    continue
                converted.append({"role": "user", "content": item})
            return converted
        if input_value is not None:
            return [{"role": "user", "content": str(input_value)}]
        return messages

    def _build_responses_output_items(
        self,
        output: RequestOutput,
        *,
        include_reasoning_summary: bool,
    ) -> list[ResponseOutputText]:
        """Build Responses API output items from a completed request output.

        Constructs a list of ``ResponseOutputText`` items that may include
        reasoning summaries, function call items, and text output items.

        Args:
            output: The completed request output from the engine.
            include_reasoning_summary: Whether to include reasoning content
                as a summary item.

        Returns:
            A list of ``ResponseOutputText`` items for the Responses API body.
        """
        completion = output.outputs[0]
        response_text = output.accumulated_text or output.get_text()
        reasoning_text = output.reasoning_content or completion.reasoning_content
        tool_calls = output.tool_calls or completion.tool_calls or []

        items: list[ResponseOutputText] = []
        if include_reasoning_summary and reasoning_text:
            items.append(
                ResponseOutputText(
                    type="reasoning",
                    summary=[{"type": "summary_text", "text": reasoning_text}],
                )
            )
        for index, call in enumerate(tool_calls):
            function_payload = dict(call.get("function") or {})
            call_id = str(call.get("id") or f"call_{index}")
            items.append(
                ResponseOutputText(
                    type="function_call",
                    id=call_id,
                    call_id=call_id,
                    name=str(function_payload.get("name") or ""),
                    arguments=str(function_payload.get("arguments") or "{}"),
                )
            )
        if response_text or not items:
            items.append(ResponseOutputText(text=response_text))
        return items

    @staticmethod
    def _coerce_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[ToolCall] | None:
        """Convert raw tool call dictionaries to ``ToolCall`` model instances.

        Args:
            tool_calls: List of raw tool call dicts from the engine output,
                or ``None``.

        Returns:
            A list of ``ToolCall`` model instances, or ``None`` if the input
            is empty or ``None``.
        """
        if not tool_calls:
            return None
        normalized: list[ToolCall] = []
        for call in tool_calls:
            function_payload = dict(call.get("function") or {})
            normalized.append(
                ToolCall(
                    id=str(call.get("id") or f"call_{len(normalized)}"),
                    type=str(call.get("type") or "function"),
                    function=FunctionCall(
                        name=str(function_payload.get("name") or ""),
                        arguments=str(function_payload.get("arguments") or "{}"),
                    ),
                )
            )
        return normalized or None

    def _completion_response(self, model: str, output: RequestOutput) -> CompletionResponse:
        """Build a ``CompletionResponse`` from an engine output.

        Args:
            model: The model name used for generation.
            output: The completed request output from the engine.

        Returns:
            A ``CompletionResponse`` model instance ready for serialization.
        """
        completion = output.outputs[0]
        return CompletionResponse(
            id=output.request_id,
            created=int(time.time()),
            model=model,
            choices=[Choice(index=completion.index, text=completion.text, finish_reason=completion.finish_reason)],
            usage=self._build_usage(output),
        )

    def _chat_response(self, model: str, output: RequestOutput) -> ChatCompletionResponse:
        """Build a ``ChatCompletionResponse`` from an engine output.

        Args:
            model: The model name used for generation.
            output: The completed request output from the engine.

        Returns:
            A ``ChatCompletionResponse`` model instance ready for serialization.
        """
        completion = output.outputs[0]
        return ChatCompletionResponse(
            id=output.request_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=completion.index,
                    message=ChatMessage(
                        role="assistant",
                        content=completion.text,
                        reasoning_content=completion.reasoning_content,
                        tool_calls=self._coerce_tool_calls(completion.tool_calls),
                    ),
                    finish_reason=completion.finish_reason,
                )
            ],
            usage=self._build_usage(output),
        )

    def _health(self) -> Any:
        """Handle GET ``/health`` -- return server health status."""
        payload = HealthResponse(models=sorted(self.engines)).model_dump()
        payload["timestamp"] = time.time()
        payload["uptime_seconds"] = round(max(time.time() - self.metrics.start_time, 0.0), 6)
        payload["active_requests"] = self.metrics.snapshot(
            models_loaded=len(self.engines),
            status=self.status,
            auth_stats=self.admin_state.stats(),
        )["active_requests"]
        return JSONResponse(payload)

    def _metrics(self, raw_request: Request) -> Any:
        """Handle GET ``/metrics`` -- return server performance metrics.

        Requires admin authentication if ``admin_api_key`` is configured.

        Args:
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response containing the metrics snapshot.
        """
        if self.admin_api_key is not None:
            self._require_admin(raw_request)
        payload = MetricsResponse(
            **self.metrics.snapshot(
                models_loaded=len(self.engines),
                status=self.status,
                auth_stats=self.admin_state.stats(),
            )
        )
        return JSONResponse(payload.model_dump())

    def _list_models(self, raw_request: Request) -> Any:
        """Handle GET ``/v1/models`` -- list all loaded models.

        Args:
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response with model info and metadata.
        """
        self._authorize_request(
            raw_request,
            endpoint="/v1/models",
            model="",
            requested_tokens=0,
        )
        payload = {
            "object": "list",
            "data": [
                {
                    **ModelInfo(id=name).model_dump(),
                    "metadata": {
                        "max_model_len": getattr(engine, "max_model_len", None),
                        "max_num_seqs": getattr(engine, "max_num_seqs", None),
                        "tool_parser": getattr(engine, "tool_parser_name", None),
                        "reasoning_parser": getattr(engine, "reasoning_parser_name", None),
                    },
                }
                for name, engine in sorted(self.engines.items())
            ],
            "total": len(self.engines),
        }
        return JSONResponse(payload)

    def _get_model(self, model_id: str, raw_request: Request) -> Any:
        """Handle GET ``/v1/models/{model_id}`` -- retrieve a single model's info.

        Args:
            model_id: The model identifier to look up.
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response with model info and metadata.

        Raises:
            HTTPException: 404 if the model is not found.
        """
        self._authorize_request(
            raw_request,
            endpoint="/v1/models/{model_id}",
            model=model_id,
            requested_tokens=0,
        )
        engine = self._get_engine(model_id)
        payload = ModelInfo(id=model_id).model_dump()
        payload["metadata"] = {
            "max_model_len": getattr(engine, "max_model_len", None),
            "max_num_seqs": getattr(engine, "max_num_seqs", None),
            "tool_parser": getattr(engine, "tool_parser_name", None),
            "reasoning_parser": getattr(engine, "reasoning_parser_name", None),
        }
        return JSONResponse(payload)

    def _complete(self, payload: CompletionRequest, raw_request: Request) -> Any:
        """Handle POST ``/v1/completions`` -- generate text completions.

        Supports both streaming and non-streaming responses.

        Args:
            payload: The completion request body.
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response or streaming response with completion results.

        Raises:
            HTTPException: 400 if the prompt is missing.
            HTTPException: 404 if the model is not found.
        """
        started_at = self.metrics.start_request(endpoint="/v1/completions", model=payload.model)
        try:
            engine = self._get_engine(payload.model)
            prompts = payload.prompts()
            if not prompts:
                raise HTTPException(status_code=400, detail="`prompt` is required")
            self._authorize_request(
                raw_request,
                endpoint="/v1/completions",
                model=payload.model,
                requested_tokens=payload.max_tokens or 16,
                uses_streaming=payload.stream,
            )

            sampling_params = self._build_sampling_params(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                presence_penalty=payload.presence_penalty,
                repetition_penalty=payload.repetition_penalty,
                stop=payload.stop,
            )
            if payload.stream:
                return self._stream_completions(
                    engine,
                    payload.model,
                    prompts,
                    sampling_params,
                    started_at,
                    raw_request=raw_request,
                )

            output = engine.generate(prompts, sampling_params)[0]
            self._finish_metrics(started_at, success=True, output=output, raw_request=raw_request)
            return JSONResponse(self._completion_response(payload.model, output).model_dump())
        except HTTPException:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise
        except Exception:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise

    def _chat_complete(self, payload: ChatCompletionRequest, raw_request: Request) -> Any:
        """Handle POST ``/v1/chat/completions`` -- generate chat completions.

        Supports streaming, tool calling, and reasoning extraction.

        Args:
            payload: The chat completion request body.
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response or streaming response with chat completion results.

        Raises:
            HTTPException: 404 if the model is not found.
        """
        started_at = self.metrics.start_request(endpoint="/v1/chat/completions", model=payload.model)
        try:
            engine = self._get_engine(payload.model)
            request_tools = self._merge_tools(payload.tools, payload.functions)
            self._authorize_request(
                raw_request,
                endpoint="/v1/chat/completions",
                model=payload.model,
                requested_tokens=payload.max_tokens or 16,
                uses_streaming=payload.stream,
                uses_function_calling=bool(request_tools),
            )
            sampling_params = self._build_sampling_params(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                presence_penalty=payload.presence_penalty,
                repetition_penalty=payload.repetition_penalty,
                stop=payload.stop,
                tools=payload.tools,
                functions=payload.functions,
                tool_choice=payload.tool_choice,
            )
            messages = [message.model_dump(exclude_none=True) for message in payload.messages]
            chat_template_kwargs = dict(payload.chat_template_kwargs or {})
            if payload.enable_thinking is not None and "enable_thinking" not in chat_template_kwargs:
                chat_template_kwargs["enable_thinking"] = payload.enable_thinking
            if payload.stream:
                return self._stream_chat(
                    engine,
                    payload.model,
                    messages,
                    sampling_params,
                    request_tools,
                    chat_template_kwargs,
                    started_at,
                    raw_request=raw_request,
                    include_usage=bool(payload.stream_options and payload.stream_options.include_usage),
                )

            output = engine.chat(
                messages,
                sampling_params=sampling_params,
                tools=request_tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            self._finish_metrics(started_at, success=True, output=output, raw_request=raw_request)
            return JSONResponse(self._chat_response(payload.model, output).model_dump())
        except HTTPException:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise
        except Exception:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise

    def _responses(self, payload: ResponsesRequest, raw_request: Request) -> Any:
        """Handle POST ``/v1/responses`` -- OpenAI Responses API endpoint.

        Supports system instructions, streaming, tool calling, and reasoning
        content inclusion. Conversation persistence is intentionally not
        owned by this server.

        Args:
            payload: The Responses API request body.
            raw_request: The incoming FastAPI request.

        Returns:
            JSON response or streaming response with the Responses API output.

        Raises:
            HTTPException: 400 if input is missing or conflicting options
                are provided.
            HTTPException: 404 if the model is not found.
        """
        started_at = self.metrics.start_request(endpoint="/v1/responses", model=payload.model)
        try:
            engine = self._get_engine(payload.model)
            request_tools = self._merge_tools(payload.tools, payload.functions)
            self._authorize_request(
                raw_request,
                endpoint="/v1/responses",
                model=payload.model,
                requested_tokens=payload.max_output_tokens or 16,
                uses_streaming=payload.stream,
                uses_function_calling=bool(request_tools),
            )
            payload_dict = payload.model_dump(exclude_none=True, exclude_unset=True)
            store_response = False
            previous_response_id = (
                payload.previous_response_id.strip() if isinstance(payload.previous_response_id, str) else None
            )
            previous_response_id = previous_response_id or None
            conversation_id = self._normalize_conversation_id(payload.conversation)
            if previous_response_id and conversation_id:
                raise HTTPException(status_code=400, detail="Cannot use both 'previous_response_id' and 'conversation'")
            if previous_response_id is not None:
                raise HTTPException(status_code=400, detail="previous_response_id persistence is not supported")
            if conversation_id is not None:
                raise HTTPException(status_code=400, detail="conversation persistence is not supported")

            input_messages = self._responses_payload_to_messages(payload_dict, include_instructions=False)
            if not input_messages:
                raise HTTPException(status_code=400, detail="`input` or `messages` is required")

            full_messages = list(input_messages)
            engine_messages = list(full_messages)
            instructions = payload.instructions.strip() if isinstance(payload.instructions, str) else None
            if instructions:
                engine_messages.insert(0, {"role": "system", "content": instructions})
            sampling_params = self._build_sampling_params(
                max_tokens=payload.max_output_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                presence_penalty=payload.presence_penalty,
                repetition_penalty=payload.repetition_penalty,
                tools=payload.tools,
                functions=payload.functions,
                tool_choice=payload.tool_choice,
            )
            include_reasoning_summary = self._responses_reasoning_summary_requested(payload_dict)
            response_id = f"resp_{uuid.uuid4().hex}"

            if payload.stream:
                return self._stream_responses(
                    engine,
                    payload.model,
                    engine_messages,
                    sampling_params,
                    request_tools,
                    started_at,
                    raw_request=raw_request,
                    response_id=response_id,
                    include_reasoning_summary=include_reasoning_summary,
                    previous_response_id=previous_response_id,
                    instructions=instructions,
                    payload_dict=payload_dict,
                )

            output = engine.chat(
                engine_messages,
                sampling_params=sampling_params,
                tools=request_tools,
                request_id=response_id,
            )
            self._finish_metrics(started_at, success=True, output=output, raw_request=raw_request)
            output_items = self._build_responses_output_items(
                output,
                include_reasoning_summary=include_reasoning_summary,
            )
            response = ResponsesResponse(
                id=response_id,
                created=int(time.time()),
                model=payload.model,
                output=[
                    ResponsesOutput(
                        id=response_id,
                        content=output_items,
                    )
                ],
                usage=self._build_usage(output),
                instructions=instructions,
                previous_response_id=previous_response_id,
                store=store_response,
                metadata=payload.metadata or {},
                parallel_tool_calls=payload.parallel_tool_calls,
                tools=request_tools or [],
                tool_choice=payload.tool_choice,
                temperature=payload.temperature,
                top_p=payload.top_p,
                max_output_tokens=payload.max_output_tokens,
                text={"format": {"type": "text"}},
            )
            return JSONResponse(response.model_dump(exclude_none=True))
        except HTTPException:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise
        except Exception:
            self._finish_metrics(started_at, success=False, raw_request=raw_request)
            raise

    def _stream_completions(
        self,
        engine: eSurge,
        model: str,
        prompts: list["str"],
        sampling_params: SamplingParams,
        started_at: float,
        *,
        raw_request: Request | None = None,
    ) -> StreamingResponse:
        """Create a streaming response for text completions.

        Args:
            engine: The ``eSurge`` engine to generate with.
            model: The model name for response metadata.
            prompts: List of input prompts.
            sampling_params: Sampling parameters for generation.
            started_at: Timestamp from ``metrics.start_request()``.
            raw_request: The raw FastAPI request for API key tracking.

        Returns:
            A ``StreamingResponse`` emitting Server-Sent Events.
        """

        async def event_generator() -> AsyncGenerator["str"]:
            last_output: RequestOutput | None = None
            try:
                for output in engine.stream(prompts, sampling_params):
                    last_output = output
                    self._record_stream_progress(started_at, output)
                    completion = output.outputs[0]
                    payload = {
                        "id": output.request_id,
                        "object": "text_completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            Choice(
                                index=completion.index,
                                text=output.delta_text,
                                finish_reason=completion.finish_reason if output.finished else None,
                            ).model_dump()
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0)
                self._finish_metrics(started_at, success=True, output=last_output, raw_request=raw_request)
                yield "data: [DONE]\n\n"
            except Exception:
                self._finish_metrics(started_at, success=False, output=last_output, raw_request=raw_request)
                raise

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    def _stream_chat(
        self,
        engine: eSurge,
        model: str,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        started_at: float,
        *,
        raw_request: Request | None = None,
        include_usage: bool = False,
    ) -> StreamingResponse:
        """Create a streaming response for chat completions.

        Applies the chat template to messages, then streams generation output
        as SSE chunks with delta messages.

        Args:
            engine: The ``eSurge`` engine to generate with.
            model: The model name for response metadata.
            messages: Chat message history.
            sampling_params: Sampling parameters for generation.
            tools: Tool definitions for chat template rendering.
            started_at: Timestamp from ``metrics.start_request()``.
            raw_request: The raw FastAPI request for API key tracking.
            include_usage: Whether to include final usage in the stream.

        Returns:
            A ``StreamingResponse`` emitting Server-Sent Events.
        """

        async def event_generator() -> AsyncGenerator["str"]:
            last_output: RequestOutput | None = None
            try:
                for output in engine.chat(
                    messages,
                    sampling_params=sampling_params,
                    tools=tools,
                    chat_template_kwargs=chat_template_kwargs,
                    stream=True,
                ):
                    last_output = output
                    self._record_stream_progress(started_at, output)
                    payload = ChatCompletionStreamResponse(
                        id=output.request_id,
                        created=int(time.time()),
                        model=model,
                        choices=[
                            ChatStreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role="assistant" if output.update_seq <= 1 else None,
                                    content=output.delta_text or None,
                                    reasoning_content=output.delta_reasoning_content,
                                    delta_reasoning_content=output.delta_reasoning_content,
                                    tool_calls=self._coerce_tool_calls(output.delta_tool_calls),
                                ),
                                finish_reason=output.outputs[0].finish_reason if output.finished else None,
                            )
                        ],
                        usage=self._build_usage(output) if include_usage and output.finished else None,
                    )
                    yield f"data: {json.dumps(payload.model_dump(exclude_none=True))}\n\n"
                    await asyncio.sleep(0)
                self._finish_metrics(started_at, success=True, output=last_output, raw_request=raw_request)
                yield "data: [DONE]\n\n"
            except Exception:
                self._finish_metrics(started_at, success=False, output=last_output, raw_request=raw_request)
                raise

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    def _stream_responses(
        self,
        engine: eSurge,
        model: str,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams,
        tools: list[dict[str, Any]] | None,
        started_at: float,
        *,
        raw_request: Request | None,
        response_id: str,
        include_reasoning_summary: bool,
        previous_response_id: str | None,
        instructions: str | None,
        payload_dict: dict[str, Any],
    ) -> StreamingResponse:
        """Create a streaming response for the Responses API.

        Emits ``response.created``, ``response.output_text.delta``,
        ``response.reasoning.delta``, and ``response.completed`` events.

        Args:
            engine: The ``eSurge`` engine to generate with.
            model: The model name for response metadata.
            messages: Chat message history (including system instructions).
            sampling_params: Sampling parameters for generation.
            tools: Tool definitions for chat template rendering.
            started_at: Timestamp from ``metrics.start_request()``.
            raw_request: The raw FastAPI request for API key tracking.
            response_id: Unique identifier for this response.
            include_reasoning_summary: Whether to emit reasoning deltas.
            previous_response_id: ID of the previous response in the chain.
            instructions: System instructions text.
            payload_dict: Raw request payload for metadata extraction.

        Returns:
            A ``StreamingResponse`` emitting Server-Sent Events.
        """

        async def event_generator() -> AsyncGenerator["str"]:
            last_output: RequestOutput | None = None
            yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'model': model}})}\n\n"
            try:
                for output in engine.chat(
                    messages,
                    sampling_params=sampling_params,
                    tools=tools,
                    stream=True,
                    request_id=response_id,
                ):
                    last_output = output
                    self._record_stream_progress(started_at, output)
                    if output.delta_text:
                        yield f"data: {json.dumps({'type': 'response.output_text.delta', 'response_id': response_id, 'delta': output.delta_text})}\n\n"
                    if include_reasoning_summary and output.delta_reasoning_content:
                        yield f"data: {json.dumps({'type': 'response.reasoning.delta', 'response_id': response_id, 'delta': output.delta_reasoning_content})}\n\n"
                    await asyncio.sleep(0)
                self._finish_metrics(started_at, success=True, output=last_output, raw_request=raw_request)

                if last_output is None:
                    yield "data: [DONE]\n\n"
                    return

                output_items = self._build_responses_output_items(
                    last_output,
                    include_reasoning_summary=include_reasoning_summary,
                )
                response = ResponsesResponse(
                    id=response_id,
                    created=int(time.time()),
                    model=model,
                    output=[ResponsesOutput(id=response_id, content=output_items)],
                    usage=self._build_usage(last_output),
                    instructions=instructions,
                    previous_response_id=previous_response_id,
                    store=False,
                    metadata=payload_dict.get("metadata") or {},
                    parallel_tool_calls=payload_dict.get("parallel_tool_calls"),
                    tools=tools or [],
                    tool_choice=payload_dict.get("tool_choice"),
                    temperature=payload_dict.get("temperature"),
                    top_p=payload_dict.get("top_p"),
                    max_output_tokens=payload_dict.get("max_output_tokens"),
                    text={"format": {"type": "text"}},
                )
                yield f"data: {json.dumps({'type': 'response.completed', 'response': response.model_dump(exclude_none=True)})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                self._finish_metrics(started_at, success=False, output=last_output, raw_request=raw_request)
                raise

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    def _create_admin_key(self, payload: AdminKeyCreateRequest, raw_request: Request) -> Any:
        """Handle POST ``/v1/admin/keys`` -- create a new admin API key.

        Args:
            payload: The key creation request body.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON response with the created key details (HTTP 201).
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        _, response = self.admin_state.create_key(actor=actor, **payload.model_dump())
        self.metrics.record_admin_action()
        return JSONResponse(AdminKeyResponse(**response).model_dump(exclude_none=True), status_code=201)

    def _list_admin_keys(self, raw_request: Request, role: str | None = None, status: str | None = None) -> Any:
        """Handle GET ``/v1/admin/keys`` -- list all admin API keys.

        Args:
            raw_request: The incoming FastAPI request (must include admin auth).
            role: Optional filter by key role.
            status: Optional filter by key status.

        Returns:
            JSON response with a list of key records and total count.
        """
        self._require_admin(raw_request)
        payload = {
            "keys": self.admin_state.list_keys(role=role, status=status),
            "total": len(self.admin_state.list_keys(role=role, status=status)),
        }
        return JSONResponse(payload)

    def _get_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle GET ``/v1/admin/keys/{key_id}`` -- retrieve a single key.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON response with the key record.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        self._require_admin(raw_request)
        record = self.admin_state.get_key(key_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")
        return JSONResponse(record)

    def _update_admin_key(self, key_id: str, payload: AdminKeyUpdateRequest, raw_request: Request) -> Any:
        """Handle PATCH ``/v1/admin/keys/{key_id}`` -- update an existing key.

        Args:
            key_id: The API key identifier.
            payload: The key update request body (partial updates supported).
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON response with the updated key record.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            record = self.admin_state.update_key(
                key_id,
                actor=actor,
                **payload.model_dump(exclude_none=True),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse(record)

    def _revoke_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle DELETE ``/v1/admin/keys/{key_id}/revoke`` -- revoke a key.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON confirmation message.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            self.admin_state.revoke_key(key_id, actor=actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse({"message": "API key revoked successfully", "key_id": key_id})

    def _suspend_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle POST ``/v1/admin/keys/{key_id}/suspend`` -- suspend a key.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON confirmation message.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            self.admin_state.suspend_key(key_id, actor=actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse({"message": "API key suspended successfully", "key_id": key_id})

    def _reactivate_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle POST ``/v1/admin/keys/{key_id}/reactivate`` -- reactivate a suspended key.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON confirmation message.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            self.admin_state.reactivate_key(key_id, actor=actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse({"message": "API key reactivated successfully", "key_id": key_id})

    def _delete_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle DELETE ``/v1/admin/keys/{key_id}`` -- permanently delete a key.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON confirmation message.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            self.admin_state.delete_key(key_id, actor=actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse({"message": "API key deleted successfully", "key_id": key_id})

    def _rotate_admin_key(self, key_id: str, raw_request: Request) -> Any:
        """Handle POST ``/v1/admin/keys/{key_id}/rotate`` -- rotate a key's secret.

        Generates a new API key secret while preserving the key's metadata
        and configuration.

        Args:
            key_id: The API key identifier.
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON response with the new key details.

        Raises:
            HTTPException: 404 if the key is not found.
        """
        actor_token = self._require_admin(raw_request)
        actor = self._admin_actor(actor_token)
        try:
            _, response = self.admin_state.rotate_key(key_id, actor=actor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"API key not found: {key_id}") from exc
        self.metrics.record_admin_action()
        return JSONResponse(AdminKeyResponse(**response).model_dump(exclude_none=True))

    def _admin_key_stats(self, raw_request: Request) -> Any:
        """Handle GET ``/v1/admin/keys/stats`` -- get API key statistics.

        Args:
            raw_request: The incoming FastAPI request (must include admin auth).

        Returns:
            JSON response with key count, role distribution, status
            distribution, and audit entry count.
        """
        self._require_admin(raw_request)
        payload = AdminKeyStatsResponse(**self.admin_state.stats())
        return JSONResponse(payload.model_dump())

    def _admin_audit_logs(
        self,
        raw_request: Request,
        limit: int = 100,
        key_id: str | None = None,
        action: str | None = None,
    ) -> Any:
        """Handle GET ``/v1/admin/audit-logs`` -- retrieve admin audit logs.

        Args:
            raw_request: The incoming FastAPI request (must include admin auth).
            limit: Maximum number of log entries to return (default 100).
            key_id: Optional filter by API key identifier.
            action: Optional filter by action type.

        Returns:
            JSON response with audit log entries and total count.
        """
        self._require_admin(raw_request)
        logs = [
            AdminAuditEntry(**item)
            for item in self.admin_state.get_audit_logs(limit=limit, key_id=key_id, action=action)
        ]
        payload = AdminAuditLogResponse(logs=logs, total=len(logs))
        return JSONResponse(payload.model_dump())
