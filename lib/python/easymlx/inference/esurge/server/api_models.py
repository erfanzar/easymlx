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

"""OpenAI-style request and response models for easymlx eSurge.

This module defines Pydantic models that mirror the OpenAI API specification
for completions, chat completions, and the newer Responses API. It also
includes admin key management models and tool execution models used by the
eSurge API server.

All models use Pydantic ``BaseModel`` for automatic validation and
serialization to/from JSON.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Represents a function call within a tool call.

    Attributes:
        name: The name of the function to call.
        arguments: JSON-encoded string of arguments to pass to the function.
    """

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a single tool call emitted by the model.

    Attributes:
        id: Unique identifier for this tool call (auto-generated if omitted).
        type: The type of tool call, currently always ``"function"``.
        function: The function call details including name and arguments.
    """

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: FunctionCall


class Usage(BaseModel):
    """Token usage statistics for a completion request.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StreamOptions(BaseModel):
    """Options controlling streamed response metadata."""

    include_usage: bool = False


class Choice(BaseModel):
    """A single completion choice from a text completion response.

    Attributes:
        index: Zero-based index of this choice in the response.
        text: The generated text content.
        finish_reason: Why generation stopped (e.g. ``"stop"``, ``"length"``),
            or ``None`` if still generating during streaming.
    """

    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """Response body for the ``/v1/completions`` endpoint.

    Attributes:
        id: Unique request identifier.
        object: Object type, always ``"text_completion"``.
        created: Unix timestamp of when the response was created.
        model: The model used for generation.
        choices: List of generated completion choices.
        usage: Token usage statistics for this request.
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message author (``"system"``, ``"user"``,
            ``"assistant"``, or ``"tool"``).
        content: The text content of the message, or ``None`` for tool-call-only
            assistant messages.
        reasoning_content: Optional internal reasoning/thinking content
            extracted by a reasoning parser.
        tool_calls: Optional list of tool calls the assistant wants to make.
    """

    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class DeltaFunctionCall(BaseModel):
    """Incremental function call update in a streaming response.

    Attributes:
        name: The function name (sent in the first chunk, ``None`` thereafter).
        arguments: Incremental JSON argument string fragment.
    """

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    """Incremental tool call update in a streaming chat response.

    Attributes:
        index: Zero-based index identifying which tool call this delta updates.
        id: Tool call identifier (sent in the first chunk for this index).
        type: Tool call type (sent in the first chunk, typically ``"function"``).
        function: Incremental function call payload or dictionary.
    """

    index: int
    id: str | None = None
    type: str | None = None
    function: dict[str, Any] | DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    """Result of extracting tool calls from model output text.

    Used internally by tool parsers to communicate extraction results
    back to the API server layer.

    Attributes:
        tools_called: Whether any tool calls were detected in the output.
        tool_calls: List of extracted tool call objects.
        content: Remaining text content after tool calls are extracted,
            or ``None`` if the entire output consisted of tool calls.
    """

    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


class DeltaMessage(BaseModel):
    """Incremental message update in a streaming chat response.

    Each streaming chunk contains a ``DeltaMessage`` with only the
    fields that changed since the previous chunk.

    Attributes:
        role: Message role (only sent in the first chunk, typically ``"assistant"``).
        content: Incremental text content fragment.
        reasoning_content: Incremental reasoning/thinking content fragment.
        delta_reasoning_content: Alias for the same incremental reasoning
            fragment, matching the internal engine field name.
        tool_calls: Incremental tool call updates.
    """

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    delta_reasoning_content: str | None = None
    tool_calls: list[ToolCall | DeltaToolCall] | None = None


class ChatChoice(BaseModel):
    """A single choice in a non-streaming chat completion response.

    Attributes:
        index: Zero-based index of this choice.
        message: The complete assistant message for this choice.
        finish_reason: Why generation stopped, or ``None`` if incomplete.
    """

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatStreamChoice(BaseModel):
    """A single choice in a streaming chat completion chunk.

    Attributes:
        index: Zero-based index of this choice.
        delta: Incremental message update for this chunk.
        finish_reason: Why generation stopped (set only in the final chunk).
    """

    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Response body for the ``/v1/chat/completions`` endpoint (non-streaming).

    Attributes:
        id: Unique request identifier.
        object: Object type, always ``"chat.completion"``.
        created: Unix timestamp of when the response was created.
        model: The model used for generation.
        choices: List of chat completion choices.
        usage: Token usage statistics for this request.
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ChatCompletionStreamResponse(BaseModel):
    """A single Server-Sent Event chunk for streaming chat completions.

    Attributes:
        id: Unique request identifier (same across all chunks in a stream).
        object: Object type, always ``"chat.completion.chunk"``.
        created: Unix timestamp of when the chunk was created.
        model: The model used for generation.
        choices: List of streaming choice deltas for this chunk.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamChoice]
    usage: Usage | None = None


class ResponseOutputText(BaseModel):
    """An individual output item in the Responses API format.

    This model is polymorphic: the ``type`` field determines which optional
    fields are populated (e.g. ``"output_text"`` uses ``text``, while
    ``"function_call"`` uses ``name``, ``arguments``, and ``call_id``).

    Attributes:
        type: Item type (``"output_text"``, ``"function_call"``, ``"reasoning"``).
        text: Generated text content (for ``"output_text"`` type).
        id: Item identifier (for ``"function_call"`` type).
        name: Function name (for ``"function_call"`` type).
        arguments: JSON-encoded function arguments (for ``"function_call"`` type).
        call_id: Tool call identifier (for ``"function_call"`` type).
        summary: Reasoning summary entries (for ``"reasoning"`` type).
    """

    type: str = "output_text"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None
    summary: list[dict[str, Any]] | None = None


class ResponsesOutput(BaseModel):
    """A single output message in the Responses API format.

    Attributes:
        id: Unique identifier for this output message.
        type: Output type, typically ``"message"``.
        role: The role of the output author, typically ``"assistant"``.
        content: List of output content items (text, function calls, reasoning).
    """

    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[ResponseOutputText]


class ResponsesResponse(BaseModel):
    """Response body for the ``/v1/responses`` endpoint.

    This follows the OpenAI Responses API format for stateless generation.
    Response persistence fields are schema-compatible echoes only.

    Attributes:
        id: Unique response identifier (prefixed with ``"resp_"``).
        object: Object type, always ``"response"``.
        created: Unix timestamp of when the response was created.
        model: The model used for generation.
        output: List of output messages.
        usage: Token usage statistics.
        instructions: System instructions that were applied.
        previous_response_id: Always ``None`` for this server.
        store: Always ``False`` for this server.
        metadata: Arbitrary key-value metadata from the request.
        parallel_tool_calls: Whether parallel tool calls were allowed.
        tools: Tool definitions that were available during generation.
        tool_choice: Tool selection strategy that was applied.
        temperature: Sampling temperature used.
        top_p: Nucleus sampling probability used.
        max_output_tokens: Maximum output token limit.
        text: Text format configuration.
    """

    id: str
    object: str = "response"
    created: int
    model: str
    output: list[ResponsesOutput]
    usage: Usage
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parallel_tool_calls: bool | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    text: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    """Request body for the ``/v1/completions`` endpoint.

    Attributes:
        model: Model identifier to use for generation.
        prompt: Input text prompt(s). Can be a single string or list of strings.
        max_tokens: Maximum number of tokens to generate (minimum 1).
        temperature: Sampling temperature (0.0 = greedy, higher = more random).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter (0 = disabled).
        presence_penalty: Additive penalty applied to generated tokens
            that already appeared in the output.
        repetition_penalty: Multiplicative penalty applied to tokens
            that already appeared in the prompt/output history.
        n: Number of completions to generate per prompt.
        stop: Stop sequence(s) that terminate generation.
        stream: Whether to stream the response as Server-Sent Events.
    """

    model: str
    prompt: str | list["str"] | None = None
    max_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    n: int = Field(1, ge=1)
    stop: list["str"] | str | None = None
    stream: bool = False

    def prompts(self) -> list["str"]:
        """Normalize the prompt field into a list of strings.

        Returns:
            A list of prompt strings. Returns an empty list if ``prompt``
            is ``None``, wraps a single string in a list, or returns the
            list as-is.
        """
        if self.prompt is None:
            return []
        if isinstance(self.prompt, str):
            return [self.prompt]
        return list(self.prompt)


class ChatCompletionRequest(BaseModel):
    """Request body for the ``/v1/chat/completions`` endpoint.

    Attributes:
        model: Model identifier to use for generation.
        messages: List of messages comprising the conversation so far.
        max_tokens: Maximum number of tokens to generate (minimum 1).
        temperature: Sampling temperature (0.0 = greedy, higher = more random).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter (0 = disabled).
        presence_penalty: Additive penalty applied to generated tokens
            that already appeared in the output.
        repetition_penalty: Multiplicative penalty applied to tokens
            that already appeared in the prompt/output history.
        n: Number of chat completion choices to generate.
        stop: Stop sequence(s) that terminate generation.
        stream: Whether to stream the response as Server-Sent Events.
        tools: Tool definitions available to the model (OpenAI format).
        functions: Legacy function definitions (converted to tools internally).
        tool_choice: Strategy for tool selection (``"auto"``, ``"none"``,
            or a specific tool specification).
        parallel_tool_calls: Whether the model may emit multiple tool calls
            in a single response.
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    n: int = Field(1, ge=1)
    stop: list["str"] | str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[dict[str, Any]] | None = None
    functions: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    enable_thinking: bool | None = None


class ResponsesRequest(BaseModel):
    """Request body for the ``/v1/responses`` endpoint.

    Supports stateless OpenAI Responses API requests with system
    instructions, tool metadata, and reasoning content inclusion.

    Attributes:
        model: Model identifier to use for generation.
        input: Input content (string, message list, or structured items).
        messages: Alternative message-based input format.
        max_output_tokens: Maximum number of tokens to generate (minimum 1).
        temperature: Sampling temperature (0.0 = greedy, higher = more random).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter (0 = disabled).
        presence_penalty: Additive penalty applied to generated tokens
            that already appeared in the output.
        repetition_penalty: Multiplicative penalty applied to tokens
            that already appeared in the prompt/output history.
        stream: Whether to stream the response as Server-Sent Events.
        tools: Tool definitions available to the model.
        functions: Legacy function definitions (converted to tools internally).
        parallel_tool_calls: Whether the model may emit multiple tool calls.
        tool_choice: Strategy for tool selection.
        instructions: System instructions prepended to the conversation.
        store: Accepted for schema compatibility but ignored.
        previous_response_id: Rejected because persistence is not server-owned.
        conversation: Rejected because persistence is not server-owned.
        include: Additional data to include (e.g. ``"reasoning"``).
        metadata: Arbitrary key-value metadata to attach to the response.
        reasoning: Whether to include reasoning content in the response.
    """

    model: str
    input: Any = None
    messages: list[ChatMessage] | None = None
    max_output_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    functions: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool | None = None
    tool_choice: str | dict[str, Any] | None = None
    instructions: str | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    conversation: str | dict[str, Any] | None = None
    include: list["str"] | None = None
    metadata: dict[str, Any] | None = None
    reasoning: bool | dict[str, Any] | None = None


class ModelInfo(BaseModel):
    """Model metadata returned by the ``/v1/models`` endpoint.

    Attributes:
        id: The model identifier string.
        object: Object type, always ``"model"``.
        owned_by: Organization that owns the model.
    """

    id: str
    object: str = "model"
    owned_by: str = "easymlx"


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint.

    Attributes:
        status: Server health status string (``"ok"`` when healthy).
        models: List of loaded model identifiers.
    """

    status: str = "ok"
    models: list["str"]


class MetricsResponse(BaseModel):
    """Response body for the ``/metrics`` endpoint.

    Provides server-wide performance and usage statistics.

    Attributes:
        uptime_seconds: Time in seconds since the server started.
        total_requests: Total number of requests received.
        successful_requests: Number of requests that completed successfully.
        failed_requests: Number of requests that resulted in errors.
        total_tokens_generated: Cumulative tokens generated across all requests.
        average_tokens_per_second: Mean generation throughput.
        active_requests: Number of currently in-flight requests.
        models_loaded: Number of models currently loaded in memory.
        status: Server status string.
        auth_stats: Authentication and API key usage statistics.
        tool_executions: Total number of tool executions performed.
        failed_tool_executions: Number of tool executions that failed.
        admin_actions: Total number of admin API actions performed.
    """

    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    average_tokens_per_second: float
    active_requests: int
    models_loaded: int
    status: str
    auth_stats: dict[str, Any] = Field(default_factory=dict)
    tool_executions: int = 0
    failed_tool_executions: int = 0
    admin_actions: int = 0


class AdminKeyCreateRequest(BaseModel):
    """Request body for creating a new admin API key.

    Attributes:
        name: Human-readable name for the key.
        role: Role assigned to the key (e.g. ``"admin"``, ``"user"``).
        description: Optional description of the key's purpose.
        expires_in_days: Number of days until the key expires.
        requests_per_minute: Rate limit for requests per minute.
        requests_per_hour: Rate limit for requests per hour.
        requests_per_day: Rate limit for requests per day.
        tokens_per_minute: Rate limit for tokens per minute.
        tokens_per_hour: Rate limit for tokens per hour.
        tokens_per_day: Rate limit for tokens per day.
        max_total_tokens: Lifetime token usage cap.
        max_total_requests: Lifetime request count cap.
        monthly_token_limit: Monthly token usage cap.
        monthly_request_limit: Monthly request count cap.
        allowed_models: Whitelist of model IDs accessible with this key.
        allowed_endpoints: Whitelist of API endpoints accessible with this key.
        allowed_ip_addresses: Whitelist of client IP addresses.
        blocked_ip_addresses: Blacklist of client IP addresses.
        enable_streaming: Whether streaming responses are allowed.
        enable_function_calling: Whether tool/function calling is allowed.
        max_tokens_per_request: Maximum tokens per single request.
        tags: Arbitrary tags for categorizing the key.
        metadata: Arbitrary key-value metadata.
    """

    name: str
    role: str = "admin"
    description: str | None = None
    expires_in_days: int | None = Field(None, ge=1)
    requests_per_minute: int | None = Field(None, ge=1)
    requests_per_hour: int | None = Field(None, ge=1)
    requests_per_day: int | None = Field(None, ge=1)
    tokens_per_minute: int | None = Field(None, ge=1)
    tokens_per_hour: int | None = Field(None, ge=1)
    tokens_per_day: int | None = Field(None, ge=1)
    max_total_tokens: int | None = Field(None, ge=1)
    max_total_requests: int | None = Field(None, ge=1)
    monthly_token_limit: int | None = Field(None, ge=1)
    monthly_request_limit: int | None = Field(None, ge=1)
    allowed_models: list["str"] | None = None
    allowed_endpoints: list["str"] | None = None
    allowed_ip_addresses: list["str"] | None = None
    blocked_ip_addresses: list["str"] | None = None
    enable_streaming: bool = True
    enable_function_calling: bool = True
    max_tokens_per_request: int | None = Field(None, ge=1)
    tags: list["str"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminKeyUpdateRequest(BaseModel):
    """Request body for updating an existing admin API key.

    Only non-``None`` fields are applied as updates. All fields are optional
    to support partial updates.

    Attributes:
        name: Updated human-readable name.
        role: Updated role assignment.
        description: Updated description.
        expires_at: Updated expiration as a Unix timestamp.
        requests_per_minute: Updated requests-per-minute rate limit.
        requests_per_hour: Updated requests-per-hour rate limit.
        requests_per_day: Updated requests-per-day rate limit.
        tokens_per_minute: Updated tokens-per-minute rate limit.
        tokens_per_hour: Updated tokens-per-hour rate limit.
        tokens_per_day: Updated tokens-per-day rate limit.
        max_total_tokens: Updated lifetime token cap.
        max_total_requests: Updated lifetime request cap.
        monthly_token_limit: Updated monthly token cap.
        monthly_request_limit: Updated monthly request cap.
        allowed_models: Updated model whitelist.
        allowed_endpoints: Updated endpoint whitelist.
        allowed_ip_addresses: Updated IP whitelist.
        blocked_ip_addresses: Updated IP blacklist.
        enable_streaming: Updated streaming permission.
        enable_function_calling: Updated function calling permission.
        max_tokens_per_request: Updated per-request token limit.
        status: Updated key status (e.g. ``"active"``, ``"suspended"``).
        tags: Updated tag list.
        metadata: Updated metadata dictionary.
    """

    name: str | None = None
    role: str | None = None
    description: str | None = None
    expires_at: float | None = None
    requests_per_minute: int | None = Field(None, ge=1)
    requests_per_hour: int | None = Field(None, ge=1)
    requests_per_day: int | None = Field(None, ge=1)
    tokens_per_minute: int | None = Field(None, ge=1)
    tokens_per_hour: int | None = Field(None, ge=1)
    tokens_per_day: int | None = Field(None, ge=1)
    max_total_tokens: int | None = Field(None, ge=1)
    max_total_requests: int | None = Field(None, ge=1)
    monthly_token_limit: int | None = Field(None, ge=1)
    monthly_request_limit: int | None = Field(None, ge=1)
    allowed_models: list["str"] | None = None
    allowed_endpoints: list["str"] | None = None
    allowed_ip_addresses: list["str"] | None = None
    blocked_ip_addresses: list["str"] | None = None
    enable_streaming: bool | None = None
    enable_function_calling: bool | None = None
    max_tokens_per_request: int | None = Field(None, ge=1)
    status: str | None = None
    tags: list["str"] | None = None
    metadata: dict[str, Any] | None = None


class AdminKeyResponse(BaseModel):
    """Response body when creating, retrieving, or rotating an admin API key.

    Attributes:
        key: The full API key string (only returned on creation or rotation).
        key_id: Unique identifier for the key record.
        key_prefix: Short prefix of the key for display purposes.
        name: Human-readable name of the key.
        description: Optional description of the key's purpose.
        role: Role assigned to the key.
        status: Current status (``"active"``, ``"suspended"``, ``"revoked"``).
        created_at: Unix timestamp of when the key was created.
        expires_at: Unix timestamp of when the key expires, or ``None``.
        tags: Tags associated with the key.
        metadata: Arbitrary metadata associated with the key.
        message: Optional informational message (e.g. on creation).
    """

    key: str | None = None
    key_id: str
    key_prefix: str
    name: str
    description: str | None = None
    role: str
    status: str
    created_at: float
    expires_at: float | None = None
    tags: list["str"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class AdminAuditEntry(BaseModel):
    """A single entry in the admin audit log.

    Attributes:
        event_id: Unique identifier for this audit event.
        action: The action that was performed (e.g. ``"create_key"``).
        key_id: The API key ID involved, if applicable.
        actor: Identifier of the admin who performed the action.
        created_at: Unix timestamp of when the action occurred.
        details: Additional details about the action.
    """

    event_id: str
    action: str
    key_id: str | None = None
    actor: str
    created_at: float
    details: dict[str, Any] = Field(default_factory=dict)


class AdminAuditLogResponse(BaseModel):
    """Response body for the ``/v1/admin/audit-logs`` endpoint.

    Attributes:
        logs: List of audit log entries.
        total: Total number of entries returned.
    """

    logs: list[AdminAuditEntry]
    total: int


class AdminKeyStatsResponse(BaseModel):
    """Response body for the ``/v1/admin/keys/stats`` endpoint.

    Attributes:
        total: Total number of API keys.
        roles: Count of keys grouped by role.
        statuses: Count of keys grouped by status.
        audit_entries: Total number of audit log entries.
    """

    total: int
    roles: dict[str, int]
    statuses: dict[str, int]
    audit_entries: int
