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

"""Worker-owned admin/API-key state with request authorization and audit logs.

Implements an in-memory API key management system with support for
role-based access control, rate limiting, quota enforcement, IP
allow/block lists, and lightweight audit logging.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

_VALID_ROLES = {"admin", "user", "readonly", "service"}
_VALID_STATUSES = {"active", "suspended", "expired", "revoked"}
_DAY_SECONDS = 86400.0


class AuthorizationError(Exception):
    """Raised when an API key is valid but not permitted to perform an action.

    This is the base exception for all authorization failures including
    permission denials, feature restrictions, and general access control
    violations.
    """


class RateLimitError(AuthorizationError):
    """Raised when a rate limit would be exceeded.

    Indicates that the per-minute, per-hour, or per-day request or
    token rate limit for an API key has been reached.
    """


class QuotaExceededError(AuthorizationError):
    """Raised when a quota would be exceeded.

    Indicates that the total or monthly request/token quota for an
    API key has been exhausted.
    """


@dataclass(slots=True)
class AuditEntry:
    """A single audit log entry recording an administrative action.

    Attributes:
        event_id: Unique identifier for this audit event (e.g.
            ``"evt_abc123def456"``).
        action: The type of action that occurred (e.g. ``"key_created"``,
            ``"key_revoked"``).
        key_id: The API key ID involved, or ``None`` for system-level
            events.
        actor: Identifier of the user or process that performed the
            action.
        created_at: Unix timestamp when the event was recorded.
        details: Optional dictionary of additional context.
    """

    event_id: str
    action: str
    key_id: str | None
    actor: str
    created_at: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize this audit entry to a plain dictionary.

        Returns:
            A dictionary representation of all fields suitable for
            JSON serialization.
        """
        return {
            "event_id": self.event_id,
            "action": self.action,
            "key_id": self.key_id,
            "actor": self.actor,
            "created_at": self.created_at,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ApiKeyRecord:
    """Full mutable record for a single API key.

    Stores identity, permissions, rate-limit configuration, quota
    limits, and cumulative usage statistics.  The record also tracks
    sliding-window event timestamps for rate-limit enforcement.

    Attributes:
        key_id: Unique identifier for this key (e.g. ``"key_abc123"``).
        key_prefix: First 12 characters of the raw key for display.
        name: Human-readable name for the key.
        role: One of ``"admin"``, ``"user"``, ``"readonly"``, or
            ``"service"``.
        description: Optional free-text description.
        status: One of ``"active"``, ``"suspended"``, ``"expired"``, or
            ``"revoked"``.
        created_at: Unix timestamp when the key was created.
        expires_at: Optional Unix timestamp after which the key is
            considered expired.
        tags: Free-form tags for organizational purposes.
        metadata: Arbitrary key-value metadata.
        requests_per_minute: Optional per-minute request rate limit.
        requests_per_hour: Optional per-hour request rate limit.
        requests_per_day: Optional per-day request rate limit.
        tokens_per_minute: Optional per-minute token rate limit.
        tokens_per_hour: Optional per-hour token rate limit.
        tokens_per_day: Optional per-day token rate limit.
        max_total_tokens: Optional lifetime token cap.
        max_total_requests: Optional lifetime request cap.
        monthly_token_limit: Optional monthly token quota.
        monthly_request_limit: Optional monthly request quota.
        allowed_models: Whitelist of model names; ``None`` means all.
        allowed_endpoints: Whitelist of endpoints; ``None`` means all.
        allowed_ip_addresses: IP allowlist; ``None`` means all.
        blocked_ip_addresses: IP blocklist; ``None`` means none blocked.
        enable_streaming: Whether streaming responses are permitted.
        enable_function_calling: Whether function calling is permitted.
        max_tokens_per_request: Optional per-request token cap.
        last_used_at: Unix timestamp of most recent usage.
        total_requests: Cumulative request count.
        total_prompt_tokens: Cumulative prompt token count.
        total_completion_tokens: Cumulative completion token count.
        total_tokens: Cumulative total token count.
        monthly_usage_period: Current month key (``"YYYY-MM"``).
        monthly_requests: Requests counted in the current month.
        monthly_tokens: Tokens counted in the current month.
        recent_requests: Deque of Unix timestamps for recent requests.
        recent_token_events: Deque of ``(timestamp, token_count)``
            tuples for recent token usage events.
    """

    key_id: str
    key_prefix: str
    name: str
    role: str
    description: str | None
    status: str
    created_at: float
    expires_at: float | None = None
    tags: list["str"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_hour: int | None = None
    tokens_per_day: int | None = None
    max_total_tokens: int | None = None
    max_total_requests: int | None = None
    monthly_token_limit: int | None = None
    monthly_request_limit: int | None = None
    allowed_models: list["str"] | None = None
    allowed_endpoints: list["str"] | None = None
    allowed_ip_addresses: list["str"] | None = None
    blocked_ip_addresses: list["str"] | None = None
    enable_streaming: bool = True
    enable_function_calling: bool = True
    max_tokens_per_request: int | None = None
    last_used_at: float | None = None
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    monthly_usage_period: str = field(default_factory=lambda: time.strftime("%Y-%m", time.gmtime()))
    monthly_requests: int = 0
    monthly_tokens: int = 0
    recent_requests: deque["float"] = field(default_factory=deque)
    recent_token_events: deque[tuple[float, int]] = field(default_factory=deque)

    def effective_status(self) -> str:
        """Return the runtime-resolved status accounting for expiry.

        If the stored status is ``"active"`` but ``expires_at`` is in
        the past, returns ``"expired"`` instead.

        Returns:
            The effective status string.
        """
        if self.status == "active" and self.expires_at is not None and self.expires_at <= time.time():
            return "expired"
        return self.status

    def as_public_dict(self) -> dict[str, Any]:
        """Serialize this record to a dictionary safe for API responses.

        The raw secret key is **not** included.  Mutable container
        fields are shallow-copied to prevent external mutation.

        Returns:
            A dictionary containing all public fields of the record.
        """
        return {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "status": self.effective_status(),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_hour": self.tokens_per_hour,
            "tokens_per_day": self.tokens_per_day,
            "max_total_tokens": self.max_total_tokens,
            "max_total_requests": self.max_total_requests,
            "monthly_token_limit": self.monthly_token_limit,
            "monthly_request_limit": self.monthly_request_limit,
            "allowed_models": list(self.allowed_models) if self.allowed_models is not None else None,
            "allowed_endpoints": list(self.allowed_endpoints) if self.allowed_endpoints is not None else None,
            "allowed_ip_addresses": list(self.allowed_ip_addresses) if self.allowed_ip_addresses is not None else None,
            "blocked_ip_addresses": list(self.blocked_ip_addresses) if self.blocked_ip_addresses is not None else None,
            "enable_streaming": self.enable_streaming,
            "enable_function_calling": self.enable_function_calling,
            "max_tokens_per_request": self.max_tokens_per_request,
            "last_used_at": self.last_used_at,
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "monthly_usage_period": self.monthly_usage_period,
            "monthly_requests": self.monthly_requests,
            "monthly_tokens": self.monthly_tokens,
        }


class AdminState:
    """In-memory API-key manager with lightweight audit logging.

    Provides thread-safe CRUD operations for API keys, request
    authorization with rate-limit and quota enforcement, usage
    recording, and audit log retrieval.  All state is held in memory
    and protected by a reentrant lock.

    Example:
        >>> state = AdminState()
        >>> raw_key, info = state.create_key(actor="admin", name="test")
        >>> state.authorize_request(
        ...     raw_key, endpoint="/v1/chat", model="llama3",
        ...     requested_tokens=100,
        ... )
    """

    def __init__(self):
        """Initialize the admin state with empty key and audit stores."""
        self._lock = RLock()
        self._records: dict[str, ApiKeyRecord] = {}
        self._tokens: dict[str, str] = {}
        self._audit: list[AuditEntry] = []

    def _normalize_role(self, role: str | None) -> str:
        """Normalize and validate a role string.

        Args:
            role: Raw role string, or ``None`` to default to ``"admin"``.

        Returns:
            The lowercased, stripped role string.

        Raises:
            ValueError: If the role is not in the set of valid roles.
        """
        normalized = (role or "admin").strip().lower()
        if normalized not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role!r}")
        return normalized

    def _normalize_status(self, status: str | None) -> str:
        """Normalize and validate a status string.

        Args:
            status: Raw status string, or ``None`` for empty.

        Returns:
            The lowercased, stripped status string (may be empty).

        Raises:
            ValueError: If the status is non-empty and not in the set
                of valid statuses.
        """
        normalized = (status or "").strip().lower()
        if normalized and normalized not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status!r}")
        return normalized

    def _append_audit(
        self, *, action: str, key_id: str | None, actor: str, details: dict[str, Any] | None = None
    ) -> None:
        """Append a new entry to the internal audit log.

        Args:
            action: Short action identifier (e.g. ``"key_created"``).
            key_id: The API key ID involved, or ``None``.
            actor: Identifier of the acting user or process.
            details: Optional extra context dictionary.
        """
        self._audit.append(
            AuditEntry(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                action=action,
                key_id=key_id,
                actor=actor,
                created_at=time.time(),
                details=dict(details or {}),
            )
        )

    @staticmethod
    def _current_month_key(now: float | None = None) -> str:
        """Return the current month as a ``"YYYY-MM"`` string.

        Args:
            now: Optional Unix timestamp; defaults to the current time.

        Returns:
            A string like ``"2026-03"`` for the given timestamp.
        """
        return time.strftime("%Y-%m", time.gmtime(time.time() if now is None else now))

    @staticmethod
    def _prune_request_events(record: ApiKeyRecord, *, now: float) -> None:
        """Remove sliding-window events older than one day.

        Args:
            record: The API key record to prune.
            now: Current Unix timestamp used as the reference point.
        """
        while record.recent_requests and (now - record.recent_requests[0]) > _DAY_SECONDS:
            record.recent_requests.popleft()
        while record.recent_token_events and (now - record.recent_token_events[0][0]) > _DAY_SECONDS:
            record.recent_token_events.popleft()

    @staticmethod
    def _count_requests(record: ApiKeyRecord, *, now: float, window_seconds: float) -> int:
        """Count the number of requests within a sliding time window.

        Args:
            record: The API key record to inspect.
            now: Current Unix timestamp.
            window_seconds: Size of the sliding window in seconds.

        Returns:
            Number of requests that occurred within the window.
        """
        return sum(1 for event_time in record.recent_requests if (now - event_time) <= window_seconds)

    @staticmethod
    def _count_tokens(record: ApiKeyRecord, *, now: float, window_seconds: float) -> int:
        """Sum the tokens consumed within a sliding time window.

        Args:
            record: The API key record to inspect.
            now: Current Unix timestamp.
            window_seconds: Size of the sliding window in seconds.

        Returns:
            Total tokens consumed within the window.
        """
        return sum(tokens for event_time, tokens in record.recent_token_events if (now - event_time) <= window_seconds)

    def _ensure_month_window(self, record: ApiKeyRecord, *, now: float) -> None:
        """Reset monthly counters if the calendar month has changed.

        Args:
            record: The API key record to check and potentially reset.
            now: Current Unix timestamp used to determine the month.
        """
        month_key = self._current_month_key(now)
        if record.monthly_usage_period == month_key:
            return
        record.monthly_usage_period = month_key
        record.monthly_requests = 0
        record.monthly_tokens = 0

    @staticmethod
    def _get_ip_permission_error(record: ApiKeyRecord, ip_address: str | None) -> str | None:
        """Check IP allow/block lists and return an error message if denied.

        Args:
            record: The API key record whose IP rules are checked.
            ip_address: The client IP address, or ``None`` if unknown.

        Returns:
            An error message string if the IP is blocked or not
            allowed, or ``None`` if the IP passes all checks.
        """
        if ip_address is None:
            return None
        if record.blocked_ip_addresses and ip_address in record.blocked_ip_addresses:
            return f"IP address {ip_address!r} is blocked for this API key"
        if record.allowed_ip_addresses and ip_address not in record.allowed_ip_addresses:
            return f"IP address {ip_address!r} is not allowed for this API key"
        return None

    @staticmethod
    def _check_upper_bound(
        current: int, incoming: int, limit: int | None, *, message: str, exc_type: type[Exception]
    ) -> None:
        """Raise if adding ``incoming`` to ``current`` would exceed ``limit``.

        Args:
            current: The current counter value.
            incoming: The amount to be added.
            limit: The upper bound, or ``None`` to skip the check.
            message: Error message for the raised exception.
            exc_type: The exception class to raise on violation.

        Raises:
            Exception: Of type ``exc_type`` if the limit would be
                exceeded.
        """
        if limit is None:
            return
        if current + incoming > int(limit):
            raise exc_type(message)

    def create_key(self, *, actor: str = "bootstrap", **payload: Any) -> tuple[str, dict[str, Any]]:
        """Create a new API key with the given configuration.

        Args:
            actor: Identifier of the user creating the key (used in
                audit logs).
            **payload: Key configuration fields including ``name``,
                ``role``, ``description``, ``expires_in_days``, ``tags``,
                ``metadata``, rate-limit fields, quota fields, and
                access-control lists.

        Returns:
            A tuple of ``(raw_key, response_dict)`` where ``raw_key``
            is the secret bearer token and ``response_dict`` contains
            the public key metadata plus the raw key.

        Raises:
            ValueError: If an invalid role is specified.
        """
        role = self._normalize_role(payload.get("role"))
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        raw_key = f"sk-emx-{secrets.token_urlsafe(24)}"
        created_at = time.time()
        expires_in_days = payload.get("expires_in_days")
        expires_at = created_at + (float(expires_in_days) * 86400.0) if expires_in_days is not None else None
        record = ApiKeyRecord(
            key_id=key_id,
            key_prefix=raw_key[:12],
            name=str(payload.get("name") or key_id),
            role=role,
            description=payload.get("description"),
            status="active",
            created_at=created_at,
            expires_at=expires_at,
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            requests_per_minute=payload.get("requests_per_minute"),
            requests_per_hour=payload.get("requests_per_hour"),
            requests_per_day=payload.get("requests_per_day"),
            tokens_per_minute=payload.get("tokens_per_minute"),
            tokens_per_hour=payload.get("tokens_per_hour"),
            tokens_per_day=payload.get("tokens_per_day"),
            max_total_tokens=payload.get("max_total_tokens"),
            max_total_requests=payload.get("max_total_requests"),
            monthly_token_limit=payload.get("monthly_token_limit"),
            monthly_request_limit=payload.get("monthly_request_limit"),
            allowed_models=list(payload.get("allowed_models") or []) or None,
            allowed_endpoints=list(payload.get("allowed_endpoints") or []) or None,
            allowed_ip_addresses=list(payload.get("allowed_ip_addresses") or []) or None,
            blocked_ip_addresses=list(payload.get("blocked_ip_addresses") or []) or None,
            enable_streaming=bool(payload.get("enable_streaming", True)),
            enable_function_calling=bool(payload.get("enable_function_calling", True)),
            max_tokens_per_request=payload.get("max_tokens_per_request"),
        )

        with self._lock:
            self._records[key_id] = record
            self._tokens[raw_key] = key_id
            self._append_audit(action="key_created", key_id=key_id, actor=actor, details={"role": role})

        response = record.as_public_dict()
        response.update(
            {
                "key": raw_key,
                "message": "API key created successfully. Store this key securely - it won't be shown again!",
            }
        )
        return raw_key, response

    def list_keys(self, *, role: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        """List all API keys, optionally filtered by role and/or status.

        Args:
            role: If provided, only keys matching this role are returned.
            status: If provided, only keys matching this effective
                status are returned.

        Returns:
            A list of public key dictionaries sorted by creation time
            in descending order (newest first).

        Raises:
            ValueError: If an invalid role or status is specified.
        """
        normalized_role = self._normalize_role(role) if role is not None else None
        normalized_status = self._normalize_status(status) if status is not None else None
        with self._lock:
            records = [record.as_public_dict() for record in self._records.values()]
        filtered: list[dict[str, Any]] = []
        for record in records:
            if normalized_role is not None and record["role"] != normalized_role:
                continue
            if normalized_status is not None and record["status"] != normalized_status:
                continue
            filtered.append(record)
        return sorted(filtered, key=lambda item: item["created_at"], reverse=True)

    def get_key(self, key_id: str) -> dict[str, Any] | None:
        """Retrieve a single API key's public metadata by its ID.

        Args:
            key_id: The unique key identifier.

        Returns:
            A public key dictionary, or ``None`` if not found.
        """
        with self._lock:
            record = self._records.get(key_id)
            return record.as_public_dict() if record is not None else None

    def _get_record(self, key_id: str) -> ApiKeyRecord:
        """Look up an :class:`ApiKeyRecord` by ID or raise.

        Args:
            key_id: The unique key identifier.

        Returns:
            The matching :class:`ApiKeyRecord`.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        record = self._records.get(key_id)
        if record is None:
            raise KeyError("key_id")
        return record

    def update_key(self, key_id: str, *, actor: str = "bootstrap", **changes: Any) -> dict[str, Any]:
        """Update mutable fields of an existing API key.

        Only fields present in ``changes`` with non-``None`` values
        that correspond to actual record attributes will be applied.

        Args:
            key_id: The unique key identifier to update.
            actor: Identifier of the user performing the update.
            **changes: Field-name/value pairs to apply.

        Returns:
            The updated public key dictionary.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
            ValueError: If an invalid role value is specified.
        """
        with self._lock:
            record = self._get_record(key_id)
            for field_name, value in changes.items():
                if value is None or not hasattr(record, field_name):
                    continue
                if field_name in {"role"}:
                    value = self._normalize_role(value)
                elif field_name in {
                    "allowed_models",
                    "allowed_endpoints",
                    "allowed_ip_addresses",
                    "blocked_ip_addresses",
                    "tags",
                }:
                    value = list(value)
                elif field_name == "metadata":
                    value = dict(value)
                setattr(record, field_name, value)
            self._append_audit(action="key_updated", key_id=key_id, actor=actor, details={"fields": sorted(changes)})
            return record.as_public_dict()

    def revoke_key(self, key_id: str, *, actor: str = "bootstrap") -> dict[str, Any]:
        """Permanently revoke an API key.

        Args:
            key_id: The unique key identifier to revoke.
            actor: Identifier of the user performing the revocation.

        Returns:
            The updated public key dictionary with status ``"revoked"``.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        with self._lock:
            record = self._get_record(key_id)
            record.status = "revoked"
            self._append_audit(action="key_revoked", key_id=key_id, actor=actor)
            return record.as_public_dict()

    def suspend_key(self, key_id: str, *, actor: str = "bootstrap") -> dict[str, Any]:
        """Temporarily suspend an API key.

        Args:
            key_id: The unique key identifier to suspend.
            actor: Identifier of the user performing the suspension.

        Returns:
            The updated public key dictionary with status
            ``"suspended"``.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        with self._lock:
            record = self._get_record(key_id)
            record.status = "suspended"
            self._append_audit(action="key_suspended", key_id=key_id, actor=actor)
            return record.as_public_dict()

    def reactivate_key(self, key_id: str, *, actor: str = "bootstrap") -> dict[str, Any]:
        """Reactivate a previously suspended or revoked API key.

        Args:
            key_id: The unique key identifier to reactivate.
            actor: Identifier of the user performing the reactivation.

        Returns:
            The updated public key dictionary with status ``"active"``.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        with self._lock:
            record = self._get_record(key_id)
            record.status = "active"
            self._append_audit(action="key_reactivated", key_id=key_id, actor=actor)
            return record.as_public_dict()

    def delete_key(self, key_id: str, *, actor: str = "bootstrap") -> dict[str, Any]:
        """Permanently delete an API key and all its token mappings.

        Args:
            key_id: The unique key identifier to delete.
            actor: Identifier of the user performing the deletion.

        Returns:
            The public key dictionary of the deleted record.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        with self._lock:
            record = self._get_record(key_id)
            response = record.as_public_dict()
            raw_keys = [token for token, mapped_id in self._tokens.items() if mapped_id == key_id]
            for _token in raw_keys:
                self._tokens.pop(_token, None)
            self._records.pop(key_id, None)
            self._append_audit(action="key_deleted", key_id=key_id, actor=actor)
        return response

    def rotate_key(self, key_id: str, *, actor: str = "bootstrap") -> tuple[str, dict[str, Any]]:
        """Rotate the secret token for an existing API key.

        Generates a new bearer token, invalidates the old one, and
        updates the key prefix.

        Args:
            key_id: The unique key identifier whose token to rotate.
            actor: Identifier of the user performing the rotation.

        Returns:
            A tuple of ``(new_raw_key, response_dict)`` where
            ``new_raw_key`` is the new secret bearer token.

        Raises:
            KeyError: If no record exists for the given ``key_id``.
        """
        new_key = f"sk-emx-{secrets.token_urlsafe(24)}"
        with self._lock:
            record = self._get_record(key_id)
            raw_keys = [token for token, mapped_id in self._tokens.items() if mapped_id == key_id]
            for _token in raw_keys:
                self._tokens.pop(_token, None)
            self._tokens[new_key] = key_id
            record.key_prefix = new_key[:12]
            self._append_audit(action="key_rotated", key_id=key_id, actor=actor)
            response = record.as_public_dict()
        response["key"] = new_key
        response["message"] = "API key rotated successfully. Store the new key securely."
        return new_key, response

    def authorize_request(
        self,
        token: str,
        *,
        endpoint: str,
        model: str,
        requested_tokens: int,
        ip_address: str | None = None,
        uses_streaming: bool = False,
        uses_function_calling: bool = False,
    ) -> dict[str, Any]:
        """Authorize an incoming API request against all configured limits.

        Validates the bearer token, checks key status, IP restrictions,
        endpoint/model allowlists, feature flags, rate limits, and
        quotas.  On success, records the request in the sliding window
        and increments usage counters.

        Args:
            token: The raw bearer token from the request.
            endpoint: The API endpoint being accessed (e.g.
                ``"/v1/chat/completions"``).
            model: The model name requested.
            requested_tokens: Estimated number of tokens for this
                request (used for token-based rate limiting).
            ip_address: Client IP address for IP-based access control,
                or ``None`` if unavailable.
            uses_streaming: Whether the request uses streaming.
            uses_function_calling: Whether the request uses function
                calling.

        Returns:
            The public key dictionary for the authorized key.

        Raises:
            AuthorizationError: If the token is invalid, the key is
                inactive, or a permission check fails.
            RateLimitError: If a rate limit would be exceeded.
            QuotaExceededError: If a quota would be exceeded.
        """
        now = time.time()
        requested_tokens = max(int(requested_tokens), 0)
        with self._lock:
            key_id = self._tokens.get(token)
            if key_id is None:
                raise AuthorizationError("Missing or invalid API key")
            record = self._records.get(key_id)
            if record is None:
                raise AuthorizationError("Missing or invalid API key")
            if record.effective_status() != "active":
                raise AuthorizationError("API key is inactive")

            ip_error = self._get_ip_permission_error(record, ip_address)
            if ip_error is not None:
                raise AuthorizationError("ip_error")

            if record.allowed_endpoints and endpoint not in record.allowed_endpoints:
                raise AuthorizationError(f"Endpoint {endpoint!r} is not allowed for this API key")
            if record.allowed_models and model and model not in record.allowed_models:
                raise AuthorizationError(f"Model {model!r} is not allowed for this API key")
            if uses_streaming and not record.enable_streaming:
                raise AuthorizationError("Streaming is disabled for this API key")
            if uses_function_calling and not record.enable_function_calling:
                raise AuthorizationError("Function calling is disabled for this API key")
            if record.max_tokens_per_request is not None and requested_tokens > record.max_tokens_per_request:
                raise AuthorizationError(
                    f"Requested tokens ({requested_tokens}) exceed the max_tokens_per_request limit "
                    f"({record.max_tokens_per_request})"
                )

            self._ensure_month_window(record, now=now)
            self._prune_request_events(record, now=now)

            self._check_upper_bound(
                self._count_requests(record, now=now, window_seconds=60.0),
                1,
                record.requests_per_minute,
                message="Requests per minute limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                self._count_requests(record, now=now, window_seconds=3600.0),
                1,
                record.requests_per_hour,
                message="Requests per hour limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                self._count_requests(record, now=now, window_seconds=_DAY_SECONDS),
                1,
                record.requests_per_day,
                message="Requests per day limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                self._count_tokens(record, now=now, window_seconds=60.0),
                requested_tokens,
                record.tokens_per_minute,
                message="Tokens per minute limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                self._count_tokens(record, now=now, window_seconds=3600.0),
                requested_tokens,
                record.tokens_per_hour,
                message="Tokens per hour limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                self._count_tokens(record, now=now, window_seconds=_DAY_SECONDS),
                requested_tokens,
                record.tokens_per_day,
                message="Tokens per day limit exceeded",
                exc_type=RateLimitError,
            )
            self._check_upper_bound(
                record.total_requests,
                1,
                record.max_total_requests,
                message="Total request quota exceeded",
                exc_type=QuotaExceededError,
            )
            self._check_upper_bound(
                record.total_tokens,
                requested_tokens,
                record.max_total_tokens,
                message="Total token quota exceeded",
                exc_type=QuotaExceededError,
            )
            self._check_upper_bound(
                record.monthly_requests,
                1,
                record.monthly_request_limit,
                message="Monthly request quota exceeded",
                exc_type=QuotaExceededError,
            )
            self._check_upper_bound(
                record.monthly_tokens,
                requested_tokens,
                record.monthly_token_limit,
                message="Monthly token quota exceeded",
                exc_type=QuotaExceededError,
            )

            record.last_used_at = now
            record.total_requests += 1
            record.monthly_requests += 1
            record.recent_requests.append(now)
            record.recent_token_events.append((now, requested_tokens))
            return record.as_public_dict()

    def record_usage(self, token: str, *, prompt_tokens: int, completion_tokens: int) -> dict[str, Any] | None:
        """Record token usage after a request has been served.

        Updates cumulative and monthly token counters for the key
        associated with the given bearer token.

        Args:
            token: The raw bearer token identifying the key.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.

        Returns:
            The updated public key dictionary, or ``None`` if the
            token is not recognized.
        """
        total_tokens = max(int(prompt_tokens), 0) + max(int(completion_tokens), 0)
        with self._lock:
            key_id = self._tokens.get(token)
            if key_id is None:
                return None
            record = self._records.get(key_id)
            if record is None:
                return None
            self._ensure_month_window(record, now=time.time())
            record.total_prompt_tokens += max(int(prompt_tokens), 0)
            record.total_completion_tokens += max(int(completion_tokens), 0)
            record.total_tokens += total_tokens
            record.monthly_tokens += total_tokens
            record.last_used_at = time.time()
            return record.as_public_dict()

    def validate_token(self, token: str, *, role: str | None = None) -> dict[str, Any] | None:
        """Validate a bearer token and optionally require a specific role.

        Args:
            token: The raw bearer token to validate.
            role: If provided, the key must have this role to pass
                validation.

        Returns:
            The public key dictionary if the token is valid and active
            (and matches the required role, if any), or ``None``
            otherwise.
        """
        required_role = self._normalize_role(role) if role is not None else None
        with self._lock:
            key_id = self._tokens.get(token)
            if key_id is None:
                return None
            record = self._records.get(key_id)
            if record is None:
                return None
            status = record.effective_status()
            if status != "active":
                return None
            if required_role is not None and record.role != required_role:
                return None
            record.last_used_at = time.time()
            return record.as_public_dict()

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about all managed API keys.

        Returns:
            A dictionary containing total key count, counts per role
            and status, total audit entries, and cumulative request
            and token totals.
        """
        with self._lock:
            records = list(self._records.values())
            roles = {role: 0 for role in sorted(_VALID_ROLES)}
            statuses = {status: 0 for status in sorted(_VALID_STATUSES)}
            for record in records:
                roles[record.role] = roles.get(record.role, 0) + 1
                statuses[record.effective_status()] = statuses.get(record.effective_status(), 0) + 1
            return {
                "total": len(records),
                "roles": roles,
                "statuses": statuses,
                "audit_entries": len(self._audit),
                "total_requests": sum(record.total_requests for record in records),
                "total_tokens": sum(record.total_tokens for record in records),
            }

    def get_audit_logs(
        self,
        *,
        limit: int = 100,
        key_id: str | None = None,
        action: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve audit log entries with optional filtering.

        Args:
            limit: Maximum number of entries to return.
            key_id: If provided, only entries for this key are returned.
            action: If provided, only entries with this action type are
                returned.

        Returns:
            A list of audit entry dictionaries sorted by creation time
            in descending order (newest first), capped at ``limit``.
        """
        with self._lock:
            logs = list(self._audit)
        if key_id is not None:
            logs = [entry for entry in logs if entry.key_id == key_id]
        if action is not None:
            logs = [entry for entry in logs if entry.action == action]
        logs = sorted(logs, key=lambda entry: entry.created_at, reverse=True)
        return [entry.as_dict() for entry in logs[: max(int(limit), 0)]]


__all__ = ("AdminState", "ApiKeyRecord", "AuditEntry", "AuthorizationError", "QuotaExceededError", "RateLimitError")
