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

"""Worker-owned auth helpers shared by eSurge API servers.

Provides a mixin class that adds API key extraction, admin
verification, request authorization, and usage recording to
FastAPI-based eSurge API server classes.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import HTTPException, Request

from .admin_state import AdminState, AuthorizationError, QuotaExceededError, RateLimitError


class AuthEndpointsMixin:
    """Concrete auth mixin shared by eSurge-serving APIs.

    Intended to be mixed into a FastAPI application class that already
    defines :attr:`admin_state`, :attr:`admin_api_key`, and
    :attr:`require_api_key`.  Provides methods for extracting API keys
    from requests, verifying admin privileges, authorizing requests
    against configured limits, and recording post-request token usage.

    Attributes:
        admin_state: The :class:`AdminState` instance managing keys.
        admin_api_key: Optional bootstrap admin key for initial setup.
        require_api_key: Whether API key authentication is mandatory.
    """

    admin_state: AdminState
    admin_api_key: str | None
    require_api_key: bool

    def _extract_api_key(self, raw_request: Request) -> str | None:
        """Extract an API key from a FastAPI request.

        Searches in order: ``Authorization: Bearer ...`` header,
        ``X-API-Key`` header, ``api_key`` query parameter, and
        ``user`` query parameter.

        Args:
            raw_request: The incoming FastAPI request object.

        Returns:
            The extracted API key string, or ``None`` if no key was
            found.
        """
        auth_header = raw_request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header.split(" ", 1)[1].strip()

        for _header_name in ("X-API-Key", "x-api-key"):
            header_value = raw_request.headers.get(_header_name)
            if header_value:
                return header_value.strip()

        for _query_name in ("api_key", "user"):
            query_value = raw_request.query_params.get(_query_name)
            if query_value:
                return query_value.strip()

        return None

    def _admin_actor(self, token: str | None) -> str:
        """Resolve a human-readable actor name from an API token.

        Used for audit logging to identify who performed an admin
        action.

        Args:
            token: The raw bearer token, or ``None`` for anonymous.

        Returns:
            A string identifying the actor (e.g. ``"bootstrap-admin"``,
            ``"anonymous"``, or the key's name).
        """
        if token is None:
            return "anonymous"
        if self.admin_api_key is not None and secrets.compare_digest(token, self.admin_api_key):
            return "bootstrap-admin"
        record = self.admin_state.validate_token(token, role="admin")
        if record is None:
            return "unknown-admin"
        return str(record.get("name") or record.get("key_id") or "admin")

    def _require_admin(self, raw_request: Request) -> str | None:
        """Verify that the request carries valid admin credentials.

        Args:
            raw_request: The incoming FastAPI request object.

        Returns:
            The validated admin token string, or ``None`` if no admin
            key is configured and no token was provided.

        Raises:
            HTTPException: With status 401 if an admin key is required
                but not provided, or 403 if the provided key is invalid
                or lacks admin privileges.
        """
        token = self._extract_api_key(raw_request)
        if token is None and self.admin_api_key is None:
            return None
        if token is None:
            raise HTTPException(status_code=401, detail="Admin API key required")
        if self.admin_api_key is not None and secrets.compare_digest(token, self.admin_api_key):
            return token
        record = self.admin_state.validate_token(token, role="admin")
        if record is not None:
            return token
        raise HTTPException(status_code=403, detail="Admin API key is invalid or lacks admin privileges")

    def _authorize_request(
        self,
        raw_request: Request | None,
        *,
        endpoint: str,
        model: str,
        requested_tokens: int,
        uses_streaming: bool = False,
        uses_function_calling: bool = False,
    ) -> dict[str, Any] | None:
        """Authorize an API request and attach key metadata to the request state.

        Extracts the API key, checks it against the bootstrap admin key
        and the :class:`AdminState`, and on success stores the key and
        its metadata on ``raw_request.state``.

        Args:
            raw_request: The incoming FastAPI request, or ``None`` to
                skip authorization entirely.
            endpoint: The API endpoint being accessed.
            model: The model name requested.
            requested_tokens: Estimated token count for rate limiting.
            uses_streaming: Whether the request uses streaming.
            uses_function_calling: Whether the request uses function
                calling.

        Returns:
            The key metadata dictionary on success, or ``None`` if no
            key was provided and authentication is not required.

        Raises:
            HTTPException: With status 401 if a key is required but
                missing, 403 for permission failures, or 429 for rate
                limit / quota violations.
        """
        if raw_request is None:
            return None

        token = self._extract_api_key(raw_request)
        if token is None:
            if self.require_api_key:
                raise HTTPException(status_code=401, detail="Missing or invalid API key")
            return None

        if self.admin_api_key is not None and secrets.compare_digest(token, self.admin_api_key):
            raw_request.state.api_key = token
            raw_request.state.api_key_metadata = {"role": "admin", "name": "bootstrap-admin"}
            return raw_request.state.api_key_metadata

        ip_address = raw_request.client.host if raw_request.client is not None else None
        try:
            metadata = self.admin_state.authorize_request(
                token,
                endpoint=endpoint,
                model=model,
                requested_tokens=requested_tokens,
                ip_address=ip_address,
                uses_streaming=uses_streaming,
                uses_function_calling=uses_function_calling,
            )
        except RateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except QuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except AuthorizationError as exc:
            if self.require_api_key:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        raw_request.state.api_key = token
        raw_request.state.api_key_metadata = metadata
        return metadata

    def _record_api_key_usage(
        self,
        raw_request: Request | None,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage for the API key attached to a request.

        Silently skips recording if the request is ``None``, no key
        was attached, or the key is the bootstrap admin key.

        Args:
            raw_request: The FastAPI request whose ``state.api_key``
                should be charged, or ``None``.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
        """
        if raw_request is None:
            return
        api_key = getattr(raw_request.state, "api_key", None)
        if not api_key or (self.admin_api_key is not None and secrets.compare_digest(api_key, self.admin_api_key)):
            return
        self.admin_state.record_usage(
            api_key,
            prompt_tokens=max(int(prompt_tokens), 0),
            completion_tokens=max(int(completion_tokens), 0),
        )


__all__ = "AuthEndpointsMixin"
