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

"""In-memory response store for OpenAI Responses API state.

Provides a thread-safe, LRU-evicting in-memory cache for response
payloads and conversation histories used by the ``/v1/responses``
API endpoint.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(slots=True)
class ResponseRecord:
    """A single cached response entry.

    Attributes:
        response_id: The unique response identifier.
        payload: The full response payload dictionary.
        created_at: Unix timestamp when the record was first stored.
        touched_at: Unix timestamp of the most recent access or update.
    """

    response_id: str
    payload: dict[str, Any]
    created_at: float
    touched_at: float


@dataclass(slots=True)
class ConversationRecord:
    """A single cached conversation entry.

    Attributes:
        conversation_id: The unique conversation identifier.
        history: Ordered list of message dictionaries.
        created_at: Unix timestamp when the record was first stored.
        touched_at: Unix timestamp of the most recent access or update.
    """

    conversation_id: str
    history: list[dict[str, Any]]
    created_at: float
    touched_at: float


class InMemoryResponseStore:
    """Thread-safe LRU response store for /v1/responses state.

    All operations are guarded by a reentrant lock for safe concurrent
    access.  When capacity limits are exceeded, the least recently used
    entries are evicted automatically.

    Example:
        >>> store = InMemoryResponseStore(max_stored_responses=100)
        >>> store.put_response("resp_1", {"model": "llama3"})
        >>> store.get_response("resp_1")
        {'model': 'llama3'}
    """

    def __init__(
        self,
        *,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
    ) -> None:
        """Initialize the in-memory response store.

        Args:
            max_stored_responses: Maximum number of response records
                to keep. Set to ``0`` to disable response caching.
            max_stored_conversations: Maximum number of conversation
                records to keep. Set to ``0`` to disable conversation
                caching.
        """
        self._lock = RLock()
        self._max_stored_responses = max(0, int(max_stored_responses))
        self._max_stored_conversations = max(0, int(max_stored_conversations))
        self._responses: OrderedDict[str, ResponseRecord] = OrderedDict()
        self._conversations: OrderedDict[str, ConversationRecord] = OrderedDict()

    @staticmethod
    def _lru_set(store: OrderedDict[str, Any], key: str, value: Any, max_size: int) -> None:
        """Insert or update an entry in an LRU ordered dict, evicting as needed.

        Args:
            store: The :class:`~collections.OrderedDict` to update.
            key: The entry key.
            value: The entry value.
            max_size: Maximum number of entries. If ``0`` or negative,
                the store is cleared after insertion.
        """
        store[key] = value
        store.move_to_end(key)
        if max_size <= 0:
            store.clear()
            return
        while len(store) > max_size:
            store.popitem(last=False)

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        """Retrieve a response payload by its ID.

        Args:
            response_id: The unique response identifier.

        Returns:
            A shallow copy of the response payload dictionary, or
            ``None`` if not found.
        """
        if not response_id:
            return None
        with self._lock:
            record = self._responses.get(response_id)
            if record is None:
                return None
            record.touched_at = time.time()
            self._responses.move_to_end(response_id)
            return dict(record.payload)

    def put_response(self, response_id: str, payload: dict[str, Any]) -> ResponseRecord | None:
        """Store or update a response payload.

        Args:
            response_id: The unique response identifier.
            payload: The response payload dictionary to store.

        Returns:
            The created or updated :class:`ResponseRecord`, or ``None``
            if response storage is disabled.
        """
        if not response_id or self._max_stored_responses == 0:
            return None
        now = time.time()
        with self._lock:
            previous = self._responses.get(response_id)
            created_at = previous.created_at if previous is not None else now
            record = ResponseRecord(
                response_id=response_id,
                payload=dict(payload),
                created_at=created_at,
                touched_at=now,
            )
            self._lru_set(self._responses, response_id, record, self._max_stored_responses)
            return record

    def delete_response(self, response_id: str) -> bool:
        """Delete a stored response by its ID.

        Args:
            response_id: The unique response identifier.

        Returns:
            ``True`` if the response was found and deleted, ``False``
            otherwise.
        """
        if not response_id:
            return False
        with self._lock:
            return self._responses.pop(response_id, None) is not None

    def get_conversation(self, conversation_id: str) -> list[dict[str, Any]] | None:
        """Retrieve a conversation history by its ID.

        Args:
            conversation_id: The unique conversation identifier.

        Returns:
            A list of shallow-copied message dictionaries, or ``None``
            if not found.
        """
        if not conversation_id:
            return None
        with self._lock:
            record = self._conversations.get(conversation_id)
            if record is None:
                return None
            record.touched_at = time.time()
            self._conversations.move_to_end(conversation_id)
            return [dict(message) for message in record.history]

    def put_conversation(self, conversation_id: str, history: list[dict[str, Any]]) -> ConversationRecord | None:
        """Store or update a conversation history.

        Args:
            conversation_id: The unique conversation identifier.
            history: List of message dictionaries to store.

        Returns:
            The created or updated :class:`ConversationRecord`, or
            ``None`` if conversation storage is disabled.
        """
        if not conversation_id or self._max_stored_conversations == 0:
            return None
        now = time.time()
        normalized_history = [dict(message) for message in history]
        with self._lock:
            previous = self._conversations.get(conversation_id)
            created_at = previous.created_at if previous is not None else now
            record = ConversationRecord(
                conversation_id=conversation_id,
                history=normalized_history,
                created_at=created_at,
                touched_at=now,
            )
            self._lru_set(self._conversations, conversation_id, record, self._max_stored_conversations)
            return record

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a stored conversation by its ID.

        Args:
            conversation_id: The unique conversation identifier.

        Returns:
            ``True`` if the conversation was found and deleted,
            ``False`` otherwise.
        """
        if not conversation_id:
            return False
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    def stats(self) -> dict[str, int]:
        """Return current storage statistics.

        Returns:
            A dictionary with keys ``"responses"``,
            ``"conversations"``, ``"max_stored_responses"``, and
            ``"max_stored_conversations"``.
        """
        with self._lock:
            return {
                "responses": len(self._responses),
                "conversations": len(self._conversations),
                "max_stored_responses": self._max_stored_responses,
                "max_stored_conversations": self._max_stored_conversations,
            }
