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

"""File-backed response store for persistent /v1/responses state.

Implements an LRU file-backed store for caching response objects and
conversation histories on disk with optional zlib compression and
atomic writes for crash safety.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("__name__")


@dataclass(slots=True)
class _Entry:
    """Internal index entry tracking a single stored file.

    Attributes:
        file: The filename (not full path) within the storage directory.
        created_at: Unix timestamp when the entry was first created.
        touched_at: Unix timestamp of the most recent access or update.
    """

    file: str
    created_at: float
    touched_at: float


class FileResponseStore:
    """Persistent LRU store for response objects and conversation histories.

    Stores response payloads and conversation histories as compressed
    binary files on disk, managed via an in-memory LRU index that is
    periodically flushed to a JSON index file.  Eviction happens
    automatically when capacity limits are exceeded.

    Attributes:
        storage_dir: Root directory for all stored files.
        responses_dir: Subdirectory for response payloads.
        conversations_dir: Subdirectory for conversation histories.
        index_file: Path to the JSON index file.
    """

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
        compression_level: int = 3,
    ) -> None:
        """Initialize the file-backed response store.

        Creates the storage directory structure if it does not exist
        and loads the existing index from disk.

        Args:
            storage_dir: Root directory for all stored data files.
            max_stored_responses: Maximum number of response records to
                keep before LRU eviction. Set to ``0`` to disable
                response storage.
            max_stored_conversations: Maximum number of conversation
                records to keep before LRU eviction. Set to ``0`` to
                disable conversation storage.
            compression_level: zlib compression level (0-9). ``0``
                disables compression; higher values yield smaller files
                at the cost of CPU time.
        """
        self.storage_dir = Path(storage_dir)
        self.responses_dir = self.storage_dir / "responses"
        self.conversations_dir = self.storage_dir / "conversations"
        self.index_file = self.storage_dir / "index.json"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self._max_stored_responses = max(0, int(max_stored_responses))
        self._max_stored_conversations = max(0, int(max_stored_conversations))
        self._compression_level = max(0, min(int(compression_level), 9))

        self._responses: OrderedDict[str, _Entry] = OrderedDict()
        self._conversations: OrderedDict[str, _Entry] = OrderedDict()
        self._dirty = False
        self._load_index()

    def _encode(self, obj: Any) -> bytes:
        """Serialize and optionally compress an object to bytes.

        Args:
            obj: A JSON-serializable object.

        Returns:
            The serialized (and possibly zlib-compressed) bytes.
        """
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self._compression_level <= 0:
            return data
        return zlib.compress(data, level=self._compression_level)

    @staticmethod
    def _decode(blob: bytes) -> Any:
        """Decompress (if needed) and deserialize bytes to a Python object.

        Args:
            blob: Raw bytes, possibly zlib-compressed.

        Returns:
            The deserialized Python object.
        """
        try:
            blob = zlib.decompress(blob)
        except zlib.error:
            pass
        return json.loads(blob.decode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write bytes to a file atomically using a temporary-then-rename strategy.

        A ``.tmp`` file is written first, then the existing file (if
        any) is backed up to ``.bak``, and finally the temporary file
        is renamed to the target path.

        Args:
            path: Destination file path.
            data: Raw bytes to write.
        """
        temporary = path.with_suffix(path.suffix + ".tmp")
        backup = path.with_suffix(path.suffix + ".bak")
        temporary.write_bytes(data)
        if path.exists():
            if backup.exists():
                backup.unlink()
            path.replace(backup)
        temporary.replace(path)

    @staticmethod
    def _load_entries(items: dict[str, Any], base_dir: Path) -> OrderedDict[str, _Entry]:
        """Reconstruct an ordered entry map from serialized index items.

        Skips entries whose backing file no longer exists on disk.

        Args:
            items: Dictionary of ``{key: metadata_dict}`` from the
                persisted index.
            base_dir: The directory that contains the backing files.

        Returns:
            An :class:`~collections.OrderedDict` of entries sorted by
            ``touched_at`` in ascending order (least recently used
            first).
        """
        loaded: list[tuple[str, _Entry]] = []
        for key, meta in items.items():
            if not isinstance(meta, dict):
                continue
            file_name = meta.get("file")
            if not isinstance(file_name, str) or not file_name:
                continue
            path = base_dir / file_name
            if not path.exists():
                continue
            created_at = float(meta.get("created_at") or 0.0)
            touched_at = float(meta.get("touched_at") or created_at or 0.0)
            loaded.append((key, _Entry(file=file_name, created_at=created_at, touched_at=touched_at)))
        loaded.sort(key=lambda item: item[1].touched_at)
        return OrderedDict(loaded)

    def _load_index(self) -> None:
        """Load the index from disk, falling back to a backup if corrupted."""
        if not self.index_file.exists():
            self._dirty = True
            self._flush_index()
            return

        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            backup = self.index_file.with_suffix(".json.bak")
            if backup.exists():
                raw = json.loads(backup.read_text(encoding="utf-8"))
            else:
                raw = {}

        responses = dict((raw.get("responses") or {}).get("items") or {})
        conversations = dict((raw.get("conversations") or {}).get("items") or {})
        self._responses = self._load_entries(responses, self.responses_dir)
        self._conversations = self._load_entries(conversations, self.conversations_dir)

    def _flush_index(self) -> None:
        """Persist the in-memory index to disk if it has been modified."""
        if not self._dirty and self.index_file.exists():
            return
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "responses": {
                "max": self._max_stored_responses,
                "items": {
                    key: {
                        "file": entry.file,
                        "created_at": entry.created_at,
                        "touched_at": entry.touched_at,
                    }
                    for key, entry in self._responses.items()
                },
            },
            "conversations": {
                "max": self._max_stored_conversations,
                "items": {
                    key: {
                        "file": entry.file,
                        "created_at": entry.created_at,
                        "touched_at": entry.touched_at,
                    }
                    for key, entry in self._conversations.items()
                },
            },
        }
        self._atomic_write(self.index_file, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        self._dirty = False

    def _touch(self, store: OrderedDict[str, _Entry], key: str) -> None:
        """Update the ``touched_at`` timestamp and move the entry to the end.

        Args:
            store: The ordered dict containing the entry.
            key: The key to touch.
        """
        entry = store.get(key)
        if entry is None:
            return
        entry.touched_at = time.time()
        store.move_to_end(key)
        self._dirty = True

    @staticmethod
    def _conversation_file_name(conversation_id: str) -> str:
        """Derive a deterministic filename from a conversation ID.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            A SHA-256-based filename with a ``.bin`` extension.
        """
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return f"{digest}.bin"

    def _evict(self) -> None:
        """Evict the least recently used entries that exceed capacity limits.

        Removes both the in-memory index entries and their backing
        files on disk for both responses and conversations.
        """
        if self._max_stored_responses == 0:
            for entry in self._responses.values():
                try:
                    (self.responses_dir / entry.file).unlink(missing_ok=True)
                except OSError:
                    pass
            self._responses.clear()
            self._dirty = True

        while self._max_stored_responses > 0 and len(self._responses) > self._max_stored_responses:
            _, entry = self._responses.popitem(last=False)
            try:
                (self.responses_dir / entry.file).unlink(missing_ok=True)
            except OSError:
                pass
            self._dirty = True

        if self._max_stored_conversations == 0:
            for entry in self._conversations.values():
                try:
                    (self.conversations_dir / entry.file).unlink(missing_ok=True)
                except OSError:
                    pass
            self._conversations.clear()
            self._dirty = True

        while self._max_stored_conversations > 0 and len(self._conversations) > self._max_stored_conversations:
            _, entry = self._conversations.popitem(last=False)
            try:
                (self.conversations_dir / entry.file).unlink(missing_ok=True)
            except OSError:
                pass
            self._dirty = True

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        """Retrieve a response payload by its ID.

        Args:
            response_id: The unique response identifier.

        Returns:
            The response payload dictionary, or ``None`` if not found
            or if the backing file is missing.
        """
        if not response_id:
            return None
        entry = self._responses.get(response_id)
        if entry is None:
            return None
        path = self.responses_dir / entry.file
        if not path.exists():
            self._responses.pop(response_id, None)
            self._dirty = True
            return None
        record = self._decode(path.read_bytes())
        self._touch(self._responses, response_id)
        self._flush_index()
        return record if isinstance(record, dict) else None

    def put_response(self, response_id: str, record: dict[str, Any]) -> None:
        """Store or update a response payload.

        Writes the payload to disk atomically and updates the LRU
        index.  May trigger eviction if the store is at capacity.

        Args:
            response_id: The unique response identifier.
            record: The response payload dictionary to store.
        """
        if not response_id or self._max_stored_responses == 0:
            return
        path = self.responses_dir / f"{response_id}.bin"
        self._atomic_write(path, self._encode(record))
        now = time.time()
        previous = self._responses.get(response_id)
        created_at = previous.created_at if previous is not None else now
        self._responses[response_id] = _Entry(file=path.name, created_at=created_at, touched_at=now)
        self._responses.move_to_end(response_id)
        self._dirty = True
        self._evict()
        self._flush_index()

    def delete_response(self, response_id: str) -> bool:
        """Delete a stored response by its ID.

        Args:
            response_id: The unique response identifier to delete.

        Returns:
            ``True`` if the response was found and deleted, ``False``
            otherwise.
        """
        if not response_id:
            return False
        entry = self._responses.pop(response_id, None)
        if entry is None:
            return False
        try:
            (self.responses_dir / entry.file).unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Failed to delete response record %s", response_id)
        self._dirty = True
        self._flush_index()
        return True

    def get_conversation(self, conversation_id: str) -> list[dict[str, Any]] | None:
        """Retrieve a conversation history by its ID.

        Args:
            conversation_id: The unique conversation identifier.

        Returns:
            A list of message dictionaries, or ``None`` if not found.
        """
        if not conversation_id:
            return None
        entry = self._conversations.get(conversation_id)
        if entry is None:
            return None
        path = self.conversations_dir / entry.file
        if not path.exists():
            self._conversations.pop(conversation_id, None)
            self._dirty = True
            return None
        record = self._decode(path.read_bytes())
        self._touch(self._conversations, conversation_id)
        self._flush_index()
        if not isinstance(record, dict):
            return None
        history = record.get("history")
        return history if isinstance(history, list) else None

    def put_conversation(self, conversation_id: str, history: list[dict[str, Any]]) -> None:
        """Store or update a conversation history.

        Args:
            conversation_id: The unique conversation identifier.
            history: List of message dictionaries to store.
        """
        if not conversation_id or self._max_stored_conversations == 0:
            return
        file_name = self._conversation_file_name(conversation_id)
        path = self.conversations_dir / file_name
        self._atomic_write(path, self._encode({"id": conversation_id, "history": history}))
        now = time.time()
        previous = self._conversations.get(conversation_id)
        created_at = previous.created_at if previous is not None else now
        self._conversations[conversation_id] = _Entry(file=file_name, created_at=created_at, touched_at=now)
        self._conversations.move_to_end(conversation_id)
        self._dirty = True
        self._evict()
        self._flush_index()

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
        entry = self._conversations.pop(conversation_id, None)
        if entry is None:
            return False
        try:
            (self.conversations_dir / entry.file).unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Failed to delete conversation record %s", conversation_id)
        self._dirty = True
        self._flush_index()
        return True

    def stats(self) -> dict[str, int]:
        """Return current storage statistics.

        Returns:
            A dictionary with keys ``"responses"``,
            ``"conversations"``, ``"max_stored_responses"``, and
            ``"max_stored_conversations"``.
        """
        return {
            "responses": len(self._responses),
            "conversations": len(self._conversations),
            "max_stored_responses": self._max_stored_responses,
            "max_stored_conversations": self._max_stored_conversations,
        }
