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

"""Tests for easymlx response-store implementations."""

from easymlx.workers.response_store import FileResponseStore, InMemoryResponseStore


def test_in_memory_response_store_roundtrip_and_lru() -> None:
    store = InMemoryResponseStore(max_stored_responses=2, max_stored_conversations=1)

    store.put_response("resp_1", {"id": "resp_1", "value": 1})
    store.put_response("resp_2", {"id": "resp_2", "value": 2})
    assert store.get_response("resp_1") == {"id": "resp_1", "value": 1}

    store.put_response("resp_3", {"id": "resp_3", "value": 3})
    assert store.get_response("resp_2") is None
    assert store.get_response("resp_3") == {"id": "resp_3", "value": 3}

    store.put_conversation("conv_1", [{"role": "user", "content": "Hi"}])
    store.put_conversation("conv_2", [{"role": "assistant", "content": "Hello"}])
    assert store.get_conversation("conv_1") is None
    assert store.get_conversation("conv_2") == [{"role": "assistant", "content": "Hello"}]
    assert store.stats()["responses"] == 2


def test_file_response_store_roundtrip(tmp_path) -> None:
    store = FileResponseStore(
        tmp_path / "response-store",
        max_stored_responses=4,
        max_stored_conversations=4,
    )
    store.put_response("resp_1", {"id": "resp_1", "value": 1})
    store.put_conversation("conv_1", [{"role": "user", "content": "Hi"}])

    assert store.get_response("resp_1") == {"id": "resp_1", "value": 1}
    assert store.get_conversation("conv_1") == [{"role": "user", "content": "Hi"}]

    store.delete_response("resp_1")
    store.delete_conversation("conv_1")
    assert store.get_response("resp_1") is None
    assert store.get_conversation("conv_1") is None
