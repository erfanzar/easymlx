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

"""Tests for the EasyMLX browser app server."""

from __future__ import annotations

import time
from typing import Any, ClassVar

from easymlx.app.server import EasyMLXAppServer
from fastapi.testclient import TestClient


class FakeEngine:
    closed = 0
    tokenizer = type("FakeTokenizer", (), {"name_or_path": "Qwen/Qwen3.5-9B", "chat_template": "{{ messages }}"})()

    def close(self) -> None:
        FakeEngine.closed += 1


class FakeESurge:
    calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeEngine:
        cls.calls.append({"args": args, "kwargs": kwargs})
        return FakeEngine()


def wait_for_model(client: TestClient, served_name: str, status: str = "ready") -> dict[str, Any]:
    deadline = time.time() + 2
    while time.time() < deadline:
        payload = client.get("/app/api/models").json()
        for model in payload["data"]:
            if model["served_name"] == served_name and model["status"] == status:
                return model
        time.sleep(0.02)
    raise AssertionError(f"Model {served_name!r} did not reach {status!r}")


def test_app_serves_index_and_config_schema():
    server = EasyMLXAppServer()
    client = TestClient(server.app)

    index = client.get("/")
    assert index.status_code == 200
    assert "EasyMLX" in index.text

    schema = client.get("/app/api/config-schema")
    assert schema.status_code == 200
    engine_fields = schema.json()["engine"]
    sampling_fields = schema.json()["sampling"]
    fields = {field["name"] for field in engine_fields}
    defaults = {field["name"]: field["default"] for field in engine_fields}
    placeholders = {field["name"]: field["placeholder"] for field in engine_fields}
    choices = {field["name"]: field["choices"] for field in engine_fields}
    sampling_defaults = {field["name"]: field["default"] for field in sampling_fields}
    categories = {field["category"] for field in engine_fields}
    assert "max_model_len" in fields
    assert "reasoning_parser" in fields
    assert "speculative_model" in fields
    assert "num_speculative_tokens" in fields
    assert "speculative_model_kwargs" in fields
    assert defaults["max_model_len"] == 4096
    assert defaults["max_num_seqs"] == 1
    assert defaults["max_num_batched_tokens"] == 4096
    assert defaults["hbm_utilization"] == 0.45
    assert defaults["page_size"] == 128
    assert defaults["num_speculative_tokens"] == 0
    assert defaults["speculative_method"] == "dflash"
    assert choices["speculative_method"] == ["dflash", "draft", "eagle3"]
    assert placeholders["speculative_model"] == "z-lab/Qwen3.5-9B-DFlash or /path/to/dflash-draft"
    assert sampling_defaults["max_tokens"] == 4096
    assert "Speculative" in categories
    assert "auto_shard_model" not in fields
    assert "distributed_mode" not in fields
    assert "Sharding" not in categories
    assert "Distributed" not in categories


def test_model_load_registers_openai_model_and_unload_closes_engine(monkeypatch):
    import easymlx.app.server as app_server

    FakeEngine.closed = 0
    FakeESurge.calls.clear()
    monkeypatch.setattr(app_server, "eSurge", FakeESurge)

    server = EasyMLXAppServer()
    client = TestClient(server.app)

    created = client.post(
        "/app/api/models/load",
        json={
            "model_id": "Qwen/Qwen3.5-9B",
            "served_name": "qwen3",
            "tokenizer": "Qwen/Qwen3.5-9B",
            "model_kwargs": {"quantization": {"mode": "affine", "bits": 4, "group_size": 64}},
            "engine_kwargs": {
                "max_model_len": 1024,
                "page_size": 64,
                "speculative_model": "z-lab/Qwen3.5-9B-DFlash",
                "num_speculative_tokens": 3,
                "speculative_model_kwargs": {"quantization": "mxfp4"},
            },
        },
    )
    assert created.status_code == 202

    model = wait_for_model(client, "qwen3")
    assert model["status"] == "ready"
    assert model["tokenizer_source"] == "Qwen/Qwen3.5-9B"
    assert model["has_chat_template"] is True
    assert FakeESurge.calls[0]["args"] == ("Qwen/Qwen3.5-9B",)
    assert FakeESurge.calls[0]["kwargs"]["tokenizer"] == "Qwen/Qwen3.5-9B"
    assert FakeESurge.calls[0]["kwargs"]["model_kwargs"]["quantization"]["mode"] == "affine"
    assert FakeESurge.calls[0]["kwargs"]["max_model_len"] == 1024
    assert FakeESurge.calls[0]["kwargs"]["speculative_model"] == "z-lab/Qwen3.5-9B-DFlash"
    assert FakeESurge.calls[0]["kwargs"]["num_speculative_tokens"] == 3
    assert FakeESurge.calls[0]["kwargs"]["speculative_model_kwargs"]["quantization"] == "mxfp4"

    listed = client.get("/v1/models")
    assert listed.status_code == 200
    assert any(item["id"] == "qwen3" for item in listed.json()["data"])

    duplicate = client.post("/app/api/models/load", json={"model_id": "Qwen/Qwen3.5-9B", "served_name": "qwen3"})
    assert duplicate.status_code == 409

    unloaded = client.post("/app/api/models/qwen3/unload", json={})
    assert unloaded.status_code == 200
    assert FakeEngine.closed == 1
    assert not client.get("/app/api/models").json()["data"]
