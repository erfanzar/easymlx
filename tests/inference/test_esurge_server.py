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

"""Tests for the easymlx eSurge FastAPI server."""

import json

from fastapi.testclient import TestClient

from easymlx.inference.esurge import CompletionOutput, RequestOutput
from easymlx.inference.esurge.server.api_models import ChatCompletionRequest, CompletionRequest, ResponsesRequest
from easymlx.inference.esurge.server.api_server import eSurgeApiServer


class DummyEngine:
    def __init__(self) -> None:
        self.max_model_len = 64
        self.max_num_seqs = 1
        self.last_sampling_params = None

    def generate(self, prompts, sampling_params):
        self.last_sampling_params = sampling_params
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        return [
            RequestOutput(
                request_id="r1",
                prompt=prompt,
                prompt_token_ids=[],
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="hello",
                        token_ids=[1, 2],
                        finish_reason="tool_calls",
                        reasoning_content="plan",
                        tool_calls=[
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"city":"paris"}'},
                            }
                        ],
                    )
                ],
                finished=True,
                metrics={"num_generated_tokens": 2, "num_computed_tokens": len(prompt)},
                accumulated_text="hello",
                delta_text="hello",
                reasoning_content="plan",
                delta_reasoning_content="plan",
                tool_calls=[
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"paris"}'},
                    }
                ],
                delta_tool_calls=[
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"paris"}'},
                    }
                ],
            )
        ]

    def stream(self, prompts, sampling_params):
        yield from self.generate(prompts, sampling_params)

    def chat(self, messages, sampling_params=None, *, tools=None, **_kwargs):
        del messages, tools
        return self.generate(["chat"], sampling_params)[0]


def lookup(city: str) -> dict[str, str]:
    country = "France" if city.lower() == "paris" else "Unknown"
    return {"city": city.title(), "country": country}


def setup_client(*, tools=None, admin_api_key=None, require_api_key=False, response_store_dir=None):
    server = eSurgeApiServer(
        {"dummy": DummyEngine()},
        tools=tools,
        admin_api_key=admin_api_key,
        require_api_key=require_api_key,
        response_store_dir=str(response_store_dir) if response_store_dir is not None else None,
    )
    return TestClient(server.app)


def _route_exists(server: TestClient, *, path: str, method: str | None = None) -> bool:
    methods = {method.upper()} if method else None
    for route in server.app.router.routes:
        if getattr(route, "path", None) != path:
            continue
        if methods is None:
            return True
        if hasattr(route, "methods") and methods.issubset(route.methods):
            return True
    return False


def test_models_endpoint():
    client = setup_client()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert any(item["id"] == "dummy" for item in body["data"])


def test_health_and_model_detail_endpoints():
    client = setup_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    model = client.get("/v1/models/dummy")
    assert model.status_code == 200
    assert model.json()["id"] == "dummy"


def test_completion_endpoint():
    client = setup_client()
    resp = client.post("/v1/completions", json={"model": "dummy", "prompt": "Hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["text"] == "hello"


def test_server_request_models_accept_penalty_fields():
    completion = CompletionRequest(
        model="dummy",
        prompt="Hi",
        presence_penalty=0.25,
        repetition_penalty=1.4,
    )
    chat = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "Hi"}],
        presence_penalty=0.5,
        repetition_penalty=1.2,
    )
    responses = ResponsesRequest(
        model="dummy",
        input="Hi",
        presence_penalty=0.75,
        repetition_penalty=1.1,
    )

    assert completion.presence_penalty == 0.25
    assert completion.repetition_penalty == 1.4
    assert chat.presence_penalty == 0.5
    assert chat.repetition_penalty == 1.2
    assert responses.presence_penalty == 0.75
    assert responses.repetition_penalty == 1.1


def test_completion_endpoint_passes_penalties_to_sampling_params():
    engine = DummyEngine()
    server = eSurgeApiServer({"dummy": engine})
    client = TestClient(server.app)

    resp = client.post(
        "/v1/completions",
        json={
            "model": "dummy",
            "prompt": "Hi",
            "presence_penalty": 0.6,
            "repetition_penalty": 1.4,
        },
    )

    assert resp.status_code == 200
    assert engine.last_sampling_params is not None
    assert engine.last_sampling_params.presence_penalty == 0.6
    assert engine.last_sampling_params.repetition_penalty == 1.4


def test_chat_completion_endpoint():
    client = setup_client()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object"}}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["message"]["reasoning_content"] == "plan"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"


def test_responses_endpoint():
    client = setup_client()
    resp = client.post("/v1/responses", json={"model": "dummy", "input": "Hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    output_text_items = [item for item in body["output"][0]["content"] if item["type"] == "output_text"]
    assert output_text_items[0]["text"] == "hello"


def test_responses_endpoint_supports_previous_response_id():
    client = setup_client()
    first = client.post("/v1/responses", json={"model": "dummy", "input": "Hi"})
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        "/v1/responses",
        json={"model": "dummy", "input": "Again", "previous_response_id": first_body["id"]},
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["previous_response_id"] == first_body["id"]
    assert second_body["store"] is True


def test_responses_streaming_emits_response_events():
    client = setup_client()
    with client.stream("POST", "/v1/responses", json={"model": "dummy", "input": "Hi", "stream": True}) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    event_types = {event["type"] for event in events}
    assert "response.created" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.completed" in event_types


def test_streaming_completion():
    client = setup_client()
    with client.stream("POST", "/v1/completions", json={"model": "dummy", "prompt": "Hi", "stream": True}) as resp:
        assert resp.status_code == 200
        data = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data.append(line[6:])
    assert data and data[-1] == "[DONE]"


def test_streaming_chat_completion():
    client = setup_client()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "dummy", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        data = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]
    assert data and data[-1] == "[DONE]"


def test_metrics_route_is_registered_and_usable():
    client = setup_client()
    assert _route_exists(client, path="/metrics", method="get")

    before = client.get("/metrics")
    assert before.status_code == 200
    initial = before.json()

    completion = client.post("/v1/completions", json={"model": "dummy", "prompt": "Hi"})
    assert completion.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models_loaded"] == 1
    assert payload["status"] == "ready"
    assert payload["total_requests"] >= initial["total_requests"] + 1
    assert payload["successful_requests"] >= initial["successful_requests"] + 1
    assert payload["total_tokens_generated"] >= initial["total_tokens_generated"] + 2


def test_tools_routes_are_registered():
    client = setup_client()
    assert _route_exists(client, path="/v1/tools", method="get")
    assert _route_exists(client, path="/v1/tools/execute", method="post")


def test_tools_execute_endpoint_accepts_payload():
    client = setup_client(
        tools={
            "lookup": {
                "handler": lookup,
                "description": "Find a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        }
    )

    response = client.post(
        "/v1/tools/execute",
        json={
            "model": "dummy",
            "tool_call": {
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"city":"paris"}'},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["name"] == "lookup"
    assert payload["output"] == {"city": "Paris", "country": "France"}


def test_tools_execute_endpoint_rejects_bad_json():
    client = setup_client(tools={"lookup": lookup})
    response = client.post("/v1/tools/execute", json={"name": "lookup", "arguments": "{bad json"})
    assert response.status_code == 400
    assert "Invalid tool argument JSON" in response.json()["detail"]


def test_tools_list_route_returns_list_payload():
    client = setup_client(
        tools={
            "lookup": {
                "handler": lookup,
                "description": "Find a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        }
    )
    response = client.get("/v1/tools")
    assert response.status_code == 200
    payload = response.json()

    tool_payload = payload.get("tools")
    assert isinstance(tool_payload, list)
    assert tool_payload[0]["function"]["name"] == "lookup"
    assert payload["models"]["dummy"]["tools"][0]["function"]["name"] == "lookup"


def test_admin_routes_are_registered():
    client = setup_client()
    for endpoint in ("/v1/admin/keys", "/v1/admin/keys/stats", "/v1/admin/audit-logs"):
        assert _route_exists(client, path=endpoint)


def test_admin_routes_require_key_when_configured():
    client = setup_client(admin_api_key="bootstrap-secret")
    response = client.get("/v1/admin/keys")
    assert response.status_code == 401


def test_admin_key_lifecycle():
    client = setup_client(admin_api_key="bootstrap-secret")
    headers = {"Authorization": "Bearer bootstrap-secret"}

    created = client.post("/v1/admin/keys", headers=headers, json={"name": "ops", "role": "admin"})
    assert created.status_code == 201
    created_payload = created.json()
    assert created_payload["name"] == "ops"
    assert created_payload["role"] == "admin"
    assert created_payload["key"].startswith("sk-emx-")

    listed = client.get("/v1/admin/keys", headers=headers)
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload["total"] >= 1
    key_id = created_payload["key_id"]

    fetched = client.get(f"/v1/admin/keys/{key_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["key_id"] == key_id

    rotated = client.post(f"/v1/admin/keys/{key_id}/rotate", headers=headers)
    assert rotated.status_code == 200
    rotated_payload = rotated.json()
    assert rotated_payload["key_id"] == key_id
    assert rotated_payload["key"] != created_payload["key"]

    revoked = client.delete(f"/v1/admin/keys/{key_id}/revoke", headers=headers)
    assert revoked.status_code == 200
    assert revoked.json()["key_id"] == key_id

    stats = client.get("/v1/admin/keys/stats", headers=headers)
    assert stats.status_code == 200
    stats_payload = stats.json()
    assert stats_payload["total"] >= 1
    assert stats_payload["statuses"]["revoked"] >= 1

    audit = client.get("/v1/admin/audit-logs", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["total"] >= 3


def test_inference_routes_require_api_key_when_enabled():
    client = setup_client(admin_api_key="bootstrap-secret", require_api_key=True)
    response = client.post("/v1/completions", json={"model": "dummy", "prompt": "Hi"})
    assert response.status_code == 401


def test_inference_api_key_enforces_streaming_and_function_limits():
    client = setup_client(admin_api_key="bootstrap-secret", require_api_key=True)
    admin_headers = {"Authorization": "Bearer bootstrap-secret"}
    created = client.post(
        "/v1/admin/keys",
        headers=admin_headers,
        json={
            "name": "limited-user",
            "role": "user",
            "allowed_models": ["dummy"],
            "allowed_endpoints": ["/v1/completions", "/v1/chat/completions", "/v1/responses"],
            "enable_streaming": False,
            "enable_function_calling": False,
            "max_tokens_per_request": 2,
        },
    )
    assert created.status_code == 201
    api_key = created.json()["key"]
    user_headers = {"Authorization": f"Bearer {api_key}"}

    completion = client.post(
        "/v1/completions",
        headers=user_headers,
        json={"model": "dummy", "prompt": "Hi", "max_tokens": 2},
    )
    assert completion.status_code == 200

    streaming = client.post(
        "/v1/completions",
        headers=user_headers,
        json={"model": "dummy", "prompt": "Hi", "max_tokens": 2, "stream": True},
    )
    assert streaming.status_code == 403
    assert "Streaming is disabled" in streaming.json()["detail"]

    tool_chat = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 2,
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )
    assert tool_chat.status_code == 403
    assert "Function calling is disabled" in tool_chat.json()["detail"]


def test_inference_api_key_rejects_excessive_token_request():
    client = setup_client(admin_api_key="bootstrap-secret", require_api_key=True)
    admin_headers = {"Authorization": "Bearer bootstrap-secret"}
    created = client.post(
        "/v1/admin/keys",
        headers=admin_headers,
        json={
            "name": "short-user",
            "role": "user",
            "allowed_models": ["dummy"],
            "max_tokens_per_request": 1,
        },
    )
    assert created.status_code == 201
    api_key = created.json()["key"]
    response = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "dummy", "prompt": "Hi", "max_tokens": 4},
    )
    assert response.status_code == 403
    assert "max_tokens_per_request" in response.json()["detail"]
