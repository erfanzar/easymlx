# API Server



EasyMLX includes `eSurgeApiServer`, an OpenAI-compatible REST API server built on FastAPI.



## Starting the Server



```python

import uvicorn

from easymlx import AutoEasyMLXModelForCausalLM, eSurge, eSurgeApiServer



model = AutoEasyMLXModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

engine = eSurge(model, max_model_len=4096, max_num_seqs=8)



server = eSurgeApiServer(

    engines={"qwen3-0.6b": engine},

    title="My EasyMLX Server",

)



uvicorn.run(server.app, host="0.0.0.0", port=8000)

```



## Endpoints



The server exposes the following OpenAI-compatible endpoints:



| Endpoint | Method | Description |

|----------|--------|-------------|

| `/v1/chat/completions` | POST | Chat completions (streaming supported) |

| `/v1/completions` | POST | Text completions |

| `/v1/models` | GET | List available models |

| `/v1/responses` | POST | Stateless Responses API |

| `/health` | GET | Health check |

| `/metrics` | GET | Server metrics |



## Using with OpenAI Python Client



```python

from openai import OpenAI



client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")



response = client.chat.completions.create(

    model="qwen3-0.6b",

    messages=[

        {"role": "system", "content": "You are a helpful assistant."},

        {"role": "user", "content": "What is EasyMLX?"},

    ],

    max_tokens=256,

    temperature=0.7,

)



print(response.choices[0].message.content)

```



### Streaming



```python

stream = client.chat.completions.create(

    model="qwen3-0.6b",

    messages=[{"role": "user", "content": "Write a poem about code."}],

    max_tokens=256,

    stream=True,

)



for chunk in stream:

    if chunk.choices[0].delta.content:

        print(chunk.choices[0].delta.content, end="", flush=True)

print()

```



## curl Examples



### Chat Completion



```bash

curl http://localhost:8000/v1/chat/completions \

  -H "Content-Type: application/json" \

  -d '{

    "model": "qwen3-0.6b",

    "messages": [{"role": "user", "content": "Hello!"}],

    "max_tokens": 128

  }'

```



### Streaming Chat Completion



```bash

curl http://localhost:8000/v1/chat/completions \

  -H "Content-Type: application/json" \

  -d '{

    "model": "qwen3-0.6b",

    "messages": [{"role": "user", "content": "Tell me a joke."}],

    "max_tokens": 128,

    "stream": true

  }'

```



### List Models



```bash

curl http://localhost:8000/v1/models

```



### Health Check



```bash

curl http://localhost:8000/health

```



## API Key Management



Enable authentication by passing `require_api_key=True` and an admin key:



```python

server = eSurgeApiServer(

    engines={"qwen3-0.6b": engine},

    require_api_key=True,

    admin_api_key="my-admin-secret",

)

```



With API keys enabled:



- Requests must include an `Authorization: Bearer <key>` header.

- Admin endpoints under `/v1/admin/keys` allow creating, listing, and revoking API keys.

- Keys support rate limiting and usage quotas.



### Creating a Key (Admin)



```bash

curl -X POST http://localhost:8000/v1/admin/keys \

  -H "Authorization: Bearer my-admin-secret" \

  -H "Content-Type: application/json" \

  -d '{"name": "user-1"}'

```



## Multiple Models



Serve multiple models from a single server:



```python

model_a = AutoEasyMLXModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

model_b = AutoEasyMLXModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")



engine_a = eSurge(model_a, max_model_len=4096, max_num_seqs=4)

engine_b = eSurge(model_b, max_model_len=4096, max_num_seqs=4)



server = eSurgeApiServer(

    engines={

        "qwen3-0.6b": engine_a,

        "llama3-8b": engine_b,

    },

)

```



Clients select the model via the `model` field in the request body.



## Responses State

`/v1/responses` is stateless in the eSurge API server. Tool metadata is passed
through to the engine for parsing, but the server does not expose `/v1/tools`,
does not execute tools, and does not persist `previous_response_id` or
`conversation` state.
