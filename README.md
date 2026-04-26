
<p align="center">
  <img src="https://raw.githubusercontent.com/erfanzar/easydel/main/images/easydel-logo-with-text.png" alt="EasyMLX Logo" width="300"/>
</p>

<h1 align="center">EasyMLX</h1>

<p align="center">
  <em>Inference-only port of <a href="https://github.com/erfanzar/EasyDeL">EasyDeL</a> for Apple Silicon via <a href="https://github.com/ml-explore/mlx">MLX</a></em>
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#supported-models">Models</a> &bull;
  <a href="#esurge-engine">eSurge</a> &bull;
  <a href="#quantization">Quantization</a> &bull;
  <a href="#api-server">API Server</a> &bull;
  <a href="#local-app">Local App</a>
</p>

---

EasyMLX brings the full EasyDeL inference stack to Apple Silicon. It provides paged attention, continuous batching, streaming, tool calling, reasoning parsers, and an OpenAI-compatible API server — all running natively on Metal via MLX.

## Installation

```bash
pip install easymlx
```

Or from source:

```bash
git clone https://github.com/erfanzar/easymlx.git
cd easymlx
pip install -e .
```

**Requirements:** Python 3.13+, macOS with Apple Silicon, MLX >= 0.31.1

## Quick Start

### Basic Inference

```python
from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
)

engine = eSurge(model, tokenizer="meta-llama/Llama-3.2-1B-Instruct")

output = engine.chat(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7),
)
print(output.accumulated_text)
```

### Streaming

```python
for chunk in engine.chat(
    messages=[{"role": "user", "content": "Write a haiku about code"}],
    sampling_params=SamplingParams(max_tokens=64),
    stream=True,
):
    print(chunk.delta_text, end="", flush=True)
```

### Direct Generation

```python
outputs = engine.generate(
    "Once upon a time",
    sampling_params=SamplingParams(max_tokens=128, temperature=0.9, top_p=0.95),
)
print(outputs[0].accumulated_text)
```

## Quantization

Quantize models at load time for faster inference and lower memory:

```python
from mlx import core as mx

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization="affine",
)

from easymlx import QuantizationConfig

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization=QuantizationConfig(mode="affine", bits=4, group_size=64),
)

from easymlx import LayerwiseQuantizationConfig, QuantizationRule

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization=LayerwiseQuantizationConfig(
        default=QuantizationConfig(mode="affine", bits=4, group_size=64),
        rules=[
            QuantizationRule(
                pattern=r"^model\\.embed_tokens$",
                config=QuantizationConfig(mode="affine", bits=8, group_size=64),
            ),
            QuantizationRule(pattern=r"^lm_head$", config=None),
        ],
    ),
)
```

**Supported modes:**

| Mode     | Bits | Group Size | Notes                              |
| -------- | ---- | ---------- | ---------------------------------- |
| `affine` | 4    | 64         | Works on all Apple Silicon         |
| `mxfp4`  | 4    | 32         | Requires MLX with GPU arch support |
| `mxfp8`  | 8    | 32         | Requires MLX with GPU arch support |
| `nvfp4`  | 4    | 16         | Requires MLX with GPU arch support |

## eSurge Engine

eSurge is the high-performance inference engine, ported from EasyDeL. It provides:

- **Paged Attention** — efficient KV cache management with block-level allocation
- **Continuous Batching** — dynamic request scheduling with configurable sequence budgets
- **`mx.compile` Warmup** — pre-traces model forward for all token/batch buckets
- **Tool Calling** — built-in parsers for Llama 3, Hermes, Mistral, Qwen, and more
- **Reasoning Parsers** — DeepSeek R1, Qwen3, and other chain-of-thought extractors
- **Streaming** — token-by-token output with delta text and TPS metrics

### Engine Configuration

```python
engine = eSurge(
    model,
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
    max_model_len=4096,
    max_num_seqs=1,
    max_num_batched_tokens=4096,
    page_size=128,
    hbm_utilization=0.45,
    runner_verbose=True,
    tool_parser="llama3_json",
    reasoning_parser="auto",
    # speculative_model=draft_model,  # optional smaller same-tokenizer model
    # speculative_method="draft",      # or "eagle3" with an EAGLE3 adapter
    # num_speculative_tokens=4,
)
```

`speculative_model` can be a loaded draft model, model id/path, or an EAGLE3 adapter when `speculative_method="eagle3"`. Speculative decoding is used for verified greedy single-prompt generation (`do_sample=False`); unsupported request shapes fall back to the normal eSurge scheduler. EAGLE3 adapters must expose a proposal method such as `propose_eagle3(...)` or `propose(...)`, and the target model must expose hidden-state features via `eagle3_hidden_states(...)` or `output_hidden_states=True`.

### Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

output = engine.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    sampling_params=SamplingParams(max_tokens=256),
    tools=tools,
)
```

## API Server

Launch an OpenAI-compatible HTTP server:

```python
from easymlx.inference.esurge.server import eSurgeApiServer

server = eSurgeApiServer(engine)
server.run(host="0.0.0.0", port=8000)
```

Then use any OpenAI-compatible client:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## Local App

Run it as a desktop app from the TypeScript workspace:

```bash
cd lib/typescript/easymlx-app
bun install
bun run app
```

The desktop app starts the Python backend on `http://127.0.0.1:8719` and opens the EasyMLX UI in an Electron window. Set `EASYMLX_DESKTOP_PORT=9000` if you want a different backend port.

Run the EasyMLX control app with one command:

```bash
easymlx app --dev
```

From a source checkout without installing the package first:

```bash
PYTHONPATH=lib/python python -m easymlx.cli app --dev
```

This starts the Python backend on `http://127.0.0.1:8000` and the Bun/Vite frontend on `http://127.0.0.1:5173`. The frontend proxies `/app/api`, `/v1`, `/health`, and `/metrics` to the backend.
Per-request access logs are off by default for the app; add `--access-log` when you want to inspect every HTTP request.

## Supported Models

### Language Models

| Family             | Variants                       | Model Types     |
| ------------------ | ------------------------------ | --------------- |
| **Llama**          | Llama 2/3/3.1/3.2              | `llama`         |
| **Llama 4**        | Llama 4 Scout/Maverick         | `llama4`        |
| **Qwen**           | Qwen 1.5                       | `qwen`          |
| **Qwen 2**         | Qwen 2/2.5                     | `qwen2`         |
| **Qwen 2 MoE**     | Qwen 2 MoE                     | `qwen2_moe`     |
| **Qwen 3**         | Qwen 3                         | `qwen3`         |
| **Qwen 3 MoE**     | Qwen 3 MoE                     | `qwen3_moe`     |
| **Qwen 3 Next**    | Qwen 3 Next (hybrid attention) | `qwen3_next`    |
| **GLM**            | GLM-4                          | `glm`           |
| **GLM-4**          | GLM-4                          | `glm4`          |
| **GLM-4 MoE**      | GLM-4 MoE                      | `glm4_moe`      |
| **GLM-4 MoE Lite** | GLM-4 MoE Lite (MLA)           | `glm4_moe_lite` |
| **GPT-OSS**        | SeedLM / GPT-OSS               | `gpt_oss`       |

### Vision-Language Models

| Family              | Model Types      |
| ------------------- | ---------------- |
| **Qwen 2 VL**       | `qwen2_vl`       |
| **Qwen 3 VL**       | `qwen3_vl`       |
| **Qwen 3 VL MoE**   | `qwen3_vl_moe`   |
| **Qwen 3 Omni MoE** | `qwen3_omni_moe` |
| **GLM-4V**          | `glm4v`          |
| **GLM-4V MoE**      | `glm4v_moe`      |
| **GLM-4.6V**        | `glm46v`         |
| **Llama 4**         | `llama4`         |

## Architecture

EasyMLX mirrors EasyDeL's architecture, adapted for MLX:

```md
lib/
├── python/
│   └── easymlx/
│       ├── app/               # Local FastAPI app and built web assets
│       ├── caching/           # KV cache implementations
│       ├── inference/
│       │   └── esurge/        # eSurge inference engine and API server
│       ├── infra/             # Base config, module, factory, bridge
│       ├── layers/            # Attention, RoPE, MoE, embeddings, linears
│       ├── modules/           # Model implementations
│       ├── operations/        # Attention kernels (SDPA, paged, vanilla)
│       └── workers/           # Server auth, logging, response store
└── typescript/
    └── easymlx-app/           # Bun + React + TypeScript + Electron app
```

### Key Differences from EasyDeL

|                 | EasyDeL                       | EasyMLX                         |
| --------------- | ----------------------------- | ------------------------------- |
| **Backend**     | JAX/Flax on TPU/GPU           | MLX on Apple Silicon Metal      |
| **Arrays**      | `jax.Array` (immutable)       | `mx.array` (mutable, eager)     |
| **Sharding**    | Automatic mesh parallelism    | None (single-device)            |
| **Training**    | Full trainer suite            | Inference only                  |
| **State**       | `EasyDeLState` checkpoints    | No state management             |
| **Attention**   | Multiple kernels + Flash/Ring | Paged + SDPA + vanilla          |
| **Compilation** | `jax.jit`                     | `mx.compile` with bucket warmup |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 The EASYDEL / EASYMLX Author [@erfanzar](https://github.com/erfanzar) (Erfan Zare Chavoshi).
