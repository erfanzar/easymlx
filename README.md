<!-- markdownlint-disable MD033 MD045 MD041 -->
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
  <a href="#api-server">API Server</a>
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

# 4-bit affine quantization (works on all Apple Silicon)
model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization="affine",  # 4-bit, group_size=64
)

# Or with explicit config
from easymlx import QuantizationConfig

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization=QuantizationConfig(mode="affine", bits=4, group_size=64),
)

# Or with ordered regex rules
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
    max_model_len=4096,          # Maximum sequence length
    max_num_seqs=4,              # Maximum concurrent sequences
    max_num_batched_tokens=2048, # Token budget per step
    page_size=64,                # KV cache page size
    memory_utilization=0.85,     # Fraction of GPU memory for KV cache
    runner_verbose=True,         # Show warmup progress and step logs
    tool_parser="llama3_json",   # Auto-detected if not set
    reasoning_parser="auto",     # Auto-detected if not set
)
```

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
# output.tool_calls contains parsed tool invocations
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
easymlx/
├── caching/           # KV cache implementations
│   ├── paged/         #   PageCacheView, PageCacheView, PageMetadata
│   ├── transformer/   #   TransformerCache (dense attention)
│   ├── recurrent/     #   RecurrentCache (Mamba/SSM)
│   └── hybrid/        #   HybridCache (mixed attention + SSM)
├── inference/
│   └── esurge/        # eSurge inference engine
│       ├── engine.py          # Core engine with paged runtime
│       ├── runners/           # Model runner, execution manager
│       ├── scheduler/         # Continuous batching scheduler
│       ├── server/            # OpenAI-compatible API server
│       ├── mixins/            # Chat, lifecycle, monitoring mixins
│       └── distributed/       # Multi-worker support
├── infra/             # Base config, module, factory, bridge
├── layers/            # Attention, RoPE, MoE, embeddings, linears
├── modules/           # Model implementations
│   ├── _base/         #   Task-specific base classes (CausalLM, VLM, etc.)
│   ├── auto/          #   Auto classes (from_pretrained)
│   └── <model>/       #   Per-model config + modeling
├── operations/        # Attention kernels (SDPA, paged, vanilla)
└── workers/           # Server auth, logging, response store
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
