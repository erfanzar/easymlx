# eSurge Inference Engine

eSurge is the core inference engine of EasyMLX. It provides paged attention, continuous batching, streaming generation, and tool calling -- all optimized for Apple Silicon via MLX.

## Overview

The eSurge engine manages:

- **Paged KV Cache** -- Memory-efficient key-value cache with page-based allocation and optional prefix caching.
- **Continuous Batching** -- A background scheduler that dynamically batches incoming requests for maximum throughput.
- **Streaming** -- Token-by-token streaming with delta text output.
- **Tool/Function Calling** -- Automatic detection and parsing of tool calls for supported model families.
- **Reasoning Extraction** -- Automatic parsing of chain-of-thought reasoning content.

## Quick Start

```python
from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams

model = AutoEasyMLXModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

engine = eSurge(
    model,
    max_model_len=4096,
    max_num_seqs=4,
    max_num_batched_tokens=2048,
    page_size=16,
    memory_utilization=0.6,
)

results = engine.generate(
    "What is machine learning?",
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7, do_sample=True),
)
print(results[0].text)
```

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | model instance | required | An EasyMLX model with paged attention support |
| `tokenizer` | str or tokenizer | `None` | Tokenizer instance or HF model ID. Auto-resolved if `None` |
| `max_model_len` | int | `8192` | Maximum total sequence length (prompt + generation) |
| `max_num_seqs` | int | `4` | Maximum concurrent sequences |
| `max_num_batched_tokens` | int | `2048` | Maximum tokens processed per scheduler step |
| `page_size` | int | `16` | Tokens per KV cache page |
| `memory_utilization` | float | `0.6` | Fraction of available memory for caches |
| `enable_prefix_caching` | bool | `True` | Share KV cache pages for common prefixes |
| `runner_verbose` | bool | `False` | Enable verbose logging in the model runner |
| `tool_parser` | str or None | `None` | Tool parser name (auto-detected if `None`) |
| `reasoning_parser` | str or None | `None` | Reasoning parser name (auto-detected if `None`) |
| `seed` | int | `0` | Random seed for reproducible sampling |

### SamplingParams

Control generation behavior:

```python
from easymlx import SamplingParams

params = SamplingParams(
    max_tokens=256,       # Maximum tokens to generate
    temperature=0.7,      # Sampling temperature
    top_k=50,             # Top-k filtering (0 = disabled)
    top_p=0.9,            # Nucleus sampling threshold (1.0 = disabled)
    do_sample=True,       # Stochastic sampling (False = greedy)
    stop=["<END>"],       # Stop strings
    n=1,                  # Number of completions per request
    tools=None,           # Tool definitions for function calling
    tool_choice="auto",   # Tool selection mode
)
```

## Streaming Generation

Use `stream()` for token-by-token output:

```python
for output in engine.stream(
    "Explain relativity.",
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7, do_sample=True),
):
    print(output.delta_text, end="", flush=True)
print()
```

Or use `chat(..., stream=True)` for chat-style streaming:

```python
for output in engine.chat(
    [{"role": "user", "content": "Hello!"}],
    sampling_params=SamplingParams(max_tokens=128),
    stream=True,
):
    print(output.delta_text, end="", flush=True)
```

## Batch Processing

Pass multiple prompts to `generate()`:

```python
prompts = [
    "Summarize quantum computing.",
    "Explain neural networks.",
    "What is reinforcement learning?",
]

results = engine.generate(
    prompts,
    sampling_params=SamplingParams(max_tokens=128, temperature=0.7, do_sample=True),
)

for i, result in enumerate(results):
    print(f"--- Prompt {i} ---")
    print(result.text)
```

## Tool Calling

eSurge supports tool/function calling with automatic parser detection for many model families (Qwen, Llama, Hermes, Mistral, DeepSeek, etc.).

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }
]

result = engine.chat(
    [{"role": "user", "content": "Search for the latest MLX release."}],
    sampling_params=SamplingParams(max_tokens=256),
    tools=tools,
)

if result.outputs[0].tool_calls:
    for tc in result.outputs[0].tool_calls:
        print(f"Function: {tc['function']['name']}")
        print(f"Arguments: {tc['function']['arguments']}")
```

You can also pass `tool_parser="hermes"` (or another parser name) to the eSurge constructor to override auto-detection.

## Reasoning Parser Support

For models that produce chain-of-thought reasoning (e.g., Qwen3, DeepSeek), eSurge automatically extracts reasoning content:

```python
result = engine.chat(
    [{"role": "user", "content": "Solve: 2x + 3 = 11"}],
    sampling_params=SamplingParams(max_tokens=512, do_sample=True, temperature=0.7),
)

if result.outputs[0].reasoning_content:
    print("Reasoning:", result.outputs[0].reasoning_content)
print("Answer:", result.text)
```

## mx.compile Warmup

The model runner calls `precompile()` during initialization to warm up `mx.compile` traced functions. This incurs a one-time cost at startup but speeds up subsequent inference steps.

## Performance Tuning

### Page Size

Larger page sizes reduce page table overhead but waste memory for short sequences. Smaller page sizes improve utilization for variable-length workloads.

- **Default**: `16` tokens per page
- **Short sequences**: Try `page_size=8`
- **Long sequences**: Try `page_size=32` or `page_size=64`

### Memory Utilization

Controls how much of the available unified memory is allocated for KV caches:

- **Default**: `0.6` (60%)
- **Conservative**: `0.4` -- leaves more room for the OS and other apps
- **Aggressive**: `0.8` -- more cache capacity, risk of OOM

### Quantization

Load quantized models to reduce memory and increase throughput:

```python
from easymlx import AutoEasyMLXModelForCausalLM

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization="affine",  # 4-bit affine quantization
)
```

See {doc}`quantization` for details.

### Batching

- `max_num_seqs` -- Controls concurrency. Higher values serve more users simultaneously but require more cache memory.
- `max_num_batched_tokens` -- Limits the total tokens per step. Lower values reduce latency per step; higher values improve throughput.

## Troubleshooting

**"model does not support paged attention"**
: The model must expose `init_operations_cache` or `init_paged_cache`. Ensure you are using an EasyMLX model class.

**OOM / slow inference**
: Lower `memory_utilization`, reduce `max_num_seqs`, or reduce `max_model_len`. Large page counts (> 8192 pages) can degrade performance on Apple Silicon.

**Tool calls not detected**
: Set `tool_parser` explicitly if auto-detection fails (e.g., `tool_parser="hermes"`).

**Slow first request**
: The first inference step includes `mx.compile` warmup. Subsequent requests will be faster.
