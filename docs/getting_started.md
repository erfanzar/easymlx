# Getting Started

This guide covers the basics of using EasyMLX for inference on Apple Silicon.

## Loading a Model

Load any supported HuggingFace model with `AutoEasyMLXModelForCausalLM`:

```python
from easymlx import AutoEasyMLXModelForCausalLM

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
)
```

## Basic Inference with eSurge

The `eSurge` engine provides paged attention and continuous batching:

```python
from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams

model = AutoEasyMLXModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

engine = eSurge(
    model,
    max_model_len=4096,
    max_num_seqs=4,
    max_num_batched_tokens=2048,
)

results = engine.generate(
    "Explain quantum computing in simple terms.",
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7, do_sample=True),
)

for output in results:
    print(output.text)
```

## Streaming Generation

Stream tokens as they are generated:

```python
for output in engine.stream(
    "Write a haiku about programming.",
    sampling_params=SamplingParams(max_tokens=64, temperature=0.8, do_sample=True),
):
    print(output.delta_text, end="", flush=True)
print()
```

## Chat with Messages

Use the `chat` method with a list of messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

result = engine.chat(
    messages,
    sampling_params=SamplingParams(max_tokens=128, temperature=0.7, do_sample=True),
)
print(result.text)
```

## Streaming Chat

```python
for output in engine.chat(
    messages,
    sampling_params=SamplingParams(max_tokens=128, temperature=0.7, do_sample=True),
    stream=True,
):
    print(output.delta_text, end="", flush=True)
print()
```

## Chat with Tool Calling

eSurge supports tool/function calling. Define tools as OpenAI-style dictionaries:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
]

result = engine.chat(
    [{"role": "user", "content": "What's the weather in Tokyo?"}],
    sampling_params=SamplingParams(max_tokens=256),
    tools=tools,
)

print(result.text)
if result.outputs[0].tool_calls:
    print("Tool calls:", result.outputs[0].tool_calls)
```

## One-Step Loading with eSurge.from_pretrained

For convenience, load and initialize in one call:

```python
from easymlx import eSurge, SamplingParams

engine = eSurge.from_pretrained(
    "Qwen/Qwen3-0.6B",
    max_model_len=4096,
    max_num_seqs=4,
)

results = engine.generate(
    "Hello, world!",
    sampling_params=SamplingParams(max_tokens=64),
)
print(results[0].text)
```

## Cleanup

When done, shut down the engine:

```python
engine.close()
```
