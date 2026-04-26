# eSurge Examples



Practical examples for common eSurge use cases.



## Chat Application



A simple multi-turn chat loop:



```python

from easymlx import eSurge, SamplingParams



engine = eSurge.from_pretrained(

    "Qwen/Qwen3-0.6B",

    max_model_len=4096,

    max_num_seqs=1,

)



messages = [{"role": "system", "content": "You are a helpful assistant."}]

params = SamplingParams(max_tokens=256, temperature=0.7, do_sample=True)



while True:

    user_input = input("You: ")

    if user_input.lower() in ("quit", "exit"):

        break



    messages.append({"role": "user", "content": user_input})

    result = engine.chat(messages, sampling_params=params)

    assistant_reply = result.text

    print(f"Assistant: {assistant_reply}")

    messages.append({"role": "assistant", "content": assistant_reply})



engine.close()

```



## Batch Processing



Process a list of prompts efficiently:



```python

from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams



model = AutoEasyMLXModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

engine = eSurge(model, max_model_len=4096, max_num_seqs=8)



prompts = [

    "Translate to French: Good morning",

    "Translate to French: How are you?",

    "Translate to French: Thank you very much",

    "Translate to French: See you tomorrow",

]



results = engine.generate(

    prompts,

    sampling_params=SamplingParams(max_tokens=64, temperature=0.3, do_sample=True),

)



for prompt, result in zip(prompts, results):

    print(f"{prompt}")

    print(f"  -> {result.text}\n")



engine.close()

```



## Streaming with TPS Metrics



Track tokens-per-second during streaming:



```python

import time

from easymlx import eSurge, SamplingParams



engine = eSurge.from_pretrained(

    "Qwen/Qwen3-0.6B",

    max_model_len=4096,

    max_num_seqs=1,

)



params = SamplingParams(max_tokens=512, temperature=0.7, do_sample=True)

prompt = "Write a short story about a robot learning to paint."



token_count = 0

start = time.perf_counter()



for output in engine.stream(prompt, sampling_params=params):

    print(output.delta_text, end="", flush=True)

    token_count += 1



elapsed = time.perf_counter() - start

print(f"\n\n--- {token_count} tokens in {elapsed:.2f}s ({token_count / elapsed:.1f} tok/s) ---")



engine.close()

```



## Using Quantization with eSurge



Load a quantized model and run inference:



```python

from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams





model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization="affine",

)



engine = eSurge(

    model,

    max_model_len=4096,

    max_num_seqs=4,

    memory_utilization=0.7,

)



results = engine.generate(

    "Explain the theory of relativity.",

    sampling_params=SamplingParams(max_tokens=256, temperature=0.7, do_sample=True),

)

print(results[0].text)



engine.close()

```



## Greedy Speculative Decoding



Use a smaller draft model to propose tokens that the target model verifies:



```python

from easymlx import eSurge, SamplingParams



engine = eSurge.from_pretrained(

    "Qwen/Qwen3-1.7B",

    speculative_model="Qwen/Qwen3-0.6B",

    speculative_method="draft",

    num_speculative_tokens=4,

    max_model_len=4096,

)



result = engine.generate(

    "Write a concise introduction to speculative decoding.",

    sampling_params=SamplingParams(max_tokens=192, do_sample=False),

)[0]

print(result.get_text())

print(result.metrics)



engine.close()

```

For EAGLE3, pass an adapter instead of a plain draft LM:

```python
engine = eSurge(
    target_model,
    tokenizer=tokenizer,
    speculative_model=eagle3_adapter,
    speculative_method="eagle3",
    num_speculative_tokens=4,
)
```

The adapter should implement `propose_eagle3(...)` or `propose(...)` and return proposed token ids. eSurge provides target hidden-state features as `hidden_states` and concatenated `eagle3_features`.



## Streaming Chat with Reasoning



For models that support chain-of-thought reasoning:



```python

from easymlx import eSurge, SamplingParams



engine = eSurge.from_pretrained(

    "Qwen/Qwen3-0.6B",

    max_model_len=4096,

    max_num_seqs=1,

)



messages = [{"role": "user", "content": "What is 15% of 240?"}]



for output in engine.chat(

    messages,

    sampling_params=SamplingParams(max_tokens=256, do_sample=True, temperature=0.7),

    stream=True,

):

    if output.delta_reasoning_content:

        print(f"[thinking] {output.delta_reasoning_content}", end="", flush=True)

    if output.delta_text:

        print(output.delta_text, end="", flush=True)

print()



engine.close()

```
