# Quantization



EasyMLX supports post-loading quantization via MLX's `nn.quantize`. Quantization reduces model memory usage and can improve inference throughput on Apple Silicon.



## Supported Modes



| Mode | Bits | Group Size | Description |

|------|------|------------|-------------|

| `affine` | 4 | 64 | Standard affine quantization. Works on all Apple Silicon. |

| `mxfp4` | 4 | 32 | Microscaling FP4 format. |

| `mxfp8` | 8 | 32 | Microscaling FP8 format. |

| `nvfp4` | 4 | 16 | NVIDIA FP4 format. |



## Usage



### String Shorthand



Pass the mode name as a string to `from_pretrained`:



```python

from easymlx import AutoEasyMLXModelForCausalLM



model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization="affine",

)

```



This applies 4-bit affine quantization with the default group size of 64.



### QuantizationConfig



For fine-grained control, pass a `QuantizationConfig` dictionary:



```python

from easymlx import AutoEasyMLXModelForCausalLM, QuantizationConfig



model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization=QuantizationConfig(mode="affine", bits=4, group_size=64),

)

```



### LayerwiseQuantizationConfig



For regex-based per-layer control, pass an ordered rule list. The first

matching rule wins, and ``config=None`` skips quantization for that

module path:



```python

from easymlx import (

    AutoEasyMLXModelForCausalLM,

    LayerwiseQuantizationConfig,

    QuantizationConfig,

    QuantizationRule,

)



model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

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



### Other Modes



```python



model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization="mxfp4",

)





model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization="mxfp8",

)





model = AutoEasyMLXModelForCausalLM.from_pretrained(

    "Qwen/Qwen3-0.6B",

    quantization="nvfp4",

)

```



## Default Parameters per Mode



Each mode has sensible defaults that are applied when you use the string shorthand:



| Mode | Default bits | Default group_size |

|------|-------------|-------------------|

| `affine` | 4 | 64 |

| `mxfp4` | 4 | 32 |

| `mxfp8` | 8 | 32 |

| `nvfp4` | 4 | 16 |



## Hardware Compatibility



- **`affine`** mode works on all Apple Silicon generations (M1 through M5).

- **`mxfp4`**, **`mxfp8`**, and **`nvfp4`** require specific Metal kernels that may not be available on older GPU architectures or older MLX versions. If a mode is not supported on your hardware, EasyMLX raises a `RuntimeError` with a suggestion to use `affine` instead or upgrade MLX.



If you encounter kernel errors:



```bash

pip install -U mlx

```



## Using Quantized Models with eSurge



Quantized models work seamlessly with eSurge:



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

    "Hello!",

    sampling_params=SamplingParams(max_tokens=64),

)

print(results[0].text)

```



## Performance Impact



Quantization trades off model quality for speed and memory:



- **4-bit affine** typically reduces memory by ~4x compared to float16 with minimal quality loss for most tasks.

- **mxfp4/nvfp4** can offer better quality than affine at the same bit width due to their floating-point representation.

- **mxfp8** provides a middle ground -- lower memory than float16 with quality close to the original.



## KV Cache Compression



EasyMLX also supports compressed paged KV caches independently from weight quantization:



```python

from easymlx.modules.llama import LlamaConfig



config = LlamaConfig(

    cache_dtype="turboquant",

    cache_bits=3,

)

```



- `cache_dtype="fp8"` stores paged KV caches in FP8 E4M3.

- `cache_dtype="turboquant"` enables TurboQuant KV compression for paged attention.

- `cache_bits` controls the TurboQuant rate and supports `2`, `3`, or `4` bits.



TurboQuant currently uses a correctness-first paged attention path in Python/NumPy rather than the Metal kernels, so it prioritizes memory reduction and integration over peak decode throughput.
