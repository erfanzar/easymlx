# EasyMLXBaseModule

`EasyMLXBaseModule` is the abstract base class for all EasyMLX neural network modules. It combines MLX's `nn.Module` with mixins for I/O, generation, and operation caching.

## Overview

```python
from easymlx import EasyMLXBaseModule, EasyMLXBaseConfig
```

All model implementations inherit from `EasyMLXBaseModule` and must override `__call__` for the forward pass.

## Inheritance Chain

```
mlx.nn.Module
    |
EasyMLXBaseModule
    + EasyBridgeMixin      (save/load/convert)
    + EasyGenerationMixin   (text generation)
    + OperationCacheMixin   (KV cache management)
```

## Key Attributes

`config`
: The active `EasyMLXBaseConfig` instance.

`config_class`
: The configuration class associated with this module. Defaults to `EasyMLXBaseConfig`.

`_model_task`
: String identifying the model task (e.g., `"causal-language-model"`). Set by the registry decorator.

`_model_type`
: String identifying the model type (e.g., `"llama"`). Set by the registry decorator.

## Loading Models

### from_pretrained

Load a model from a HuggingFace checkpoint or local directory:

```python
from easymlx import AutoEasyMLXModelForCausalLM

model = AutoEasyMLXModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=mx.float16,
)
```

The `from_pretrained` method handles:

1. Resolving the model path (local or HuggingFace Hub download).
2. Loading and converting HuggingFace weights to MLX format (with caching).
3. Instantiating the model with the loaded weights.

### save_pretrained

Save the model to a local directory:

```python
model.save_pretrained("/path/to/save")
```

## Implementing a New Module

```python
import mlx.core as mx
import mlx.nn as nn
from easymlx import EasyMLXBaseModule, EasyMLXBaseConfig

class MyModel(EasyMLXBaseModule):
    def __init__(self, config: EasyMLXBaseConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # ... define layers ...

    def __call__(self, input_ids: mx.array, **kwargs):
        x = self.embed(input_ids)
        # ... forward pass ...
        return x
```

See {doc}`adding_models` for a complete guide.
