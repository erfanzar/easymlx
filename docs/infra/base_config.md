# EasyMLXBaseConfig

`EasyMLXBaseConfig` is the base configuration class for all EasyMLX models. It extends HuggingFace's `PretrainedConfig` to add MLX-specific configuration.

## Overview

```python
from easymlx import EasyMLXBaseConfig

config = EasyMLXBaseConfig(attn_mechanism="sdpa", dtype="float16")
print(config.mlx_dtype)  # mx.float16
```

## Parameters

`attn_mechanism`
: The attention implementation to use. One of `"auto"`, `"vanilla"`, `"sdpa"`, or `"paged"`. Default: `"sdpa"`.

`dtype`
: String representation of the MLX dtype. Supported values: `"float16"` / `"fp16"`, `"bfloat16"` / `"bf16"`, `"float32"` / `"fp32"`. Default: `"float16"`.

## Attention Mechanisms

| Mechanism | Description |
|-----------|-------------|
| `auto` | Automatically selects the best available mechanism |
| `vanilla` | Standard dot-product attention |
| `sdpa` | Scaled dot-product attention (default) |
| `paged` | Paged attention for eSurge engine |

## HuggingFace Compatibility

`EasyMLXBaseConfig` inherits from `PretrainedConfig`, so it supports standard HuggingFace config operations:

```python
# Save
config.save_pretrained("/path/to/dir")

# Load
config = EasyMLXBaseConfig.from_pretrained("/path/to/dir")
```

## Subclassing

Model-specific configs should inherit from `EasyMLXBaseConfig`:

```python
from easymlx import EasyMLXBaseConfig
from easymlx.infra.factory import register_config

@register_config("my_model")
class MyModelConfig(EasyMLXBaseConfig):
    model_type = "my_model"

    def __init__(self, hidden_size=768, num_layers=12, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
```
