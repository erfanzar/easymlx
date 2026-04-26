# Adding Models to EasyMLX



This guide walks through adding a new model architecture to EasyMLX.



## Step 1: Create the Config



Create a configuration class that inherits from `EasyMLXBaseConfig` and register it:



```python



from easymlx.infra.base_config import EasyMLXBaseConfig

from easymlx.infra.factory import register_config



@register_config("my_model")

class MyModelConfig(EasyMLXBaseConfig):

    model_type = "my_model"



    def __init__(

        self,

        vocab_size=32000,

        hidden_size=4096,

        num_hidden_layers=32,

        num_attention_heads=32,

        num_key_value_heads=8,

        intermediate_size=11008,

        max_position_embeddings=4096,

        rms_norm_eps=1e-6,

        **kwargs,

    ):

        super().__init__(**kwargs)

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.intermediate_size = intermediate_size

        self.max_position_embeddings = max_position_embeddings

        self.rms_norm_eps = rms_norm_eps

```



## Step 2: Implement the Module



Create the model class inheriting from `EasyMLXBaseModule` and register it with a task type:



```python



import mlx.core as mx

import mlx.nn as nn



from easymlx.infra.base_module import EasyMLXBaseModule

from easymlx.infra.factory import TaskType, register_module

from .my_model_configuration import MyModelConfig



@register_module(

    task_type=TaskType.CAUSAL_LM,

    config=MyModelConfig,

    model_type="my_model",

)

class MyModelForCausalLM(EasyMLXBaseModule):

    config_class = MyModelConfig



    def __init__(self, config: MyModelConfig, **kwargs):

        super().__init__(config, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)



        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)



    def __call__(

        self,

        input_ids: mx.array,

        attention_mask: mx.array | None = None,

        position_ids: mx.array | None = None,

        cache=None,

        **kwargs,

    ):

        hidden_states = self.embed_tokens(input_ids)



        logits = self.lm_head(hidden_states)

        return logits

```



## Step 3: Create the Package



Set up the `__init__.py` to export your classes:



```python



from .my_model_configuration import MyModelConfig

from .modeling_my_model import MyModelForCausalLM, MyModelModel



__all__ = ("MyModelConfig", "MyModelForCausalLM", "MyModelModel")

```



## Step 4: Register in the Import Structure



Add your module to the lazy import structure in `easymlx/__init__.py`:



```python

_import_structure = {



    "modules.my_model": [

        "MyModelConfig",

        "MyModelForCausalLM",

        "MyModelModel",

    ],

}

```



## Step 5: Weight Conversion



EasyMLX uses `EasyBridgeMixin.from_pretrained` which handles HuggingFace weight conversion automatically. If your model's weight names differ from the EasyMLX module structure, you may need to add a `WEIGHT_MAP` class attribute or implement `_convert_weight_name` to map HuggingFace parameter names to EasyMLX parameter names.



## Step 6: Test



```python

from easymlx import AutoEasyMLXModelForCausalLM, eSurge, SamplingParams



model = AutoEasyMLXModelForCausalLM.from_pretrained("org/my-model")

engine = eSurge(model, max_model_len=4096, max_num_seqs=2)



results = engine.generate(

    "Hello, world!",

    sampling_params=SamplingParams(max_tokens=64),

)

print(results[0].text)

engine.close()

```



## Key Design Rules



1. **All arrays must be `mx.array`** -- never use NumPy arrays in model code.

2. **Eager execution** -- MLX uses eager mode. No tracing, no `jit`, no sharding.

3. **Mutable modules** -- MLX `nn.Module` is mutable. Parameters are updated in-place.

4. **No training** -- EasyMLX is inference-only. Do not add training-specific code.

5. **Paged attention** -- Models must support paged KV cache via `init_operations_cache` or `init_paged_cache` to work with eSurge.
