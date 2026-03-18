# Infrastructure Overview

The `easymlx.infra` package provides the foundational abstractions for all model implementations in EasyMLX. It mirrors EasyDeL's infrastructure layer but is optimized for MLX with eager, mutable semantics and no sharding.

## Key Components

### EasyMLXBaseConfig

The base configuration class for all EasyMLX models. Extends HuggingFace's `PretrainedConfig` with MLX-specific knobs like attention mechanism selection and dtype mapping.

See {doc}`base_config` for details.

### EasyMLXBaseModule

The base module class for all EasyMLX neural network modules. Combines MLX's `nn.Module` with mixins for model I/O, text generation, and operation caching.

See {doc}`base_module` for details.

### Factory and Registry

The registry system (`easymlx.infra.factory`) maps model-type strings and task types to their corresponding configuration and module classes. The `register_config` and `register_module` decorators are the primary API for registering new model implementations.

### Supported Tasks

EasyMLX supports the following task types via `TaskType`:

| Task | Description |
|------|-------------|
| `CAUSAL_LM` | Autoregressive causal language modeling |
| `BASE_MODULE` | Generic base module |
| `SEQUENCE_CLASSIFICATION` | Sequence-level classification |
| `TOKEN_CLASSIFICATION` | Token-level classification (NER) |
| `QUESTION_ANSWERING` | Extractive question answering |
| `IMAGE_TEXT_TO_TEXT` | Multimodal image+text to text |
| `IMAGE_CLASSIFICATION` | Image classification |
| `EMBEDDING` | Embedding / retrieval models |

### Mixins

- **EasyBridgeMixin** -- Model saving, loading, and HuggingFace checkpoint conversion.
- **EasyGenerationMixin** -- Text generation (greedy and sampling).
- **OperationCacheMixin** -- Cache requirement discovery and allocation.
