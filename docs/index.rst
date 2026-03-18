EasyMLX
=======

**MLX-native inference framework for Apple Silicon.**

EasyMLX is an inference-only port of `EasyDeL <https://github.com/erfanzar/EasyDeL>`_,
built from the ground up for MLX on Apple Silicon. It provides high-performance
text generation with paged attention, continuous batching, and an
OpenAI-compatible API server.

Key Features
------------

- **Paged Attention** -- Efficient KV cache management via page-based allocation.
- **Continuous Batching** -- eSurge engine schedules requests dynamically for maximum throughput.
- **Streaming** -- Token-by-token streaming with delta text output.
- **Tool Calling** -- Built-in tool/function calling with auto-detected parsers for major model families.
- **Reasoning Extraction** -- Automatic extraction of chain-of-thought reasoning content.
- **Quantization** -- Affine, MXFP4, MXFP8, and NVFP4 quantization modes.
- **OpenAI-compatible API** -- Drop-in server compatible with the OpenAI Python client.
- **Apple Silicon Optimized** -- Designed for Metal GPU acceleration via MLX.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Inference

   esurge
   esurge_examples
   api_server

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure

   infra/overview
   infra/base_config
   infra/base_module
   infra/adding_models

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   quantization
   environment_variables

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_docs/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
