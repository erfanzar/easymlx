# Environment Variables

EasyMLX reads the following environment variables to control runtime behavior.

## Cache and Storage

`EMLX_CACHE_HOME`
: Base directory for EasyMLX caches. Defaults to the platform-specific cache directory.

`EASYMLX_CONVERTED_CACHE`
: Directory for storing converted HuggingFace checkpoint weights. Defaults to `~/.cache/easymlx/converted`.

## Attention Kernels

`EASYMLX_UNIFIED_ATTENTION_USE_METAL`
: Set to `0` or `false` to disable Metal-accelerated unified attention and fall back to the reference implementation. Default: `true`.

`EASYMLX_UNIFIED_ATTENTION_FORCE_REFERENCE`
: Set to `1` or `true` to force the pure-Python reference attention implementation. Useful for debugging. Default: `false`.

`EASYMLX_UNIFIED_ATTENTION_ALLOW_FALLBACK`
: Set to `0` or `false` to raise an error instead of silently falling back to the reference implementation when the Metal kernel fails. Default: `true`.

## Distributed

`EASYMLX_DISTRIBUTED_SERVICE_NAME`
: Service name for distributed multi-node discovery. Used by the distributed controller to locate worker nodes via mDNS/Bonjour.

## Logging

`LOGGING_LEVEL_ED`
: Logging level for EasyMLX loggers. Accepts standard Python logging level names (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default: `INFO`.
