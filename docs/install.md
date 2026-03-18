# Installation

## Requirements

- **Python** >= 3.13
- **macOS** on Apple Silicon (M1/M2/M3/M4/M5)
- **MLX** >= 0.31.1

## Install from PyPI

```bash
pip install easymlx
```

To include the optional API server dependencies:

```bash
pip install easymlx[server]
```

## Install from Source

Using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/erfanzar/easymlx.git
cd easymlx
uv sync
```

Using pip:

```bash
git clone https://github.com/erfanzar/easymlx.git
cd easymlx
pip install -e .
```

For server support from source:

```bash
pip install -e ".[server]"
```

## Verify Installation

```python
import easymlx
print(easymlx.__version__)
```

## Dependencies

Core dependencies installed automatically:

- `mlx >= 0.31.1` -- Apple's ML framework for Apple Silicon
- `transformers >= 4.57.3` -- HuggingFace model loading and tokenizers
- `pydantic >= 2.12.5` -- Data validation

Optional:

- `fastapi >= 0.100.0` -- Required for the eSurge API server (`easymlx[server]`)
