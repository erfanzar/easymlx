# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model I/O helpers for easymlx.

This mixin intentionally targets MLX-native weights (typically ``model*.safetensors``)
and HuggingFace-compatible ``config.json``.

It provides three main capabilities:

1. **Saving** -- :meth:`EasyBridgeMixin.save_pretrained`
2. **HuggingFace conversion** -- :meth:`EasyBridgeMixin.convert_hf_checkpoint`
3. **Loading** -- :meth:`EasyBridgeMixin.from_pretrained`
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import mmap
import os
import shutil
import struct
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..base_config import EasyMLXBaseConfig
from ..errors import EasyMLXRuntimeError
from ..etils import MODEL_WEIGHTS_GLOB, MODEL_WEIGHTS_NAME, QuantizationConfig, QuantizationMode

_HF_SUPPORT_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
_CONVERTED_SOURCE_METADATA = "easymlx_source.json"


@contextlib.contextmanager
def _no_weight_init():
    """Suppress random weight initialisation during model construction.

    MLX modules (``nn.Linear``, ``nn.Embedding``, …) allocate random
    weights in ``__init__``.  When we are about to overwrite every
    parameter with pretrained weights, building the random computation
    graph is wasted work.  This context manager temporarily replaces
    ``mx.random.uniform`` and ``mx.random.normal`` with zero-returning
    stubs so that model construction is essentially free.
    """
    orig_uniform = mx.random.uniform
    orig_normal = mx.random.normal

    def _zero_uniform(*_a, shape=(), dtype=mx.float32, **_kw):
        return mx.zeros(shape, dtype=dtype)

    def _zero_normal(*_a, shape=(), dtype=mx.float32, **_kw):
        return mx.zeros(shape, dtype=dtype)

    mx.random.uniform = _zero_uniform
    mx.random.normal = _zero_normal
    try:
        yield
    finally:
        mx.random.uniform = orig_uniform
        mx.random.normal = orig_normal


_QUANT_MODE_DEFAULTS: dict[str, dict[str, int]] = {
    "affine": {"bits": 4, "group_size": 64},
    "mxfp4": {"bits": 4, "group_size": 32},
    "mxfp8": {"bits": 8, "group_size": 32},
    "nvfp4": {"bits": 4, "group_size": 16},
}


def _parse_quantization_config(
    quantization: QuantizationConfig | QuantizationMode,
) -> dict:
    """Parse a quantization spec into a validated kwargs dict.

    Args:
        quantization: A mode string or a :class:`QuantizationConfig` dict.

    Returns:
        A dict with ``mode``, ``bits``, ``group_size`` ready for
        :func:`mlx.nn.quantize` / :func:`mx.quantize`.

    Raises:
        ValueError: If the mode is not recognised.
    """
    if isinstance(quantization, str):
        quantization = QuantizationConfig(mode=quantization)

    mode = quantization.get("mode", "affine")
    if mode not in _QUANT_MODE_DEFAULTS:
        raise ValueError(f"Unknown quantization mode '{mode}'. Supported: {tuple(_QUANT_MODE_DEFAULTS)}")

    defaults = _QUANT_MODE_DEFAULTS[mode]
    return {
        "mode": mode,
        "bits": quantization.get("bits", defaults["bits"]),
        "group_size": quantization.get("group_size", defaults["group_size"]),
    }


def _validate_quantization_kernel(quant_kwargs: dict) -> None:
    """Run a tiny dummy quantize+matmul to verify the Metal kernel loads.

    Raises:
        RuntimeError: With a user-friendly message if the kernel is
            unavailable on the current device.
    """
    gs = quant_kwargs["group_size"]
    bits = quant_kwargs["bits"]
    mode = quant_kwargs["mode"]
    try:
        dummy_w = mx.ones((gs, gs))
        quant_result = mx.quantize(dummy_w, gs, bits, mode=mode)
        q, scales = quant_result[0], quant_result[1]
        biases = quant_result[2] if len(quant_result) > 2 else None
        out = mx.quantized_matmul(
            mx.ones((1, gs)),
            q,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=gs,
            bits=bits,
            mode=mode,
        )
        mx.eval(out)
    except RuntimeError as exc:
        if "Unable to load kernel" in str(exc):
            device_info = mx.metal.device_info()
            device_name = device_info.get("device_name", "unknown")
            raise RuntimeError(
                f"Quantization mode '{mode}' is not supported on {device_name} "
                f"(arch={device_info.get('architecture', '?')}). "
                f"The required Metal kernel is not available in MLX {mx.__version__}. "
                f"Try mode='affine' (bits=4, group_size=64) which works on all Apple Silicon, "
                f"or upgrade MLX: pip install -U mlx"
            ) from exc
        raise


def _quantize_weights(
    weights: dict[str, mx.array],
    quantized_weight_keys: frozenset[str],
    quant_kwargs: dict,
) -> dict[str, mx.array]:
    """Quantize full-precision weights that belong to quantized modules.

    For every key in *weights* that appears in *quantized_weight_keys*
    **and** has a floating-point dtype, the value is replaced with the
    three arrays produced by :func:`mx.quantize` (packed weight, scales,
    biases).  All other entries pass through unchanged.

    This lets us quantize weights while streaming shards, so full-precision
    data for the whole model never needs to live in memory at once.

    Args:
        weights: Flat weight dictionary (already sanitized / cast).
        quantized_weight_keys: Set of ``"prefix.weight"`` keys whose
            corresponding module was converted to ``QuantizedLinear``
            (or ``QuantizedEmbedding``).
        quant_kwargs: ``dict(mode=..., bits=..., group_size=...)``.

    Returns:
        A new dictionary with quantized entries replacing the originals.
    """
    group_size = quant_kwargs["group_size"]
    bits = quant_kwargs["bits"]
    mode = quant_kwargs["mode"]

    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key in quantized_weight_keys and mx.issubdtype(value.dtype, mx.floating):
            quant_result = mx.quantize(value, group_size, bits, mode=mode)
            # key is already "prefix.weight"; reuse it directly.
            out[key] = quant_result[0]
            prefix = key[: -len(".weight")]
            out[f"{prefix}.scales"] = quant_result[1]
            if len(quant_result) > 2:
                out[f"{prefix}.biases"] = quant_result[2]
        else:
            out[key] = value
    return out


def _apply_quantization(
    model: EasyBridgeMixin,
    quantization: QuantizationConfig | QuantizationMode | None,
) -> None:
    """Parse, validate, and apply quantization to a model in-place.

    Convenience wrapper used by external callers (e.g. auto-model classes).
    ``from_pretrained`` uses the lower-level helpers directly so it can
    quantize per-shard for better memory behaviour.

    Args:
        model: The model to quantize.
        quantization: Mode string or :class:`QuantizationConfig` dict.
            ``None`` / ``""`` is a no-op.
    """
    import logging

    import mlx.nn as nn

    if quantization is None or quantization == "":
        return

    quant_kwargs = _parse_quantization_config(quantization)
    logger = logging.getLogger("easymlx.quantization")
    logger.info(
        "Quantizing model: mode=%s, bits=%d, group_size=%d",
        quant_kwargs["mode"],
        quant_kwargs["bits"],
        quant_kwargs["group_size"],
    )
    _validate_quantization_kernel(quant_kwargs)
    nn.quantize(model, **quant_kwargs)
    logger.info("Quantization complete")


def _resolve_model_path(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a model identifier to a local directory path.

    If the path already exists locally it is returned directly. Otherwise
    the model is downloaded via ``huggingface_hub.snapshot_download``.

    Args:
        pretrained_model_name_or_path: A local filesystem path or a
            HuggingFace Hub repository ID (e.g.
            ``"meta-llama/Llama-3-8B"``).
        revision: Optional git revision (branch, tag, or commit SHA) for
            the Hub download.
        local_files_only: If ``True``, never attempt a network download.

    Returns:
        A :class:`Path` pointing to the local model directory.

    Raises:
        EasyMLXRuntimeError: If the path does not exist locally and the
            ``huggingface_hub`` package is not available.
    """
    path = Path(pretrained_model_name_or_path)
    if path.exists():
        return path

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise EasyMLXRuntimeError(
            f"Could not resolve {pretrained_model_name_or_path!r} as a local path, and huggingface_hub is unavailable."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def _resolve_weight_files(model_path: Path, *, weights_name: str | None = None) -> list[Path]:
    """Locate weight files inside a model directory.

    Args:
        model_path: The model directory to search.
        weights_name: If provided, return exactly this filename under
            *model_path* without searching.

    Returns:
        A sorted list of :class:`Path` objects pointing to the
        discovered weight files. May be empty if nothing matches.
    """
    if weights_name is not None:
        return [model_path / weights_name]

    weight_files = [Path(path) for path in sorted(glob.glob(str(model_path / MODEL_WEIGHTS_GLOB)))]
    if weight_files:
        return weight_files

    fallback = model_path / "weights.safetensors"
    return [fallback] if fallback.exists() else []


_SAFETENSORS_DTYPE_MAP: dict[str, tuple[mx.Dtype, int]] = {
    "BOOL": (mx.bool_, 1),
    "U8": (mx.uint8, 1),
    "I8": (mx.int8, 1),
    "U16": (mx.uint16, 2),
    "I16": (mx.int16, 2),
    "U32": (mx.uint32, 4),
    "I32": (mx.int32, 4),
    "U64": (mx.uint64, 8),
    "I64": (mx.int64, 8),
    "F16": (mx.float16, 2),
    "BF16": (mx.bfloat16, 2),
    "F32": (mx.float32, 4),
    "F64": (mx.float64, 8),
}


def _load_safetensors_file(path: Path) -> dict[str, mx.array]:
    """Load a safetensors file by parsing the binary format directly.

    Handles dtypes that ``mx.load`` or numpy cannot read natively,
    including ``BF16`` and ``F8_E5M2``/``F8_E4M3``.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        A dictionary mapping parameter names to MLX arrays.
    """
    import numpy as np

    weights: dict[str, mx.array] = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack("<Q", mm[:8])[0]
        header = json.loads(mm[8 : 8 + header_size])
        data_offset = 8 + header_size

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str: str = meta["dtype"]
            shape: list[int] = meta["shape"]
            start, end = meta["data_offsets"]
            raw = mm[data_offset + start : data_offset + end]

            if dtype_str in ("F8_E5M2", "F8_E4M3FN", "F8_E4M3"):
                # FP8 E5M2 has same exponent layout as float16 — left-shift by 8.
                # FP8 E4M3 needs exponent bias adjustment: bias 7 -> bias 15.
                u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
                if dtype_str == "F8_E5M2":
                    f16_bits = u8 << 8
                else:
                    # E4M3: sign(1) exp(4) man(3) -> sign(1) exp(5) man(10)
                    sign = (u8 >> 7) & 1
                    exp = (u8 >> 3) & 0xF
                    man = u8 & 0x7
                    # Re-bias exponent: E4M3 bias=7, F16 bias=15
                    new_exp = np.where(exp == 0, np.uint16(0), exp.astype(np.uint16) - 7 + 15)
                    f16_bits = (sign << 15) | (new_exp << 10) | (man.astype(np.uint16) << 7)
                arr = mx.array(f16_bits.reshape(shape)).view(mx.float16)
            elif dtype_str in _SAFETENSORS_DTYPE_MAP:
                mlx_dtype, itemsize = _SAFETENSORS_DTYPE_MAP[dtype_str]
                if dtype_str == "BF16":
                    np_arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                    arr = mx.array(np_arr).view(mx.bfloat16)
                else:
                    np_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[itemsize]
                    np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                    arr = mx.array(np_arr).view(mlx_dtype)
            else:
                raise EasyMLXRuntimeError(
                    f"Unsupported safetensors dtype {dtype_str!r} for tensor {key!r} in {path!s}."
                )
            weights[key] = arr

        mm.close()
    return weights


def _load_torch_file(path: Path) -> dict[str, mx.array]:
    """Load a PyTorch checkpoint file and convert tensors to MLX arrays.

    Args:
        path: Path to a ``.bin``, ``.pt``, or ``.pth`` file.

    Returns:
        A dictionary mapping parameter names to MLX arrays.

    Raises:
        EasyMLXRuntimeError: If the ``torch`` package is not installed.
    """
    try:
        import torch  # type:ignore
    except Exception as exc:  # pragma: no cover
        raise EasyMLXRuntimeError(
            f"Could not load raw Hugging Face torch checkpoint from {path!s}; install `torch`."
        ) from exc

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")

    weights: dict[str, mx.array] = {}
    for key, tensor in state_dict.items():
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        weights[key] = mx.array(tensor)
    return weights


def _load_weights_file(path: Path) -> dict[str, mx.array]:
    """Load weights from a file, trying MLX native format first.

    Falls back to :func:`_load_safetensors_file` for ``.safetensors``
    files and :func:`_load_torch_file` for ``.bin``/``.pt``/``.pth``
    files when native loading fails.

    Args:
        path: Path to the weight file.

    Returns:
        A dictionary mapping parameter names to MLX arrays.

    Raises:
        Exception: Re-raises the original exception if the file format
            is not recognized by any loader.
    """
    try:
        return mx.load(str(path))
    except Exception:
        if path.suffix == ".safetensors":
            return _load_safetensors_file(path)
        if path.suffix in {".bin", ".pt", ".pth"}:
            return _load_torch_file(path)
        raise


def _copy_hf_support_files(source_dir: Path, target_dir: Path) -> None:
    """Copy HuggingFace tokenizer and generation support files.

    Copies files like ``tokenizer.json``, ``tokenizer_config.json``,
    ``generation_config.json``, etc. from *source_dir* to *target_dir*
    if they exist.

    Args:
        source_dir: Source directory containing the support files.
        target_dir: Destination directory to copy files into.
    """
    for filename in _HF_SUPPORT_FILES:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)


def _cast_weights(
    weights: dict[str, mx.array],
    *,
    dtype: mx.Dtype | None,
) -> dict[str, mx.array]:
    """Cast floating-point weights to a target dtype.

    Non-floating-point arrays (e.g. integer embeddings) and arrays
    that already match the target dtype are left unchanged.

    Args:
        weights: Dictionary of parameter name to MLX array.
        dtype: Target dtype. If ``None``, the weights are returned
            unmodified.

    Returns:
        A new dictionary with the same keys, where floating-point
        values have been cast to *dtype*.
    """
    if dtype is None:
        return weights

    casted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype:
            casted[key] = value.astype(dtype)
        else:
            casted[key] = value
    return casted


def _default_converted_cache_root() -> Path:
    """Return the default cache root for converted checkpoints.

    Returns:
        A :class:`Path` to ``~/.cache/easymlx/converted``.
    """
    return Path.home() / ".cache" / "easymlx" / "converted"


def _conversion_cache_path(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    revision: str | None = None,
    converted_cache_dir: str | os.PathLike | None = None,
) -> Path:
    """Compute the cache directory path for a converted checkpoint.

    The path is deterministic and based on a SHA-1 digest of the source
    identifier and revision.

    Args:
        pretrained_model_name_or_path: Original model path or Hub ID.
        revision: Git revision string. Defaults to ``"main"`` for
            digest computation.
        converted_cache_dir: Override for the cache root directory.
            If ``None``, :func:`_default_converted_cache_root` is used.

    Returns:
        A :class:`Path` for the conversion cache directory.
    """
    source = str(pretrained_model_name_or_path)
    slug = Path(source.rstrip("/")).name or "model"
    digest = hashlib.sha1(f"{source}@{revision or 'main'}".encode()).hexdigest()[:12]
    root = Path(converted_cache_dir) if converted_cache_dir is not None else _default_converted_cache_root()
    return root / f"{slug}-{digest}"


def _has_converted_checkpoint(path: Path, *, weights_name: str) -> bool:
    """Check whether a converted checkpoint already exists at *path*.

    Args:
        path: Directory to check.
        weights_name: Expected weight file name.

    Returns:
        ``True`` if both ``config.json`` and the weight file exist.
    """
    return (path / "config.json").exists() and (path / weights_name).exists()


def _resolve_config(
    cls: type,
    config: EasyMLXBaseConfig | dict | str | os.PathLike | None,
    model_path: Path,
) -> EasyMLXBaseConfig:
    """Resolve a config argument into an ``EasyMLXBaseConfig`` instance."""
    config_class = getattr(cls, "config_class", None) or EasyMLXBaseConfig
    if config is None:
        return config_class.from_pretrained(str(model_path))
    if isinstance(config, (str, os.PathLike)):
        return config_class.from_pretrained(str(config))
    if isinstance(config, dict):
        return config_class(**config)
    if not isinstance(config, config_class):  # pragma: no cover
        raise TypeError(f"Unsupported config type: {type(config)}")
    return config


class EasyBridgeMixin:
    """Mixin providing model saving, loading, and HuggingFace conversion.

    This mixin is designed to be composed into
    :class:`~easymlx.infra.base_module.EasyMLXBaseModule`. It adds
    :meth:`save_pretrained`, :meth:`from_pretrained`, and
    :meth:`convert_hf_checkpoint` class/instance methods.

    Attributes:
        config: The active model configuration.
        config_class: The configuration class to use for loading.
    """

    config: EasyMLXBaseConfig
    config_class: type[EasyMLXBaseConfig] | None = None

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *,
        weights_name: str = MODEL_WEIGHTS_NAME,
    ) -> None:
        """Save the model configuration and weights to a directory.

        Args:
            save_directory: Destination directory. Created if it does
                not exist.
            weights_name: Filename for the safetensors weight file.
                Defaults to ``"model.safetensors"``.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        if getattr(self, "config", None) is not None:
            self.config.save_pretrained(str(path))
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(path / weights_name), weights)

    @classmethod
    def convert_hf_checkpoint(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        save_directory: str | os.PathLike,
        config: EasyMLXBaseConfig | dict | str | os.PathLike | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        weights_name: str = MODEL_WEIGHTS_NAME,
        copy_support_files: bool = True,
    ) -> Path:
        """Convert a HuggingFace checkpoint to easymlx format.

        Downloads (if necessary) the source checkpoint, instantiates
        the model to obtain the ``sanitize`` mapping, casts weights to
        the configured dtype, and writes the result to *save_directory*.

        Args:
            pretrained_model_name_or_path: Local path or HuggingFace
                Hub repository ID.
            save_directory: Where to write the converted checkpoint.
            config: Configuration to use. Accepts an
                ``EasyMLXBaseConfig`` instance, a ``dict`` of kwargs, a
                path to a ``config.json`` file, or ``None`` to auto-load
                from the source checkpoint.
            revision: Git revision for Hub downloads.
            local_files_only: Disable network access.
            weights_name: Filename for the output safetensors file.
            copy_support_files: If ``True``, copy tokenizer and other
                HuggingFace support files alongside the weights.

        Returns:
            The :class:`Path` to *save_directory* after writing.

        Raises:
            FileNotFoundError: If no weight files are found in the
                source checkpoint.
            TypeError: If *config* is an unsupported type.
        """
        model_path = _resolve_model_path(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )

        config = _resolve_config(cls, config, model_path)

        with _no_weight_init():
            model = cls(config)
        weight_files = _resolve_weight_files(model_path)
        if not weight_files:
            raise FileNotFoundError(f"No weights found under {model_path!s}. Expected {MODEL_WEIGHTS_GLOB!r}.")

        weights: dict[str, mx.array] = {}
        for weight_file in weight_files:
            weights.update(_load_weights_file(weight_file))

        sanitize = getattr(model, "sanitize", None)
        if callable(sanitize):
            sanitized = sanitize(weights)
            if sanitized is not None:
                weights = sanitized

        weights = _cast_weights(weights, dtype=getattr(config, "mlx_dtype", None))

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(str(save_path))
        mx.save_safetensors(str(save_path / weights_name), weights)
        if copy_support_files:
            _copy_hf_support_files(model_path, save_path)
        metadata = {
            "source": str(pretrained_model_name_or_path),
            "resolved_path": str(model_path),
            "revision": revision,
            "weights_name": weights_name,
        }
        (save_path / _CONVERTED_SOURCE_METADATA).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return save_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        config: EasyMLXBaseConfig | dict | str | os.PathLike | None = None,
        dtype: mx.Dtype | None = None,
        device: mx.Device | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        weights_name: str | None = None,
        strict: bool = True,
        lazy: bool = False,
        auto_convert_hf: bool | None = None,
        converted_cache_dir: str | os.PathLike | None = None,
        force_conversion: bool = False,
        copy_support_files: bool = True,
        quantization: QuantizationConfig | QuantizationMode | None = None,
    ):
        """Load a pretrained model from a local directory or Hub ID.

        If the source is a HuggingFace checkpoint (not already in
        easymlx format) and ``auto_convert_hf`` is enabled, the
        checkpoint is automatically converted and cached before loading.

        Args:
            pretrained_model_name_or_path: Local path or HuggingFace
                Hub repository ID.
            config: Configuration override. See
                :meth:`convert_hf_checkpoint` for accepted types.
            dtype: Override the dtype for all floating-point weights.
                If ``None``, the config's ``mlx_dtype`` is used.
            device: If provided, sets the MLX default device before
                loading weights.
            revision: Git revision for Hub downloads.
            local_files_only: Disable network access.
            weights_name: Explicit weight file name. If ``None``,
                auto-discovered via glob.
            strict: If ``True``, all weight keys must match the model
                parameters exactly.
            lazy: If ``True``, skip eager evaluation of parameters
                after loading.
            auto_convert_hf: Whether to auto-convert HuggingFace
                checkpoints. Defaults to ``True`` for Hub IDs and
                ``False`` for local paths.
            converted_cache_dir: Override directory for the conversion
                cache.
            force_conversion: If ``True``, re-convert even if a cached
                conversion exists.
            copy_support_files: If ``True``, copy tokenizer files
                during conversion.

        Returns:
            An instance of this class with weights loaded.

        Raises:
            FileNotFoundError: If no weight files are found.
            TypeError: If *config* is an unsupported type.
        """
        source_is_local = Path(pretrained_model_name_or_path).exists()
        if auto_convert_hf is None:
            auto_convert_hf = not source_is_local

        resolved_weights_name = weights_name or MODEL_WEIGHTS_NAME
        if auto_convert_hf:
            cache_path = _conversion_cache_path(
                pretrained_model_name_or_path,
                revision=revision,
                converted_cache_dir=converted_cache_dir,
            )
            if force_conversion or not _has_converted_checkpoint(cache_path, weights_name=resolved_weights_name):
                cls.convert_hf_checkpoint(
                    pretrained_model_name_or_path,
                    save_directory=cache_path,
                    config=config,
                    revision=revision,
                    local_files_only=local_files_only,
                    weights_name=resolved_weights_name,
                    copy_support_files=copy_support_files,
                )
            model_path = cache_path
            config = None
            revision = None
            local_files_only = True
            weights_name = resolved_weights_name
        else:
            model_path = _resolve_model_path(
                pretrained_model_name_or_path,
                revision=revision,
                local_files_only=local_files_only,
            )

        config = _resolve_config(cls, config, model_path)

        # Build the model skeleton without allocating random weights.
        with _no_weight_init():
            model = cls(config)

        weight_files = _resolve_weight_files(model_path, weights_name=weights_name)

        if not weight_files:
            raise FileNotFoundError(
                f"No weights found under {model_path!s}. Expected {MODEL_WEIGHTS_GLOB!r} (or pass weights_name=...)."
            )

        sanitize_fn = getattr(model, "sanitize", None)
        if not callable(sanitize_fn):
            sanitize_fn = None

        cast_dtype = dtype or config.mlx_dtype

        if device is not None:
            mx.set_default_device(device)

        # ------------------------------------------------------------------
        # Shard-level quantization: if the caller requested quantization we
        # convert the model's Linear/Embedding modules to their quantized
        # variants *first* (on the zero-init skeleton — essentially free),
        # then quantize each shard's fp weights on-the-fly as they stream
        # in.  This way full-precision data only exists for one shard at a
        # time → peak memory ≈ quantized-model + one-fp-shard.
        # ------------------------------------------------------------------
        quant_kwargs: dict | None = None
        quantized_weight_keys: frozenset[str] = frozenset()

        if quantization is not None:
            import logging

            import mlx.nn as nn

            quant_kwargs = _parse_quantization_config(quantization)
            _validate_quantization_kernel(quant_kwargs)

            logger = logging.getLogger("easymlx.quantization")
            logger.info(
                "Quantizing model: mode=%s, bits=%d, group_size=%d",
                quant_kwargs["mode"],
                quant_kwargs["bits"],
                quant_kwargs["group_size"],
            )

            # Set up QuantizedLinear / QuantizedEmbedding structure.
            nn.quantize(model, **quant_kwargs)

            # Collect the weight keys the model now expects in quantized
            # form so we can transform each shard to match.
            quantized_weight_keys = frozenset(
                k.rsplit(".scales", 1)[0] + ".weight"
                for k, _ in tree_flatten(model.parameters())
                if k.endswith(".scales")
            )

        # ------------------------------------------------------------------
        # Stream weights shard-by-shard.  After loading each shard we
        # immediately evaluate its arrays so the mmap-backed lazy data is
        # materialised into real (and possibly quantized) memory and the
        # file pages can be reclaimed.  This keeps the memory watermark
        # proportional to (quantized-model-so-far + one-shard) instead of
        # (entire-fp-model + quantized-model).
        # ------------------------------------------------------------------
        loaded_keys: set[str] = set()
        for weight_file in weight_files:
            shard = _load_weights_file(weight_file)
            if sanitize_fn is not None:
                sanitized = sanitize_fn(shard)
                if sanitized is not None:
                    shard = sanitized
            shard = _cast_weights(shard, dtype=cast_dtype)

            if quant_kwargs is not None:
                shard = _quantize_weights(shard, quantized_weight_keys, quant_kwargs)

            shard_items = list(shard.items())
            loaded_keys.update(k for k, _ in shard_items)
            model.load_weights(shard_items, strict=False)

            # Eagerly evaluate this shard so its backing mmap pages can be
            # released before the next shard is opened.
            if not lazy:
                mx.eval([v for _, v in shard_items])

            del shard, shard_items

        if strict:
            model_keys = {k for k, _ in tree_flatten(model.parameters())}
            missing = model_keys - loaded_keys
            extra = loaded_keys - model_keys
            if missing or extra:
                parts: list[str] = []
                if missing:
                    parts.append(f"Missing keys in checkpoint: {missing}")
                if extra:
                    parts.append(f"Unexpected keys in checkpoint: {extra}")
                raise ValueError(". ".join(parts))

        # Catch any remaining un-evaluated params (e.g. norms, biases not
        # covered by shards).  For already-evaluated arrays this is a no-op.
        if not lazy:
            mx.eval(model.parameters())
        model.eval()
        model.name_or_path = str(model_path)  # type: ignore[attr-defined]

        if model.config is None:
            model.config = config  # type: ignore[attr-defined]
        try:
            model.config.name_or_path = str(model_path)  # type: ignore[attr-defined]
        except Exception:
            pass
        return model
