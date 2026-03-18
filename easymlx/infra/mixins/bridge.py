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

import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..base_config import EasyMLXBaseConfig
from ..errors import EasyMLXRuntimeError
from ..etils import MODEL_WEIGHTS_GLOB, MODEL_WEIGHTS_NAME

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


def _load_safetensors_file(path: Path) -> dict[str, mx.array]:
    """Load a safetensors file using the ``safetensors`` library.

    The tensors are loaded via NumPy and then converted to ``mx.array``.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        A dictionary mapping parameter names to MLX arrays.

    Raises:
        EasyMLXRuntimeError: If the ``safetensors`` package is not
            installed.
    """
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover
        raise EasyMLXRuntimeError(
            f"Could not load raw Hugging Face safetensors from {path!s}; install `safetensors`."
        ) from exc

    weights: dict[str, mx.array] = {}
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():
            weights[key] = mx.array(handle.get_tensor(key))
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

    Respects the ``EASYMLX_CONVERTED_CACHE`` environment variable.
    Falls back to ``~/.cache/easymlx/converted``.

    Returns:
        A :class:`Path` to the cache root directory.
    """
    return Path(os.getenv("EASYMLX_CONVERTED_CACHE", Path.home() / ".cache" / "easymlx" / "converted"))


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

        config_class = getattr(cls, "config_class", None) or EasyMLXBaseConfig
        if config is None:
            config = config_class.from_pretrained(str(model_path))
        elif isinstance(config, (str, os.PathLike)):
            config = config_class.from_pretrained(str(config))
        elif isinstance(config, dict):
            config = config_class(**config)
        elif not isinstance(config, config_class):  # pragma: no cover
            raise TypeError(f"Unsupported config type: {type(config)}")

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

        config_class = EasyMLXBaseConfig if cls.config_class is None else cls.config_class
        if config is None:
            config = config_class.from_pretrained(str(model_path))
        elif isinstance(config, (str, os.PathLike)):
            config = config_class.from_pretrained(str(config))
        elif isinstance(config, dict):
            config = config_class(**config)
        elif not isinstance(config, config_class):  # pragma: no cover
            raise TypeError(f"Unsupported config type: {type(config)}")

        model = cls(config)

        weight_files = _resolve_weight_files(model_path, weights_name=weights_name)

        if not weight_files:
            raise FileNotFoundError(
                f"No weights found under {model_path!s}. Expected {MODEL_WEIGHTS_GLOB!r} (or pass weights_name=...)."
            )

        weights: dict[str, mx.array] = {}
        for weight_file in weight_files:
            weights.update(_load_weights_file(weight_file))

        sanitize = getattr(model, "sanitize", None)
        if callable(sanitize):
            sanitized = sanitize(weights)
            if sanitized is not None:
                weights = sanitized

        cast_dtype = dtype or config.mlx_dtype
        weights = _cast_weights(weights, dtype=cast_dtype)

        if device is not None:
            mx.set_default_device(device)

        model.load_weights(list(weights.items()), strict=strict)
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
