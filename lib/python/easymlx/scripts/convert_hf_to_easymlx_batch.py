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

"""Batch checkpoint conversion helpers.

Provides a convenience wrapper around :func:`convert_hf_checkpoint` for
converting multiple Hugging Face checkpoints in a single call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .convert_hf_to_easymlx import convert_hf_checkpoint


def convert_hf_checkpoints(
    model_cls: type[Any],
    model_names_or_paths: list["str"],
    output_root: str | Path,
    **kwargs: Any,
) -> list[Path]:
    """Convert multiple Hugging Face checkpoints into easymlx format.

    Iterates over a list of model identifiers or paths, converting each
    one into a subdirectory under ``output_root``.  Subdirectory names
    are derived from the basename of each model path, with ``/``
    replaced by ``_``.

    Args:
        model_cls: A model class that exposes a ``convert_hf_checkpoint``
            class method (see :func:`convert_hf_checkpoint`).
        model_names_or_paths: List of Hugging Face Hub model identifiers
            or local filesystem paths to pretrained checkpoint directories.
        output_root: Root directory under which each converted checkpoint
            will be stored in its own subdirectory.
        **kwargs: Additional keyword arguments forwarded to each call of
            :func:`convert_hf_checkpoint`.

    Returns:
        A list of :class:`~pathlib.Path` objects, one per converted
        checkpoint, in the same order as ``model_names_or_paths``.

    Example:
        >>> paths = convert_hf_checkpoints(
        ...     MyModelClass,
        ...     ["meta-llama/Llama-3-8B", "mistralai/Mistral-7B"],
        ...     "/tmp/converted_models",
        ... )
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for item in model_names_or_paths:
        target = output_root / Path(item).name.replace("/", "_")
        outputs.append(convert_hf_checkpoint(model_cls, item, target, **kwargs))
    return outputs


__all__ = "convert_hf_checkpoints"
