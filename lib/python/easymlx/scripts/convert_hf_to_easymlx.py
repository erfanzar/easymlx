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

"""Helpers for converting raw Hugging Face checkpoints into easymlx format.

Provides a thin wrapper around a model class's ``convert_hf_checkpoint``
method, standardizing the output path handling and return value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def convert_hf_checkpoint(
    model_cls: type[Any],
    model_name_or_path: str,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Delegate to a bridge-enabled model class to convert a HF checkpoint.

    Converts a single Hugging Face checkpoint into the easymlx format by
    calling the ``convert_hf_checkpoint`` class method on the provided
    model class.

    Args:
        model_cls: A model class that exposes a ``convert_hf_checkpoint``
            class method accepting ``(model_name_or_path, output_path, **kwargs)``.
        model_name_or_path: A Hugging Face Hub model identifier (e.g.
            ``"meta-llama/Llama-3-8B"``) or a local filesystem path to a
            pretrained checkpoint directory.
        output_dir: Destination directory where the converted checkpoint
            will be written.
        **kwargs: Additional keyword arguments forwarded to
            ``model_cls.convert_hf_checkpoint``.

    Returns:
        The resolved :class:`~pathlib.Path` to the output directory.

    Example:
        >>> from pathlib import Path
        >>> output = convert_hf_checkpoint(
        ...     MyModelClass, "meta-llama/Llama-3-8B", Path("/tmp/converted")
        ... )
    """

    output_path = Path(output_dir)
    model_cls.convert_hf_checkpoint(model_name_or_path, output_path, **kwargs)
    return output_path


__all__ = "convert_hf_checkpoint"
