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

"""Shared enums, type aliases, and constants for easymlx infra (serving-only).

This module centralizes lightweight type definitions and sentinel values
used across the easymlx infrastructure layer. It deliberately avoids
importing heavy dependencies so that it can be imported early.

Attributes:
    AttentionMechanism: A ``Literal`` type alias restricting attention
        mechanism names to ``"auto"``, ``"vanilla"``, ``"sdpa"``, and
        ``"paged"``.
    DEFAULT_ATTENTION_MECHANISM: The default attention mechanism used
        when none is explicitly specified. Currently ``"sdpa"``.
    RopeType: A ``Literal`` type alias for supported RoPE variants.
    MODEL_CONFIG_NAME: Standard filename for the model configuration
        JSON file (``"config.json"``).
    MODEL_WEIGHTS_GLOB: Glob pattern for discovering safetensors weight
        shards (``"model*.safetensors"``).
    MODEL_WEIGHTS_NAME: Default single-file weight name
        (``"model.safetensors"``).
"""

from __future__ import annotations

import typing as tp

AttentionMechanism = tp.Literal["auto", "vanilla", "sdpa", "paged"]
DEFAULT_ATTENTION_MECHANISM: AttentionMechanism = "sdpa"

RopeType = tp.Literal["default", "linear", "llama3", "yarn", "longrope", "mrope"]


MODEL_CONFIG_NAME = "config.json"
MODEL_WEIGHTS_GLOB = "model*.safetensors"
MODEL_WEIGHTS_NAME = "model.safetensors"


QuantizationMode = tp.Literal["affine", "mxfp4", "mxfp8", "nvfp4"]


class QuantizationConfig(tp.TypedDict, total=False):
    """Quantization configuration for model loading.

    Attributes:
        mode: Quantization mode. Supported modes:

            - ``"affine"`` — Standard affine quantization (default). Requires
              ``bits`` and ``group_size``.
            - ``"mxfp4"`` — Microscaling FP4 format.
            - ``"mxfp8"`` — Microscaling FP8 format.
            - ``"nvfp4"`` — NVIDIA FP4 format.
        bits: Number of bits per weight element. Only used for ``"affine"``
            mode. Common values: ``4`` (default), ``8``, ``2``.
        group_size: Number of elements sharing a scale/bias. Only used for
            ``"affine"`` mode. Default ``64``.

    Example:
        >>> QuantizationConfig(mode="affine", bits=4, group_size=64)
        >>> QuantizationConfig(mode="mxfp4")
        >>> QuantizationConfig(mode="nvfp4")
    """

    mode: QuantizationMode
    bits: int
    group_size: int
