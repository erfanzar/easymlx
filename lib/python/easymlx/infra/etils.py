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
        mechanism names.
    DEFAULT_ATTENTION_MECHANISM: The default attention mechanism used
        when none is explicitly specified. Currently ``"unified"``.
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

AttentionMechanism = tp.Literal[
    "auto",
    "vanilla",
    "sdpa",
    "paged",
    "unified",
    "unified_attention",
    "page_attention",
    "paged_attention",
]
DEFAULT_ATTENTION_MECHANISM: AttentionMechanism = "unified"


def canonical_attention_mechanism(value: object | None) -> str | None:
    """Return the canonical runtime operation name for an attention setting."""
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    mechanism = str(value).lower()
    if mechanism in {"", "auto", "paged", "unified", "unified_attention"}:
        return "unified_attention"
    if mechanism in {"page_attention", "paged_attention"}:
        return "page_attention"
    return mechanism


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


class QuantizationRule(tp.TypedDict, total=False):
    """Ordered regex rule for layer-wise quantization.

    Attributes:
        pattern: Regular expression matched against the MLX module path
            (for example ``"model.embed_tokens"`` or
            ``"model.layers.0.self_attn.q_proj"``).
        config: Quantization config to apply when *pattern* matches.
            Use ``None`` to explicitly skip quantization for matching
            modules.

    Example:
        >>> QuantizationRule(pattern=r"^model\\.embed_tokens$", config=QuantizationConfig(bits=8))
        >>> QuantizationRule(pattern=r"^model\\.layers\\.0\\.mlp\\.down_proj$", config=None)
    """

    pattern: str
    config: QuantizationConfig | QuantizationMode | None


class LayerwiseQuantizationConfig(tp.TypedDict, total=False):
    """Regex-driven layer-wise quantization specification.

    Rules are evaluated in order and the first matching rule wins. If no
    rule matches, the optional ``default`` config is used. If ``default``
    is omitted or ``None``, unmatched quantizable modules stay in full
    precision.

    Attributes:
        default: Fallback quantization config applied to quantizable
            modules that do not match any rule.
        rules: Ordered regex rules matched against module paths.

    Example:
        >>> LayerwiseQuantizationConfig(
        ...     default=QuantizationConfig(mode="affine", bits=4, group_size=64),
        ...     rules=[
        ...         QuantizationRule(
        ...             pattern=r"^model\\.embed_tokens$",
        ...             config=QuantizationConfig(mode="affine", bits=8, group_size=64),
        ...         ),
        ...         QuantizationRule(pattern=r"^lm_head$", config=None),
        ...     ],
        ... )
    """

    default: QuantizationConfig | QuantizationMode | None
    rules: list[QuantizationRule]


QuantizationSpec = QuantizationConfig | QuantizationMode | LayerwiseQuantizationConfig
