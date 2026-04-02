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

"""easymlx infra (serving-only).

This package is a lightweight port of EasyDeL's *infrastructure* patterns,
adapted for MLX and inference/serving use-cases only.
"""

from __future__ import annotations

import typing as tp

from .factory import TaskType, register_config, register_module, registry

if tp.TYPE_CHECKING:  # pragma: no cover
    from .base_config import EasyMLXBaseConfig
    from .base_module import EasyMLXBaseModule
    from .etils import (
        LayerwiseQuantizationConfig,
        QuantizationConfig,
        QuantizationMode,
        QuantizationRule,
        QuantizationSpec,
    )
    from .modeling_outputs import CausalLMOutput, GenerateOutput, GreedySearchOutput, SampleOutput
__all__ = (
    "CausalLMOutput",
    "EasyMLXBaseConfig",
    "EasyMLXBaseModule",
    "GenerateOutput",
    "GreedySearchOutput",
    "LayerwiseQuantizationConfig",
    "QuantizationConfig",
    "QuantizationMode",
    "QuantizationRule",
    "QuantizationSpec",
    "SampleOutput",
    "TaskType",
    "register_config",
    "register_module",
    "registry",
)


def __getattr__(name: str):  # pragma: no cover
    """Lazily import module attributes on first access.

    Provides deferred importing of heavy classes to avoid circular imports
    and reduce startup time. Only loads the requested class when it is
    actually accessed.

    Args:
        name: The attribute name being accessed on this module.

    Returns:
        The requested class or object if it is a known lazy-loadable attribute.

    Raises:
        AttributeError: If the requested name is not a known attribute of
            this module.
    """
    if name == "EasyMLXBaseConfig":
        from .base_config import EasyMLXBaseConfig

        return EasyMLXBaseConfig
    if name == "EasyMLXBaseModule":
        from .base_module import EasyMLXBaseModule

        return EasyMLXBaseModule
    if name in {"CausalLMOutput", "GenerateOutput", "GreedySearchOutput", "SampleOutput"}:
        from . import modeling_outputs as _mo

        return getattr(_mo, name)
    if name == "QuantizationConfig":
        from .etils import QuantizationConfig

        return QuantizationConfig

    if name == "LayerwiseQuantizationConfig":
        from .etils import LayerwiseQuantizationConfig

        return LayerwiseQuantizationConfig

    if name == "QuantizationMode":
        from .etils import QuantizationMode

        return QuantizationMode

    if name == "QuantizationRule":
        from .etils import QuantizationRule

        return QuantizationRule

    if name == "QuantizationSpec":
        from .etils import QuantizationSpec

        return QuantizationSpec

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover
    """Return a sorted list of all public names available in this module.

    Combines the names from the module's globals with the explicitly
    declared ``__all__`` tuple so that tab-completion and ``dir()`` calls
    reflect all available symbols, including lazily-loaded ones.

    Returns:
        A sorted list of attribute name strings available in this module.
    """
    return sorted(set(globals().keys()) | set(__all__))
