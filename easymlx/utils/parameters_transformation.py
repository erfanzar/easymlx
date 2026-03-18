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

"""Parameter transformation helpers for checkpoint conversion.

Defines a three-level converter hierarchy (tensor, state-dict, model)
used during Hugging Face-to-easymlx checkpoint conversion.  The default
implementations are identity converters that pass values through
unchanged; subclasses override :meth:`convert` for real transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorConverter:
    """Identity tensor converter placeholder.

    Serves as the base converter for individual tensor values.
    The default implementation returns the value unchanged; subclasses
    should override :meth:`convert` to apply dtype casting, transposition,
    or other per-tensor transformations.
    """

    def convert(self, value: Any) -> Any:
        """Convert a single tensor value.

        Args:
            value: The tensor (or arbitrary object) to convert.

        Returns:
            The converted value. The default implementation returns
            ``value`` unchanged.
        """
        return value


@dataclass
class StateDictConverter:
    """Identity state-dict converter placeholder.

    Applies a :class:`TensorConverter` to every value in a state
    dictionary.  Subclasses may override :meth:`convert` to add key
    renaming or pruning logic.

    Attributes:
        tensor_converter: The per-tensor converter applied to each value.
    """

    tensor_converter: TensorConverter = field(default_factory=TensorConverter)

    def convert(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an entire state dictionary.

        Args:
            state_dict: Mapping of parameter names to tensor values.

        Returns:
            A new dictionary with each value transformed by
            :attr:`tensor_converter`.
        """
        return {name: self.tensor_converter.convert(value) for name, value in state_dict.items()}


@dataclass
class ModelConverter:
    """Identity model converter placeholder.

    Top-level converter that delegates to a :class:`StateDictConverter`.
    Subclasses may override :meth:`convert` to add model-level
    pre-processing or post-processing steps.

    Attributes:
        state_dict_converter: The state-dict converter used for the
            actual parameter transformation.
    """

    state_dict_converter: StateDictConverter = field(default_factory=StateDictConverter)

    def convert(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a model's full state dictionary.

        Args:
            state_dict: Mapping of parameter names to tensor values.

        Returns:
            The converted state dictionary.
        """
        return self.state_dict_converter.convert(state_dict)


__all__ = ("ModelConverter", "StateDictConverter", "TensorConverter")
