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

"""Lazy import helpers for easymlx.

Provides a :class:`LazyModule` that defers sub-module imports until first
attribute access, keeping ``import easymlx`` fast.  Adapted from the
HuggingFace ``transformers`` / EasyDeL lazy module pattern.
"""

import importlib
import importlib.util
import os
import typing as tp
from itertools import chain
from types import ModuleType


class LazyModule(ModuleType):
    """Module subclass that defers sub-module imports until first access.

    Used as the top-level ``__init__`` module so that ``import easymlx``
    is fast and only imports sub-packages on demand.

    Attributes:
        _modules: Set of known sub-module names.
        _class_to_module: Mapping from exported class name to its sub-module.
        _import_structure: Flattened mapping of module -> list of exported names.
        _objects: Extra objects injected at construction time.
    """

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, object] | None = None,
    ):
        """Initialize the lazy module proxy.

        Args:
            name: The module name exposed to callers.
            module_file: The ``__file__`` of the real package ``__init__.py``.
            import_structure: Mapping of ``"submodule.path"`` to list of
                exported names from that submodule.
            module_spec: Optional ``ModuleSpec`` from the real package.
            extra_objects: Extra objects (e.g. ``__version__``) to expose
                without any import.
        """
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> list[str]:
        """Return all known attributes for autocompletion.

        Returns:
            Combined list of standard module attributes and lazy exports.
        """
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> tp.Any:
        """Lazily resolve an attribute by importing its sub-module.

        Args:
            name: The attribute name to resolve.

        Returns:
            The resolved attribute (module or object from a module).

        Raises:
            AttributeError: If *name* is not in the import structure.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        elif name in self._modules:
            value = self._get_module(name)
        else:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        """Import a sub-module by dotted name.

        Args:
            module_name: Dotted path relative to this package.

        Returns:
            The imported module.

        Raises:
            RuntimeError: If the import fails.
        """
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the "
                f"following error (look up to see its traceback):\n{exc}"
            ) from exc

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


def is_package_available(package_name: str) -> bool:
    """Check whether a Python package is importable.

    Args:
        package_name: The package name (e.g. ``"numpy"``).

    Returns:
        ``True`` if the package can be found, ``False`` otherwise.
    """
    return importlib.util.find_spec(package_name.replace("-", "_")) is not None
