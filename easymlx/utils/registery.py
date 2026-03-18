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

"""Small registry helper mirroring EasyDeL's module name.

Provides a lightweight dictionary-based registry for mapping string
names to arbitrary objects (model classes, converters, etc.).
"""

from __future__ import annotations


class Registry(dict[str, object]):
    """Tiny named-object registry.

    A simple ``dict`` subclass that maps string names to objects.
    Useful for registering model classes, converters, or other
    components by name.

    Example:
        >>> registry = Registry()
        >>> registry.register("my_model", MyModelClass)
        >>> registry["my_model"]
        <class 'MyModelClass'>
    """

    def register(self, name: str, value: object) -> object:
        """Register an object under the given name.

        Args:
            name: The key to associate with the object.
            value: The object to register.

        Returns:
            The registered object (``value``), enabling decorator-style
            usage.
        """
        self[name] = value
        return value


__all__ = "Registry"
