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

"""Executable tool registry for the easymlx eSurge API server.

This module provides ``ToolRegistry``, a thread-safe container for registering,
listing, and executing tool definitions and their handlers. Tools follow the
OpenAI function-calling format and can be registered from dictionaries, callables,
or ``ToolSpec`` dataclass instances.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


def _default_parameters() -> dict[str, Any]:
    """Return the default empty JSON Schema parameters for a tool.

    Returns:
        A dictionary with ``type`` set to ``object`` and empty ``properties``.
    """
    return {
        "type": object,
        "properties": {},
    }


@dataclass(slots=True)
class ToolSpec:
    """Tool metadata plus an optional local execution handler."""

    name: str
    handler: Callable[..., Any] | None = None
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=_default_parameters)
    strict: bool = False

    def as_openai_tool(self) -> dict[str, Any]:
        """Convert this tool specification to OpenAI function-calling format.

        Returns:
            A dictionary with ``"type": "function"`` and a nested ``"function"``
            object containing the tool name, parameters, and optional description.
        """
        function_payload: dict[str, Any] = {
            "name": self.name,
            "parameters": dict(self.parameters),
        }
        if self.description:
            function_payload["description"] = self.description
        if self.strict:
            function_payload["strict"] = True
        return {
            "type": "function",
            "function": function_payload,
        }


class ToolRegistry:
    """Thread-safe registry for tool definitions and execution handlers."""

    def __init__(self, tools: dict[str, Any] | list[Any] | None = None):
        """Initialize the registry with an optional initial set of tools.

        Args:
            tools: Initial tool definitions to register. Accepts a dictionary
                mapping names to ``ToolSpec``/callable/dict values, or a list
                of ``ToolSpec``/dict items. ``None`` creates an empty registry.
        """
        self._lock = RLock()
        self._tools: dict[str, ToolSpec] = {}
        if tools:
            self.load_tools(tools)

    def load_tools(self, tools: dict[str, Any] | list[Any]) -> None:
        """Bulk-register tools from a dictionary or list.

        Args:
            tools: Tool definitions to register. Accepts a dictionary mapping
                names to ``ToolSpec``, callables, or OpenAI-format dicts; or
                a list of ``ToolSpec`` or OpenAI-format dict items.

        Raises:
            TypeError: If a tool payload has an unsupported type.
            ValueError: If a list-format tool definition is missing a name.
        """
        if isinstance(tools, dict):
            items = tools.items()
            for name, value in items:
                if isinstance(value, ToolSpec):
                    spec = value
                    if spec.name != name:
                        spec = ToolSpec(
                            name=name,
                            handler=spec.handler,
                            description=spec.description,
                            parameters=dict(spec.parameters),
                            strict=spec.strict,
                        )
                    self._store(spec)
                elif callable(value):
                    self.register_tool(name, value)
                elif isinstance(value, dict):
                    payload = dict(value.get("function") or value)
                    self.register_tool(
                        str(payload.get(name) or name),
                        value.get("handler"),
                        description=payload.get("description"),
                        parameters=dict(payload.get("parameters") or _default_parameters()),
                        strict=bool(payload.get("strict", False)),
                    )
                else:
                    raise TypeError(f"Unsupported tool registration payload for {name!r}: {type(value)!r}")
            return

        for item in tools:
            if isinstance(item, ToolSpec):
                self._store(item)
            elif isinstance(item, dict):
                payload = dict(item.get("function") or item)
                name = payload.get(name)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Tool definitions must include a function name")
                self.register_tool(
                    name.strip(),
                    item.get("handler"),
                    description=payload.get("description"),
                    parameters=dict(payload.get("parameters") or _default_parameters()),
                    strict=bool(payload.get("strict", False)),
                )
            else:
                raise TypeError(f"Unsupported tool registration payload: {type(item)!r}")

    def _store(self, spec: ToolSpec) -> None:
        """Store a tool spec in the registry under a write lock.

        Args:
            spec: The tool specification to store.
        """
        with self._lock:
            self._tools[spec.name] = spec

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any] | None,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> ToolSpec:
        """Register a single tool with an optional execution handler.

        Args:
            name: Unique tool name.
            handler: Callable that implements the tool logic. If ``None``, the
                tool is registered as metadata-only (execution will raise
                ``NotImplementedError``).
            description: Human-readable description of the tool.
            parameters: JSON Schema for the tool's parameters.
            strict: Whether strict parameter validation is required.

        Returns:
            The created ``ToolSpec`` instance.
        """
        spec = ToolSpec(
            name=name,
            handler=handler,
            description=description,
            parameters=dict(parameters or _default_parameters()),
            strict=strict,
        )
        self._store(spec)
        return spec

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools in OpenAI function-calling format.

        Returns:
            A sorted list of OpenAI-format tool definition dictionaries.
        """
        with self._lock:
            return [self._tools[name].as_openai_tool() for name in sorted(self._tools)]

    def get_spec(self, name: str) -> ToolSpec | None:
        """Look up a tool specification by name.

        Args:
            name: The tool name to look up.

        Returns:
            The ``ToolSpec`` if found, or ``None`` if no tool with that name
            is registered.
        """
        with self._lock:
            return self._tools.get(name)

    @staticmethod
    def normalize_arguments(arguments: str | dict[str, Any] | list[Any] | None) -> Any:
        """Normalize tool arguments to a Python dict or list.

        Args:
            arguments: Raw arguments as a JSON string, dict, list, or ``None``.

        Returns:
            Parsed arguments as a dict or list. Returns an empty dict for
            ``None`` or empty-string input.

        Raises:
            TypeError: If ``arguments`` is not a string, dict, list, or ``None``.
            ValueError: If ``arguments`` is a string containing invalid JSON.
        """
        if arguments is None:
            return {}
        if isinstance(arguments, (dict, list)):
            return arguments
        if not isinstance(arguments, str):
            raise TypeError("Tool arguments must be a JSON string, dict, list, or None")

        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool argument JSON: {exc.msg}") from exc

    async def execute_tool(self, name: str, arguments: str | dict[str, Any] | list[Any] | None) -> tuple[Any, Any]:
        """Execute a registered tool by name with the given arguments.

        Normalizes arguments, invokes the tool's handler, and awaits the
        result if the handler is asynchronous.

        Args:
            name: The name of the registered tool to execute.
            arguments: Arguments to pass to the tool handler. Accepts JSON
                strings, dicts, lists, or ``None``.

        Returns:
            A tuple of ``(normalized_arguments, result)`` where
            ``normalized_arguments`` is the parsed argument value and
            ``result`` is the handler's return value.

        Raises:
            KeyError: If no tool with the given name is registered.
            NotImplementedError: If the tool has no execution handler.
            TypeError: If arguments cannot be normalized.
            ValueError: If arguments contain invalid JSON.
        """
        spec = self.get_spec(name)
        if spec is None:
            raise KeyError("name")
        if spec.handler is None:
            raise NotImplementedError(f"Tool {name!r} is registered without an execution handler")

        normalized_arguments = self.normalize_arguments(arguments)
        if isinstance(normalized_arguments, dict):
            result = spec.handler(**normalized_arguments)
        else:
            result = spec.handler(normalized_arguments)
        if inspect.isawaitable(result):
            result = await result
        return normalized_arguments, result
