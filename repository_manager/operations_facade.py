"""Reflection-only view of ``Git`` for per-operation MCP tool registration.

``register_tool_surface`` builds one tool per public client method from the method
signatures. Several ``Git`` methods carry keyword-only private parameters
(``_pinned``, ``_boundary_assertion``, ``_directory_fd`` ...) that hold live handles
and callables a tool schema cannot describe. The facade class repeats each public
method with those private parameters removed; tool calls still run on the real
``Git`` instance returned by the client factory.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any


def _stub(name: str, method: Any) -> Any:
    signature = inspect.signature(method)
    try:
        hints = typing.get_type_hints(method)
    except Exception:  # unresolved forward references: keep the raw annotations
        hints = dict(getattr(method, "__annotations__", {}))
    parameters = [
        parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
        for parameter in signature.parameters.values()
        if not parameter.name.startswith("_") or parameter.name == "self"
    ]

    def stub(self: Any, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, name)(*args, **kwargs)

    stub.__name__ = name
    stub.__doc__ = method.__doc__
    stub.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=hints.get("return", signature.return_annotation),
    )
    stub.__annotations__ = {
        p.name: p.annotation for p in parameters if p.name != "self"
    } | {"return": hints.get("return", signature.return_annotation)}
    return stub


def operations_facade(cls: type, name: str = "GitOperations") -> type:
    """A class with ``cls``'s public methods minus their private parameters."""
    members = {
        method_name: _stub(method_name, method)
        for method_name, method in inspect.getmembers(cls, inspect.isfunction)
        if not method_name.startswith("_")
    }
    return type(name, (), members)
