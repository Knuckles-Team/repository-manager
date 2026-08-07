#!/usr/bin/env python

import importlib
import inspect
from typing import Any

__all__: list[str] = []

CORE_MODULES: list[str] = ["repository_manager.repository_manager"]

OPTIONAL_MODULES = {
    "repository_manager.agent_server": "agent",
    "repository_manager.mcp_server": "mcp",
}

# D-SKC-4: a submodule's own basename (e.g. "mcp_server") is reserved and is
# never bound as a package-global by ``_expose_members``, even though
# ``mcp_server.py`` happens to export a public function of the exact same
# name (its CLI entrypoint, ``def mcp_server() -> None``, wired directly via
# ``[project.scripts]`` module:attr -- never through this package's
# namespace). Binding that function into ``repository_manager`` globals
# would SHADOW the ``repository_manager.mcp_server`` MODULE for any later
# ``from repository_manager import mcp_server``, and because the optional
# modules are only exposed lazily (the first time ANY missing attribute is
# looked up via ``__getattr__``, not only an ``mcp_server``-named one),
# whether the shadowing had already happened depended on which attributes
# an earlier import/test in the SAME process touched first -- a gate whose
# verdict depended on invocation order rather than on the code.
_RESERVED_SUBMODULE_NAMES = frozenset(
    module_name.rsplit(".", 1)[-1] for module_name in (*CORE_MODULES, *OPTIONAL_MODULES)
)


def _expose_members(module):
    """Expose public classes and functions from a module into globals and __all__."""
    for name, obj in inspect.getmembers(module):
        if name in _RESERVED_SUBMODULE_NAMES:
            continue
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith(
            "_"
        ):
            globals()[name] = obj
            if name not in __all__:
                __all__.append(name)


# Eagerly import core modules (keeps API wrappers fast & light)
for module_name in CORE_MODULES:
    if module_name:
        module = importlib.import_module(module_name)
        _expose_members(module)

# Dynamic/lazy loading of optional modules (agent_server, mcp_server)
_loaded_optional_modules: dict[str, Any] = {}


def _import_module_safely(module_name: str):
    """Try to import a module and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def __getattr__(name: str) -> Any:
    # Handle availability flags dynamically without eager imports
    if name == "_MCP_AVAILABLE":
        mcp_key = next((k for k in OPTIONAL_MODULES if "mcp_server" in k), None)
        if mcp_key:
            return _import_module_safely(mcp_key) is not None
        return False
    if name == "_AGENT_AVAILABLE":
        agent_key = next((k for k in OPTIONAL_MODULES if "agent_server" in k), None)
        if agent_key:
            return _import_module_safely(agent_key) is not None
        return False

    # Check optional modules
    for module_name in OPTIONAL_MODULES:
        if module_name not in _loaded_optional_modules:
            module = _import_module_safely(module_name)
            if module is not None:
                _loaded_optional_modules[module_name] = module
                _expose_members(module)

        module = _loaded_optional_modules.get(module_name)
        if module is not None and hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
