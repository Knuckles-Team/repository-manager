#!/usr/bin/env python
# coding: utf-8

import importlib
import inspect

# List of modules to import from
MODULES = [
    "repository_manager.repository_manager",
    "repository_manager.repository_manager_mcp",
]

# Initialize __all__ to expose all public classes and functions
__all__ = []

# Dynamically import all classes and functions from the specified modules
for module_name in MODULES:
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module):
        # Include only classes and functions, excluding private (starting with '_')
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith(
            "_"
        ):
            globals()[name] = obj
            __all__.append(name)

"""
report-manager

Manage your git projects
"""
