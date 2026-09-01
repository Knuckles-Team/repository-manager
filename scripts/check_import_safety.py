#!/usr/bin/env python3
"""Cross-platform import-safety gate.

Bare invocation imports every module under a package in the real interpreter,
failing on any exception raised at import time.  This is the highest-yield
check for the "works on Linux, breaks on Windows" class of defect.

``--simulate-windows`` cannot safely be implemented by changing
``sys.platform``/``os.name`` or by poisoning ``fcntl``.  The former breaks
pathlib and other process-global platform dispatch, while the latter reports a
correctly guarded import as a false positive (the guard still sees the real
POSIX platform).  Simulated mode therefore performs two complementary checks:

* a conservative AST pass over every checkout/standalone source file, which
  rejects POSIX-only imports unless a Windows-false platform branch or an
  explicit ``ImportError`` fallback provably protects them; and
* a native import of the target package root and each standalone script, which
  still exercises normal import-time wiring without pretending that a Linux
  process is Windows.

Run the gate on an actual Windows runner as well.  Native Windows execution
remains the authoritative check for all runtime branches; the AST pass is a
deterministic local preflight, not a platform emulator.

This script is intentionally stdlib-only so it can run in the fast
pre-commit tier with no environment sync.

Usage::

    python3 scripts/check_import_safety.py --package agent_utilities
    python3 scripts/check_import_safety.py --package agent_utilities --simulate-windows
    python3 scripts/check_import_safety.py --package repository_manager --script scripts/security_contract.py --simulate-windows
    python3 scripts/check_import_safety.py --package agent_utilities --exclude agent_utilities.some.heavy_optional_module
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import pkgutil
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

# Stdlib modules that exist only on POSIX.  An import of any of these must be
# behind a platform branch or an ImportError fallback before Windows can load
# the containing module.
POSIX_ONLY_STDLIB_MODULES = ("fcntl", "termios", "pwd", "resource", "grp")


def _checkout_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _package_parts(package_name: str) -> tuple[str, ...] | None:
    parts = tuple(package_name.split("."))
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return parts


def _checkout_package_path(package_name: str) -> Path | None:
    parts = _package_parts(package_name)
    if parts is None:
        return None
    package_path = _checkout_root().joinpath(*parts)
    if package_path.is_dir() or package_path.with_suffix(".py").is_file():
        return package_path
    return None


def ensure_package_discoverable(package_name: str) -> bool:
    """Make a checkout-owned package authoritative for this gate.

    Hooks execute this file with ``scripts/`` as ``sys.path[0]``.  If the
    package exists in this checkout, its parent is moved to the front of the
    import path so a stale installed distribution cannot be certified in its
    place.  If this checkout does not own the package, the interpreter's
    normal installed-package resolution is left untouched.

    The module cache is cleared for the package prefix because a caller may
    have imported a conflicting installed package before invoking this helper
    (the command-line gate itself does not).  This is deliberately scoped to
    the requested package and its children.
    """
    if _checkout_package_path(package_name) is None:
        return False

    root_text = str(_checkout_root())
    sys.path[:] = [root_text, *(entry for entry in sys.path if str(entry) != root_text)]
    importlib.invalidate_caches()
    prefix = f"{package_name}."
    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(prefix):
            del sys.modules[loaded_name]
    return True


def _source_files_for_path(package_path: Path) -> list[Path]:
    if package_path.is_file():
        return [package_path]
    return sorted(
        path
        for path in package_path.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _source_files_for_installed_package(package_name: str) -> list[Path]:
    """Resolve source files from an installed distribution's import spec."""
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return []
    if spec is None:
        return []
    search_paths = [Path(path) for path in (spec.submodule_search_locations or ())]
    if search_paths:
        files: list[Path] = []
        for search_path in search_paths:
            files.extend(_source_files_for_path(search_path))
        return sorted(set(files))
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        origin = Path(spec.origin)
        return [origin] if origin.is_file() else []
    return []


def source_files_for_package(package_name: str) -> list[Path]:
    """Resolve source files for static Windows import analysis.

    A checkout-owned package is resolved first by
    :func:`ensure_package_discoverable`; otherwise an installed package's
    import spec supplies its origin/search path.  The fallback keeps the gate
    useful for packages that are intentionally tested from an installed
    environment without allowing that environment to override a checkout.
    """
    package_path = _checkout_package_path(package_name)
    if package_path is not None:
        return _source_files_for_path(package_path)
    return _source_files_for_installed_package(package_name)


def _attribute_platform_value(node: ast.Attribute) -> str | None:
    if not isinstance(node.value, ast.Name):
        return None
    if node.value.id == "sys" and node.attr == "platform":
        return "win32"
    if node.value.id == "os" and node.attr == "name":
        return "nt"
    return None


def _call_platform_value(node: ast.Call) -> str | None:
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "platform"
        and node.func.attr == "system"
        and not node.args
        and not node.keywords
    ):
        return "Windows"
    return None


def _windows_platform_value(node: ast.AST) -> str | None:
    """Return the platform probe represented by a small, known AST shape."""
    if isinstance(node, ast.Attribute):
        return _attribute_platform_value(node)
    if isinstance(node, ast.Call):
        return _call_platform_value(node)
    return None


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _constant_strings(node: ast.AST) -> set[str] | None:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_constant_string(element) for element in node.elts]
        # An `all(... is not None)` guard does not narrow the element type, so
        # build the narrowed set in the comprehension and gate on membership.
        # (Counting instead would be wrong: a duplicated literal collapses in
        # the set and used to be accepted.)
        strings = {value for value in values if value is not None}
        return None if None in values else strings
    value = _constant_string(node)
    return {value} if value is not None else None


def _windows_bool_op_condition(node: ast.BoolOp) -> bool | None:
    values = [_windows_condition(value) for value in node.values]
    if isinstance(node.op, ast.And):
        if False in values:
            return False
        return True if all(value is True for value in values) else None
    if True in values:
        return True
    return False if all(value is False for value in values) else None


def _windows_call_condition(node: ast.Call) -> bool | None:
    if (
        not isinstance(node.func, ast.Attribute)
        or node.func.attr not in {"startswith", "endswith"}
        or len(node.args) != 1
        or node.keywords
    ):
        return None
    platform_value = _windows_platform_value(node.func.value)
    prefix = _constant_string(node.args[0])
    if platform_value is None or prefix is None:
        return None
    if node.func.attr == "startswith":
        return platform_value.startswith(prefix)
    return platform_value.endswith(prefix)


_COMPARE_OPERATION_EVALUATORS: dict[
    type[ast.cmpop], Callable[[str, set[str]], bool]
] = {
    ast.Eq: lambda platform_value, compared_values: (
        platform_value in compared_values and len(compared_values) == 1
    ),
    ast.NotEq: lambda platform_value, compared_values: (
        platform_value not in compared_values
    ),
    ast.In: lambda platform_value, compared_values: platform_value in compared_values,
    ast.NotIn: lambda platform_value, compared_values: (
        platform_value not in compared_values
    ),
}


def _windows_compare_condition(node: ast.Compare) -> bool | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    platform_value = _windows_platform_value(node.left)
    compared_values = _constant_strings(node.comparators[0])
    if platform_value is None or compared_values is None:
        return None
    evaluator = _COMPARE_OPERATION_EVALUATORS.get(type(node.ops[0]))
    if evaluator is None:
        return None
    return evaluator(platform_value, compared_values)


# The exact AST class used as the key determines each handler's narrower input
# type, so this table is intentionally heterogeneous at the callable boundary.
_WINDOWS_CONDITION_DISPATCH: dict[type[ast.AST], Callable[..., bool | None]] = {
    ast.BoolOp: _windows_bool_op_condition,
    ast.Call: _windows_call_condition,
    ast.Compare: _windows_compare_condition,
}


def _windows_condition_leaf(node: ast.AST) -> bool | None:
    """Handle the non-dispatchable leaf shapes: literal bool, TYPE_CHECKING, not."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _windows_condition(node.operand)
        return None if value is None else not value
    return None


def _windows_condition(node: ast.AST) -> bool | None:
    """Evaluate a conservative subset of platform conditions on Windows."""
    leaf_types = (ast.Constant, ast.Name, ast.UnaryOp)
    if isinstance(node, leaf_types):
        return _windows_condition_leaf(node)
    handler = _WINDOWS_CONDITION_DISPATCH.get(type(node))
    return handler(node) if handler is not None else None


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    return bool(names) and all(
        name in {"ImportError", "ModuleNotFoundError"} for name in names
    )


class _WindowsImportVisitor:
    """Find POSIX-only imports not proven safe on Windows by source syntax."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def scan(self, tree: ast.Module) -> list[str]:
        self._visit_statements(tree.body, reaches_windows=True, guarded=False)
        return self.findings

    def _visit_statements(
        self,
        statements: list[ast.stmt],
        *,
        reaches_windows: bool,
        guarded: bool,
    ) -> None:
        for statement in statements:
            self._visit(statement, reaches_windows=reaches_windows, guarded=guarded)

    def _visit_import(
        self,
        node: ast.Import | ast.ImportFrom,
        *,
        reaches_windows: bool,
        guarded: bool,
    ) -> None:
        if not (reaches_windows and not guarded):
            return
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        for name in names:
            top_level = name.split(".", 1)[0]
            if top_level in POSIX_ONLY_STDLIB_MODULES:
                self.findings.append(
                    f"{self.path}:{node.lineno}: unguarded POSIX-only import {top_level!r}"
                )

    def _visit_if(self, node: ast.If, *, reaches_windows: bool, guarded: bool) -> None:
        condition = _windows_condition(node.test)
        body_reaches = reaches_windows and condition is not False
        else_reaches = reaches_windows and condition is not True
        self._visit_statements(node.body, reaches_windows=body_reaches, guarded=guarded)
        self._visit_statements(
            node.orelse, reaches_windows=else_reaches, guarded=guarded
        )

    def _visit_try(
        self, node: ast.Try, *, reaches_windows: bool, guarded: bool
    ) -> None:
        catches_import_error = any(
            _catches_import_error(handler) for handler in node.handlers
        )
        self._visit_statements(
            node.body,
            reaches_windows=reaches_windows,
            guarded=guarded or catches_import_error,
        )
        for handler in node.handlers:
            self._visit_statements(
                handler.body, reaches_windows=reaches_windows, guarded=guarded
            )
        self._visit_statements(
            node.orelse, reaches_windows=reaches_windows, guarded=guarded
        )
        self._visit_statements(
            node.finalbody, reaches_windows=reaches_windows, guarded=guarded
        )

    def _visit(self, node: ast.AST, *, reaches_windows: bool, guarded: bool) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._visit_import(node, reaches_windows=reaches_windows, guarded=guarded)
            return

        if isinstance(node, ast.If):
            self._visit_if(node, reaches_windows=reaches_windows, guarded=guarded)
            return

        if isinstance(node, ast.Try):
            self._visit_try(node, reaches_windows=reaches_windows, guarded=guarded)
            return

        for child in ast.iter_child_nodes(node):
            self._visit(child, reaches_windows=reaches_windows, guarded=guarded)


def analyze_windows_imports(paths: list[Path]) -> list[tuple[str, str]]:
    """Return ``(path, finding)`` pairs for unsafe POSIX-only imports."""
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        for finding in _WindowsImportVisitor(path).scan(tree):
            failures.append((str(path), finding))
    return failures


def discover_modules(package_name: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (discovered module names, walk-time failures).

    ``pkgutil.walk_packages`` imports each subpackage to read its path.  An
    explicit ``onerror`` callback records a failed subpackage instead of
    silently dropping its entire subtree.
    """
    ensure_package_discoverable(package_name)
    package = importlib.import_module(package_name)
    names = [package_name]
    walk_failures: list[tuple[str, str]] = []

    def _on_walk_error(broken_name: str) -> None:
        exc = sys.exc_info()[1]
        detail = (
            f"{type(exc).__name__}: {exc}" if exc else "import failed while walking"
        )
        walk_failures.append((broken_name, detail))

    walkable_path = getattr(package, "__path__", None)
    if walkable_path is not None:
        for info in pkgutil.walk_packages(
            walkable_path,
            prefix=f"{package_name}.",
            onerror=_on_walk_error,
        ):
            names.append(info.name)
    return sorted(names), walk_failures


def import_standalone_script(path: Path) -> None:
    """Import a standalone .py file that package walking cannot see."""
    spec = importlib.util.spec_from_file_location(
        f"_import_safety_script_check__{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--package",
        required=True,
        help="top-level package/module name to walk, e.g. agent_utilities",
    )
    parser.add_argument(
        "--script",
        action="append",
        default=[],
        metavar="PATH",
        help="standalone .py file to import directly (repeatable)",
    )
    parser.add_argument(
        "--simulate-windows",
        action="store_true",
        help="run conservative AST platform analysis plus native imports",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DOTTED_PREFIX",
        help="dotted module name/prefix to skip in the native module walk",
    )
    return parser


def _resolve_mode(simulate_windows: bool) -> str:
    if simulate_windows:
        return "simulated-windows"
    return "windows" if sys.platform == "win32" else "native"


def _print_fatal_import_failure(package: str, exc: BaseException) -> None:
    print(f"FATAL: could not import {package!r} at all: {type(exc).__name__}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _run_simulated_windows(
    package: str,
) -> tuple[list[tuple[str, str]], int, Exception | None]:
    """Resolve before importing so a local checkout wins over an installed one."""
    ensure_package_discoverable(package)
    source_paths = source_files_for_package(package)
    failures = analyze_windows_imports(source_paths)
    checked = len(source_paths)
    try:
        importlib.import_module(package)
    except Exception as exc:
        return failures, checked, exc
    return failures, checked, None


def _run_native_walk(
    package: str, is_excluded: Callable[[str], bool]
) -> tuple[list[tuple[str, str]], int, Exception | None]:
    try:
        module_names, walk_failures = discover_modules(package)
    except Exception as exc:
        return [], 0, exc

    failures: list[tuple[str, str]] = [
        (
            name,
            f"{err} (package failed to import WHILE WALKING -- its children could not be enumerated at all)",
        )
        for name, err in walk_failures
        if not is_excluded(name)
    ]
    already_recorded = {name for name, _ in failures}
    checked = 0
    for name in module_names:
        if is_excluded(name) or name in already_recorded:
            continue
        checked += 1
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    checked += len(already_recorded)
    return failures, checked, None


def _check_standalone_scripts(
    scripts: list[str], *, simulate_windows: bool
) -> tuple[list[tuple[str, str]], int]:
    failures: list[tuple[str, str]] = []
    checked = 0
    for script_path in scripts:
        path = Path(script_path)
        checked += 1
        if not path.is_file():
            failures.append(
                (str(path), "FileNotFoundError: standalone script does not exist")
            )
            continue
        if simulate_windows:
            failures.extend(analyze_windows_imports([path]))
        try:
            import_standalone_script(path)
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
    return failures, checked


def _report_results(
    *,
    mode: str,
    package: str,
    checked: int,
    script_count: int,
    failures: list[tuple[str, str]],
    simulate_windows: bool,
) -> int:
    standalone_suffix = (
        f" (+{script_count} standalone script(s))" if script_count else ""
    )
    print(
        f"import-safety[{mode}]: checked {checked} modules/scripts under {package!r}"
        f"{standalone_suffix}"
    )
    if failures:
        print(f"import-safety[{mode}]: {len(failures)} finding(s) FAILED:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1

    success_detail = (
        "passed AST/native checks" if simulate_windows else "imported cleanly"
    )
    print(f"import-safety[{mode}]: all {checked} modules {success_detail}")
    return 0


def main() -> int:
    args = _build_arg_parser().parse_args()

    excluded = tuple(args.exclude)

    def is_excluded(name: str) -> bool:
        return any(
            name == prefix or name.startswith(prefix + ".") for prefix in excluded
        )

    mode = _resolve_mode(args.simulate_windows)

    if args.simulate_windows:
        failures, checked, fatal = _run_simulated_windows(args.package)
    else:
        failures, checked, fatal = _run_native_walk(args.package, is_excluded)

    if fatal is not None:
        _print_fatal_import_failure(args.package, fatal)
        return 1

    script_failures, script_checked = _check_standalone_scripts(
        args.script, simulate_windows=args.simulate_windows
    )
    failures = [*failures, *script_failures]
    checked += script_checked

    return _report_results(
        mode=mode,
        package=args.package,
        checked=checked,
        script_count=len(args.script),
        failures=failures,
        simulate_windows=args.simulate_windows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
