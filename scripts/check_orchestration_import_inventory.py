#!/usr/bin/env python3
"""Check RM-CONNECTOR-01's reviewed, source-bound import dispositions.

Run with --discover to print observations without assigning classifications.
The default checks docs/development/orchestration-import-inventory.json.
All Git-tracked and nonignored untracked Python files are inspected, including
tests and tooling. No repository package or discovered module is imported.

Both the historic agent_orchestration and current agent_utilities namespaces
are in scope (AGENTS.md, Package Relationships). This is a bounded disposition
check, not a replacement for development/fleet_census.py or dependency_readiness.
Those tools record dependency edges, not individual reviewed dispositions.

Dynamic support: importlib.import_module and builtins.__import__, import aliases,
and simple assignment aliases. Alias propagation is deliberately conservative
and scope-insensitive: rebinding cannot hide a potential import. Only literal
targets are resolved; other expressions remain blockers even with a disposition.
Arbitrary reflection, exec, and custom loader wrappers are not Python imports
recognized by this static contract. Nothing here proves runtime reachability.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

NAMESPACES = ("agent_orchestration", "agent_utilities")
CATEGORIES = ("orchestration", "connector transport", "governance", "obsolete")
INVENTORY = "docs/development/orchestration-import-inventory.json"
LOADERS = {"importlib.import_module", "builtins.__import__"}


def qualified(node: ast.expr, aliases: dict[str, set[str]]) -> set[str]:
    """Return possible qualified names, retaining ambiguous loader bindings."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, set())
    if isinstance(node, ast.Attribute):
        return {f"{name}.{node.attr}" for name in qualified(node.value, aliases)}
    return set()


def imported_aliases(node: ast.AST) -> dict[str, set[str]]:
    if isinstance(node, ast.Import):
        return {
            item.asname or item.name.split(".")[0]: {
                item.name if item.asname else item.name.split(".")[0]
            }
            for item in node.names
        }
    if isinstance(node, ast.ImportFrom) and not node.level:
        return {
            item.asname or item.name: {f"{node.module}.{item.name}"}
            for item in node.names
        }
    return {}


def assignment_aliases(
    node: ast.AST, aliases: dict[str, set[str]]
) -> dict[str, set[str]]:
    if isinstance(node, ast.Assign):
        return {
            target.id: qualified(node.value, aliases)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id: qualified(node.value, aliases)} if node.value else {}
    return {}


def loader_aliases(tree: ast.AST) -> dict[str, set[str]]:
    aliases = {"__import__": {"builtins.__import__"}}
    nodes = list(ast.walk(tree))
    for node in nodes:
        for name, values in imported_aliases(node).items():
            aliases.setdefault(name, set()).update(values)
    # Only loader/module identities propagate, so attribute cycles cannot grow.
    allowed = LOADERS | {"importlib", "builtins"}
    changed = True
    while changed:
        before = repr(sorted((k, sorted(v)) for k, v in aliases.items()))
        for node in nodes:
            for name, values in assignment_aliases(node, aliases).items():
                aliases.setdefault(name, set()).update(values & allowed)
        changed = before != repr(sorted((k, sorted(v)) for k, v in aliases.items()))
    return aliases


def argument(call: ast.Call, position: int, name: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((item.value for item in call.keywords if item.arg == name), None)


def literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value or None
    return None


def dynamic_target(call: ast.Call, loaders: set[str]) -> str | None:
    target = literal(argument(call, 0, "name"))
    if not target:
        return None
    if "builtins.__import__" in loaders:
        level = argument(call, 4, "level")
        if level is not None and not (
            isinstance(level, ast.Constant) and level.value == 0
        ):
            return None
    if not target.startswith("."):
        return target
    package = literal(argument(call, 1, "package"))
    if not package or loaders != {"importlib.import_module"}:
        return None
    try:
        return importlib.util.resolve_name(target, package)
    except (ImportError, ValueError):
        return None


def static_targets(node: ast.AST) -> list[tuple[ast.alias, str, str]]:
    if isinstance(node, ast.Import):
        return [(item, item.name, "") for item in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        return [(item, node.module, item.name) for item in node.names]
    return []


def matching(target: str) -> bool:
    return target.split(".")[0] in NAMESPACES


def source_observations(path: str, raw: bytes) -> tuple[list[dict], list[dict]]:
    tree = ast.parse(raw, filename=path)
    aliases = loader_aliases(tree)
    digest = hashlib.sha256(raw).hexdigest()
    imports: list[dict] = []
    unresolved: list[dict] = []
    for node in ast.walk(tree):
        for alias, target, symbol in static_targets(node):
            if matching(target):
                imports.append(
                    observation(path, alias, digest, "static", target, symbol)
                )
        row = dynamic_observation(path, node, digest, aliases)
        if row is not None:
            (unresolved if row["target"] is None else imports).append(row)
    return imports, unresolved


def dynamic_observation(
    path: str, node: ast.AST, digest: str, aliases: dict[str, set[str]]
) -> dict | None:
    if not isinstance(node, ast.Call):
        return None
    loaders = qualified(node.func, aliases) & LOADERS
    if not loaders:
        return None
    target = dynamic_target(node, loaders)
    if target is not None and not matching(target):
        return None
    return observation(path, node, digest, "dynamic", target, "")


def observation(
    path: str,
    node: ast.Call | ast.alias,
    digest: str,
    kind: str,
    target: str | None,
    symbol: str,
) -> dict:
    return {
        "path": path,
        "line": node.lineno,
        "column": node.col_offset,
        "kind": kind,
        "target": target,
        "symbol": symbol,
        "expression": ast.unparse(node),
        "source_sha256": digest,
    }


def discover(root: Path) -> dict:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    imports, unresolved = [], []
    paths = sorted(set(result.stdout.decode().split("\0")) - {""})
    for relative in paths:
        if not relative.endswith(".py"):
            continue
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source is not a regular in-repository file: {relative}")
        found, unknown = source_observations(relative, path.read_bytes())
        imports.extend(found)
        unresolved.extend(unknown)
    return {
        "schema_version": 1,
        "namespaces": list(NAMESPACES),
        "imports": sorted(imports, key=identity),
        "unresolved": sorted(unresolved, key=identity),
    }


def identity(row: dict) -> str:
    return f"{row['path']}:{row['line']}:{row['column']}:{row['kind']}:{row['target']}:{row['symbol']}"


def review_errors(row: dict) -> list[str]:
    errors = []
    if row.get("classification") not in CATEGORIES:
        errors.append("missing or invalid classification")
    for field in ("evidence", "reviewer"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"missing {field}")
    return errors


def validate(actual: dict, inventory: dict) -> list[str]:
    errors = []
    for field in ("schema_version", "namespaces", "unresolved"):
        if inventory.get(field) != actual[field]:
            errors.append(f"stale or missing {field}")
    expected = {identity(row): row for row in actual["imports"]}
    seen: set[str] = set()
    for entry in inventory["imports"]:
        key = identity(entry)
        if key in seen:
            errors.append(f"duplicate classification: {key}")
        seen.add(key)
        errors.extend(entry_errors(entry, expected.get(key)))
    errors.extend(
        f"missing classification: {key}" for key in sorted(expected.keys() - seen)
    )
    errors.extend(
        f"unresolved dynamic target: {identity(row)} ({row['expression']})"
        for row in actual["unresolved"]
    )
    return errors


def entry_errors(entry: dict, expected: dict | None) -> list[str]:
    key = identity(entry)
    errors = [f"{message}: {key}" for message in review_errors(entry)]
    observed = {
        k: v
        for k, v in entry.items()
        if k not in {"classification", "evidence", "reviewer"}
    }
    if observed != expected:
        errors.append(f"stale classification/source evidence: {key}")
    return errors


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args(argv)
    try:
        actual = discover(args.root)
        if args.discover:
            print(json.dumps(actual, indent=2))
            return 1 if actual["unresolved"] else 0
        path = args.inventory or args.root / INVENTORY
        errors = validate(
            actual, json.loads(path.read_text(), object_pairs_hook=unique_fields)
        )
    except (
        OSError,
        ValueError,
        SyntaxError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"import inventory: CANNOT VERIFY: {exc}", file=sys.stderr)
        return 2
    for error in errors:
        print(f"import inventory: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"import inventory: OK ({len(actual['imports'])} classified imports)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
