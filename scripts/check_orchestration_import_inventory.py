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
and scope-insensitive: rebinding cannot hide a potential import. Literal targets,
single-assignment module constants/re-exports, and literal iteration are resolved
without execution. Other expressions require explicit reviewed dynamic-boundary
dispositions tied to the complete source digest. Discovery still reports those
runtime targets as unknown; a disposition classifies the call's responsibility,
not its possible module names. Missing or stale dispositions fail closed.
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
STRING_BINDING_FIELDS = {
    "MatchAs": "name",
    "MatchStar": "name",
    "MatchMapping": "rest",
    "ExceptHandler": "name",
    "Global": "names",
    "Nonlocal": "names",
    "TypeVar": "name",
    "ParamSpec": "name",
    "TypeVarTuple": "name",
}


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


def dynamic_target(
    call: ast.Call, loaders: set[str], target: str | None = None
) -> str | None:
    target = target or literal(argument(call, 0, "name"))
    if not target:
        return None
    if "builtins.__import__" in loaders and not absolute_builtin(call):
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


def absolute_builtin(call: ast.Call) -> bool:
    level = argument(call, 4, "level")
    return level is None or (isinstance(level, ast.Constant) and level.value == 0)


def binding_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [item.asname or item.name.split(".")[0] for item in node.names]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    return string_bindings(node)


def string_bindings(node: ast.AST) -> list[str]:
    """Captures/declarations use string fields rather than Name(Store) nodes.

    Type-parameter node names keep the scanner usable on Python 3.11, where
    those AST classes do not exist yet. Declarations and deletions conservatively
    invalidate constant proof, even when they do not assign a replacement value.
    """
    field = STRING_BINDING_FIELDS.get(type(node).__name__)
    value = getattr(node, field, None) if field else None
    if isinstance(value, str):
        return [value]
    return value if isinstance(value, list) else []


def constant_bindings(tree: ast.Module, name: str) -> list[ast.AST]:
    """Wildcard imports may replace any name; they invalidate unique binding."""
    return [node for node in ast.walk(tree) if may_bind_name(node, name)]


def may_bind_name(node: ast.AST, name: str) -> bool:
    names = binding_names(node)
    return name in names or "*" in names


def literal_sequence(node: ast.expr) -> set[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = [literal(item) for item in node.elts]
    return {value for value in values if value is not None} if all(values) else None


def assigned_value(statement: ast.stmt, name: str) -> ast.expr | None:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return None
    names = [
        n.id
        for n in ast.walk(statement)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    ]
    return statement.value if names == [name] else None


def binds_loop_variable(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )


def loop_rebinds(node: ast.For, name: str) -> bool:
    return any(
        may_bind_name(child, name)
        for statement in node.body
        for child in ast.walk(statement)
    )


def sequence_escapes(tree: ast.Module, name: str) -> bool:
    """Only direct iteration or unpacking may read a finite sequence constant.

    Passing it to a function, aliasing it, or attribute/subscript access could mutate
    the sequence. Such cases stay dynamic instead of inferring an incomplete set.
    """
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
        ):
            parent = parents[node]
            if not (
                isinstance(parent, ast.For) and parent.iter is node
            ) and not isinstance(parent, ast.Starred):
                return True
    return False


class TargetResolver:
    """Bounded source-only constant resolution; never import a project module."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.sources: dict[str, ast.Module] = {}

    def tree(self, path: str) -> ast.Module:
        source = self.root / path
        if source.is_symlink() or not source.resolve().is_relative_to(self.root):
            raise ValueError(f"constant source escapes repository: {path}")
        if path not in self.sources:
            self.sources[path] = ast.parse(source.read_bytes(), filename=path)
        return self.sources[path]

    def imported_source(self, path: str, node: ast.ImportFrom) -> str | None:
        module = node.module or ""
        if node.level:
            package = ".".join(Path(path).parent.parts)
            module = importlib.util.resolve_name("." * node.level + module, package)
        base = Path(*module.split("."))
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
        present = [str(p) for p in candidates if (self.root / p).is_file()]
        return present[0] if len(present) == 1 else None

    def value(
        self,
        path: str,
        node: ast.expr,
        seen: frozenset = frozenset(),
        sequence: bool = False,
    ) -> set[str] | None:
        text = literal(node)
        if text and not sequence:
            return {text}
        if sequence and isinstance(node, (ast.List, ast.Tuple)):
            return literal_sequence(node)
        if not isinstance(node, ast.Name):
            return None
        return self.named_value(path, node.id, seen, sequence)

    def named_value(
        self, path: str, name: str, seen: frozenset, sequence: bool
    ) -> set[str] | None:
        if (path, name) in seen or len(seen) >= 32:
            return None
        tree = self.tree(path)
        bindings = constant_bindings(tree, name)
        if len(bindings) != 1 or (sequence and sequence_escapes(tree, name)):
            return None
        return self.definition(path, name, seen | {(path, name)}, sequence)

    def definition(
        self, path: str, name: str, seen: frozenset, sequence: bool
    ) -> set[str] | None:
        for statement in self.tree(path).body:
            value = assigned_value(statement, name)
            if value is not None:
                return self.value(path, value, seen, sequence)
            if isinstance(statement, ast.ImportFrom):
                result = self.reexport(path, statement, name, seen, sequence)
                if result is not None:
                    return result
        return None

    def reexport(
        self,
        path: str,
        node: ast.ImportFrom,
        name: str,
        seen: frozenset,
        sequence: bool,
    ) -> set[str] | None:
        for item in node.names:
            if (item.asname or item.name) == name:
                source = self.imported_source(path, node)
                if source:
                    return self.value(source, ast.Name(id=item.name), seen, sequence)
        return None

    def targets(self, path: str, call: ast.Call) -> set[str] | None:
        expression = argument(call, 0, "name")
        if expression is None:
            return None
        if isinstance(expression, ast.Name):
            loop = self.enclosing_loop(path, call, expression.id)
            if loop is not None:
                return self.value(path, loop.iter, sequence=True)
        return self.value(path, expression)

    def enclosing_loop(self, path: str, call: ast.Call, name: str) -> ast.For | None:
        parents = {
            child: parent
            for parent in ast.walk(self.tree(path))
            for child in ast.iter_child_nodes(parent)
        }
        node: ast.AST = call
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return None
            if binds_loop_variable(node, name):
                if isinstance(node, ast.For) and not loop_rebinds(node, name):
                    return node
                return None
        return None


def static_targets(node: ast.AST) -> list[tuple[ast.alias, str, str]]:
    if isinstance(node, ast.Import):
        return [(item, item.name, "") for item in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        return [(item, node.module, item.name) for item in node.names]
    return []


def matching(target: str) -> bool:
    return target.split(".")[0] in NAMESPACES


def parsed_source(path: str, raw: bytes, resolver: TargetResolver | None) -> ast.Module:
    tree = ast.parse(raw, filename=path)
    if resolver is not None:
        resolver.sources[path] = tree
    return tree


def source_observations(
    path: str, raw: bytes, resolver: TargetResolver | None = None
) -> tuple[list[dict], list[dict]]:
    tree = parsed_source(path, raw, resolver)
    aliases = loader_aliases(tree)
    # Bind the committed text: a CRLF checkout (Windows autocrlf) of the same
    # commit must produce the same evidence as the LF blob Git stores.
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    imports: list[dict] = []
    unresolved: list[dict] = []
    for node in ast.walk(tree):
        for alias, target, symbol in static_targets(node):
            if matching(target):
                imports.append(
                    observation(path, alias, digest, "static", target, symbol)
                )
        for row in dynamic_observations(path, node, digest, aliases, resolver):
            (unresolved if row["target"] is None else imports).append(row)
    return imports, unresolved


def dynamic_observations(
    path: str,
    node: ast.AST,
    digest: str,
    aliases: dict[str, set[str]],
    resolver: TargetResolver | None,
) -> list[dict]:
    if not isinstance(node, ast.Call):
        return []
    loaders = qualified(node.func, aliases) & LOADERS
    if not loaders:
        return []
    values = resolver.targets(path, node) if resolver else None
    targets = {dynamic_target(node, loaders, value) for value in values or [None]}
    return [
        observation(path, node, digest, "dynamic", target, "")
        for target in targets
        if target is None or matching(target)
    ]


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
    resolver = TargetResolver(root)
    paths = sorted(set(result.stdout.decode().split("\0")) - {""})
    for relative in paths:
        if not relative.endswith(".py"):
            continue
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"source is not a regular in-repository file: {relative}")
        found, unknown = source_observations(relative, path.read_bytes(), resolver)
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
    errors.extend(
        validate_entries(
            actual["imports"], inventory["imports"], "missing classification"
        )
    )
    errors.extend(
        validate_entries(
            actual["unresolved"],
            inventory.get("dynamic_boundaries", []),
            "unresolved dynamic target (missing classification)",
        )
    )
    return errors


def validate_entries(
    observations: list[dict], entries: list[dict], missing: str
) -> list[str]:
    errors = []
    expected = {identity(row): row for row in observations}
    seen: set[str] = set()
    for entry in entries:
        key = identity(entry)
        if key in seen:
            errors.append(f"duplicate classification: {key}")
        seen.add(key)
        errors.extend(entry_errors(entry, expected.get(key)))
    errors.extend(f"{missing}: {key}" for key in sorted(expected.keys() - seen))
    return errors


def entry_errors(entry: dict, expected: dict | None) -> list[str]:
    key = identity(entry)
    errors = [f"{message}: {key}" for message in review_errors(entry)]
    review_fields, boundary_findings = entry_review(entry, expected)
    errors.extend(boundary_findings)
    observed = {k: v for k, v in entry.items() if k not in review_fields}
    if observed != expected:
        errors.append(f"stale classification/source evidence: {key}")
    return errors


def entry_review(entry: dict, expected: dict | None) -> tuple[set[str], list[str]]:
    review_fields = {"classification", "evidence", "reviewer"}
    if expected is not None and expected["target"] is None:
        review_fields.update({"target_resolution", "boundary"})
        return review_fields, boundary_errors(entry)
    return review_fields, []


def boundary_errors(entry: dict) -> list[str]:
    key = identity(entry)
    errors = []
    if entry.get("target_resolution") != "runtime-supplied (not statically resolved)":
        errors.append(
            f"unresolved dynamic target requires explicit runtime boundary: {key}"
        )
    if not isinstance(entry.get("boundary"), str) or not entry["boundary"].strip():
        errors.append(f"unresolved dynamic target requires boundary evidence: {key}")
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
    print(
        f"import inventory: OK ({len(actual['imports'])} classified imports; "
        f"{len(actual['unresolved'])} reviewed dynamic boundaries with runtime-unknown targets)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
