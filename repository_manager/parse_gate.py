"""Tier-0 parse gate — does every source file in this tree still PARSE?

CONCEPT:RM-PARSE-GATE

Why this exists, concretely. On 2026-08-10 five ``epistemic-graph`` branches
were merged by a mechanical keep-both-sides resolution. The result was not
subtly wrong — **four files did not parse at all**, and the workspace
``Cargo.toml`` was rejected outright with ``duplicate key``, so cargo could not
even read the manifest. Nothing caught it: the merges succeeded, the diffs
reviewed clean, the branch bookkeeping said five landed. The damage surfaced
only when something finally tried to compile, which was hours and one handoff
document later.

The signature is always the same. Two branches add a sibling item at the SAME
place; the resolution INTERLEAVES them instead of concatenating, and the first
item's closing delimiter is swallowed by the second item's doc comment:

    Method::Quantum { op: QuantumOp,        <- lost its `},`
    /// Native visualization render surface  <- the next variant's doc comment
    Method::Viz { op: VizOp },

Or, for a count that advanced on main while two branches each added a term to
the older base, two whole statements survive where one belonged::

    let expected = 385
    let expected = 369 + ... ;
        + usize::from(cfg!(feature = "viz"));

Every one of those is caught by *parsing*, in seconds, with no build, no test
run, no environment, and no network. That is the entire thesis of this module:
**the cheapest possible stage should reject the cheapest possible class of
defect**, so a merge wave cannot hand N downstream lanes a tree that was never
syntactically valid.

Deliberately NOT a linter. It asks one question — "is this file syntactically
well-formed?" — and answers with a parser, never a heuristic or a regex. A grep
for duplicate keys, for instance, both false-positives (``name =`` legitimately
repeats across TOML sections) and misses the real in-section duplicate; only a
real TOML parser gets it right, which is why ``tomllib`` is used below.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess  # nosec B404 - fixed argv, no shell
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ParseFailure",
    "ParseReport",
    "check_tree",
    "dispatch",
    "main",
]

#: Directories never worth parsing: build output, vendored deps, VCS internals.
#: ``target`` and ``node_modules`` alone are the difference between seconds and
#: minutes on these repositories.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "target-isolated",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".uv-workspace-siblings",
    }
)


@dataclass
class ParseFailure:
    """One file that does not parse, and the parser's own words for why."""

    path: str
    language: str
    detail: str
    line: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParseReport:
    """The verdict for a whole tree."""

    ok: bool
    tree: str
    scanned: int
    failures: list[ParseFailure] = field(default_factory=list)
    by_language: dict[str, int] = field(default_factory=dict)
    skipped_languages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tree": self.tree,
            "scanned": self.scanned,
            "failures": [f.as_dict() for f in self.failures],
            "by_language": self.by_language,
            "skipped_languages": self.skipped_languages,
        }


def _walk(tree: Path, suffixes: frozenset[str]) -> Iterable[Path]:
    """Yield files under ``tree`` with a matching suffix, pruning build dirs."""
    for root, dirnames, filenames in os.walk(tree):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        base = Path(root)
        for name in filenames:
            if Path(name).suffix in suffixes:
                yield base / name


def _check_python(paths: Sequence[Path], tree: Path) -> list[ParseFailure]:
    failures: list[ParseFailure] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(
                ParseFailure(str(path.relative_to(tree)), "python", str(exc))
            )
            continue
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(
                ParseFailure(
                    str(path.relative_to(tree)),
                    "python",
                    exc.msg or "syntax error",
                    line=exc.lineno or 0,
                )
            )
    return failures


def _check_toml(paths: Sequence[Path], tree: Path) -> list[ParseFailure]:
    """``tomllib`` rejects duplicate keys — which is the whole point here.

    The ``members`` key appeared TWICE in one ``[workspace]`` table and cargo
    refused the manifest. No amount of grepping finds that reliably; a parser
    finds it every time.
    """
    failures: list[ParseFailure] = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            failures.append(ParseFailure(str(path.relative_to(tree)), "toml", str(exc)))
    return failures


def _check_json(paths: Sequence[Path], tree: Path) -> list[ParseFailure]:
    failures: list[ParseFailure] = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            line = getattr(exc, "lineno", 0) or 0
            failures.append(
                ParseFailure(str(path.relative_to(tree)), "json", str(exc), line=line)
            )
    return failures


def _check_rust(paths: Sequence[Path], tree: Path) -> tuple[list[ParseFailure], bool]:
    """Parse Rust with ``rustfmt``, which is a parser we already have.

    ``rustfmt --emit stdout`` fails loudly on a file it cannot parse and is
    dramatically cheaper than a build: it caught three of the 2026-08-10
    defects across 716 files in about two minutes, where ``cargo check`` needed
    four minutes per attempt and stopped at the FIRST error.

    Returns ``(failures, ran)`` — ``ran`` is False when rustfmt is unavailable,
    so the caller can report the language as SKIPPED rather than silently
    claiming a clean Rust tree it never examined.
    """
    rustfmt = shutil.which("rustfmt") or str(Path.home() / ".cargo" / "bin" / "rustfmt")
    if not Path(rustfmt).exists():
        return [], False

    failures: list[ParseFailure] = []
    for path in paths:
        try:
            proc = subprocess.run(  # nosec B603 - fixed argv, no shell
                [rustfmt, "--edition", "2021", "--emit", "stdout", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(ParseFailure(str(path.relative_to(tree)), "rust", str(exc)))
            continue
        if proc.returncode != 0:
            detail = next(
                (
                    ln.strip()
                    for ln in proc.stderr.splitlines()
                    if ln.startswith("error")
                ),
                proc.stderr.strip().splitlines()[0]
                if proc.stderr.strip()
                else "parse failed",
            )
            failures.append(ParseFailure(str(path.relative_to(tree)), "rust", detail))
    return failures, True


def check_tree(
    path: Path | str | None = None,
    *,
    languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse every supported source file under ``path``. Mutates nothing.

    Fast enough to run on every commit — that is the design constraint. If this
    ever becomes slow enough that someone is tempted to skip it, it has failed
    at its job regardless of what it catches.
    """
    tree = Path(path or Path.cwd()).resolve()
    wanted = set(languages or ("python", "toml", "json", "rust"))
    failures: list[ParseFailure] = []
    by_language: dict[str, int] = {}
    skipped: list[str] = []

    if "python" in wanted:
        files = list(_walk(tree, frozenset({".py", ".pyi"})))
        by_language["python"] = len(files)
        failures.extend(_check_python(files, tree))
    if "toml" in wanted:
        files = list(_walk(tree, frozenset({".toml"})))
        by_language["toml"] = len(files)
        failures.extend(_check_toml(files, tree))
    if "json" in wanted:
        files = list(_walk(tree, frozenset({".json"})))
        by_language["json"] = len(files)
        failures.extend(_check_json(files, tree))
    if "rust" in wanted:
        files = list(_walk(tree, frozenset({".rs"})))
        rust_failures, ran = _check_rust(files, tree)
        if ran:
            by_language["rust"] = len(files)
            failures.extend(rust_failures)
        else:
            skipped.append("rust (rustfmt not found)")

    report = ParseReport(
        ok=not failures,
        tree=str(tree),
        scanned=sum(by_language.values()),
        failures=failures,
        by_language=by_language,
        skipped_languages=skipped,
    )
    return report.as_dict()


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core shared by the CLI and the MCP tool so they cannot drift."""
    if action == "check":
        return check_tree(kwargs.get("path"), languages=kwargs.get("languages"))
    return {"ok": False, "error": f"unknown action: {action}"}


def main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.parse_gate [path]`` — exit 1 on any failure."""
    import argparse

    parser = argparse.ArgumentParser(prog="parse-gate", description=__doc__)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        choices=["python", "toml", "json", "rust"],
        help="restrict to one language (repeatable); default is all",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    args = parser.parse_args(argv)

    report = check_tree(args.path, languages=args.languages)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        counts = ", ".join(
            f"{n} {lang}" for lang, n in sorted(report["by_language"].items())
        )
        print(f"parse gate: scanned {report['scanned']} files ({counts})")
        for skipped in report["skipped_languages"]:
            print(f"  SKIPPED: {skipped}")
        for failure in report["failures"]:
            where = f":{failure['line']}" if failure["line"] else ""
            print(
                f"  PARSE-FAIL [{failure['language']}] {failure['path']}{where}: {failure['detail']}"
            )
        print(
            "OK"
            if report["ok"]
            else f"FAILED: {len(report['failures'])} file(s) do not parse"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
