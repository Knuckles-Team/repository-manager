"""Differential pre-push test selection — the third gate tier (CONCEPT:RM-DIFF-SELECT).

**Where this sits (GOC-69).** The fleet has three gate tiers: pre-commit (seconds,
static analysis only), pre-push (minutes, differential), CI (unbounded, the full
suite — the only authority). This module is the missing middle tier's mapping
step: **changed files -> the test files that could plausibly catch a regression
in them.** ``rm_gates(action=run, stage=heavy)`` (a sibling lane) is the intended
caller — this module is a library function it calls, **not a competing
entrypoint**. There is no CLI subcommand or MCP tool of its own beyond the thin
``--differential-select`` CLI flag added alongside it for standalone
demonstration/debugging.

**What is reused from the merge queue, and what is genuinely new.**
:func:`repository_manager.merge_queue.changed_paths` computes "files this ref
changed relative to its merge-base with the base ref" — that is reused
VERBATIM, not reimplemented, because the merge queue already solved "which
files changed" correctly (merge-base, not a plain diff — see its docstring).
What the merge queue does NOT do is pick a *subset* of tests: its
``compare: pytest-ids`` machinery (:mod:`repository_manager.merge_queue`
``run_gate``/``compute_gate_baseline``/``_compare_gate``) judges whether a
pytest run's failing-id set is NEW relative to the base ref, but the pytest
*command itself* is a fixed, repo-declared argv — every gate always runs the
same target (e.g. ``tests/unit``). That fixed-target invocation is exactly
right for pre-commit-strength differential *gating* (only NEW failures block)
but does nothing for pre-push-strength differential *selection* (run fewer
tests, faster). This module supplies that missing mapping; the judging step
downstream stays on the merge queue's existing pytest-ids comparator — see
:func:`pytest_argv_for_selection` and the module docstring section "The seam".

**The core problem, restated:** a changed-files -> affected-tests mapping WILL
miss dependencies (dynamic imports, fixtures, plugin registries, lazy
``__getattr__`` re-exports). So this mapping must **fail OPEN, never closed** —
run more tests, or the full suite, whenever it cannot prove a change's blast
radius is small. Every fallback branch below names the specific reason so a
false "ran everything" is at least explainable, and a false "ran too little"
never happens by construction (each rule is unable to under-select rather than
merely intending not to). ``full_suite=True`` in the result is a first-class
outcome, not a failure of the tool — the module docstring and every fallback
:class:`FileVerdict` say so explicitly, because pre-push output must never be
mistaken for a completeness claim (the ninth-gate lesson this program keeps
finding: a gate that reports coverage it does not have).

**Rules, in the order applied to each changed file:**

1. **Not Python / not under a configured root** -> ignored for pytest-selection
   purposes UNLESS it is a recognized build/test-configuration file
   (``pyproject.toml``, ``pytest.ini``, ``conftest.py`` handled specially below,
   the repo's own ``.mergequeue.yaml``), in which case it forces the full suite:
   changing what governs the run invalidates any narrower answer.
2. **A test file itself changed** (under a configured test root, filename
   matches pytest's own default discovery pattern) -> select that file. No
   import-graph reasoning needed; the file IS its own coverage.
3. **A ``conftest.py`` changed** -> select every test under that conftest's
   directory (fixtures apply directory-wide, not file-by-file). If that
   directory IS a configured test root itself (a suite-wide conftest), this
   collapses to the full suite.
4. **The file cannot be parsed** (a ``SyntaxError`` mid-edit, or any other
   ``ast.parse`` failure) -> full suite. An unparsable diff cannot be reasoned
   about, so it is not narrowed.
5. **The file defines a module-level ``__getattr__``** (PEP 562 lazy exports —
   au's 43-instance pattern) -> full suite. A static import graph cannot see
   through ``__getattr__``-mediated access, so a change here could affect any
   caller that reaches it lazily; the caller side of that same pattern is
   covered by rule 7.
6. **The file's basename is a known hub name** (``__init__.py``, ``config.py``,
   ``models.py``, ``base.py``, ``settings.py``) -> full suite. These are
   conventionally imported everywhere; treating them specially avoids relying
   on the import graph (built from what is ON DISK today) to already contain
   every real caller.
7. **The file's module name is referenced as a string literal inside some
   OTHER file's module-level ``__getattr__`` body** (the lazy-registry pattern:
   ``def __getattr__(name): if name == "Foo": from .foo import Foo``) ->
   full suite. This is what closes the gap rule 5 leaves at the *use* site
   instead of the *definition* site: a change to the target of a lazy
   re-export is invisible to a plain reverse-import walk.
8. **The file's static reverse-import fan-in exceeds the configured
   threshold** (default 25 importers) -> full suite, even when its basename
   is unremarkable. Fan-in is a *measured* hub signal, not a naming
   convention, so it catches a hub the naming heuristic (rule 6) missed.
9. **Zero test files found importing the module, transitively, by static
   analysis** -> full suite. This is deliberately conservative: "no
   importer found" is indistinguishable from "the import graph missed the
   real caller" (dynamic dispatch, a plugin registry, an MCP tool
   registration decorator) without further proof, and under-selecting to an
   EMPTY set is the single worst outcome a differential tier can produce. A
   genuinely dead/orphaned module pays the same cost as any other
   unclassifiable change here — correct behaviour, not a false positive.
10. **Otherwise** -> select every test file that imports the changed module,
    DIRECTLY OR TRANSITIVELY (unbounded BFS over the reverse-import graph —
    depth is never capped, because capping it would silently reintroduce
    under-selection). Any conftest.py reached along the way is expanded to
    its whole directory per rule 3's reasoning, not left as a bare file
    selection (a conftest has no test functions of its own).

If ANY changed file falls into a fail-open rule, the WHOLE result is
``full_suite=True`` — fail-open composes by disjunction, never diluted by
the other changed files that happened to classify narrowly.

**What this deliberately does not attempt.** No attempt is made to trim BY
TEST FUNCTION (node-id) — only by file/directory. Pytest's own collection
already amortizes file-level selection well, and node-id-level selection
would require resolving `self`/fixture parametrization statically, which is
exactly the kind of indirection rule 9's fail-open exists to not paper over.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from repository_manager.merge_queue import changed_paths

#: Basenames conventionally imported from everywhere in a Python package —
#: treating a change to one of these as narrow would require the import graph
#: to already be complete, which is precisely what cannot be assumed.
_HUB_BASENAMES = frozenset(
    {"__init__.py", "config.py", "models.py", "base.py", "settings.py"}
)

#: Files whose presence anywhere in the diff invalidates any narrower answer:
#: they govern HOW tests run, not what they cover.
_GOVERNING_FILENAMES = frozenset(
    {"pyproject.toml", "pytest.ini", "setup.cfg", ".mergequeue.yaml", "tox.ini"}
)

_TEST_FILE_RE = re.compile(r"^test_.*\.py$|^.*_test\.py$")

_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".specify",
    }
)

DEFAULT_FANIN_FALLBACK_THRESHOLD = 25


def _iter_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _EXCLUDED_DIR_NAMES:
                    continue
                stack.append(entry)
            elif entry.suffix == ".py":
                out.append(entry)
    return out


@dataclass(frozen=True)
class _ModuleInfo:
    name: str
    is_init: bool


def _module_info_for(
    repo_root: Path, file_path: Path, src_roots: tuple[str, ...]
) -> _ModuleInfo | None:
    resolved = file_path.resolve()
    for root in src_roots:
        root_dir = (
            repo_root.resolve() if root in (".", "") else (repo_root / root).resolve()
        )
        try:
            rel = resolved.relative_to(root_dir)
        except ValueError:
            continue
        parts = list(rel.parts)
        if not parts or not parts[-1].endswith(".py"):
            continue
        is_init = parts[-1] == "__init__.py"
        if is_init:
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts:
            continue
        return _ModuleInfo(name=".".join(parts), is_init=is_init)
    return None


def _relative_base(module_name: str, is_init: bool, level: int) -> str:
    package = (
        module_name
        if is_init
        else module_name.rsplit(".", 1)[0]
        if "." in module_name
        else ""
    )
    parts = package.split(".") if package else []
    drop = level - 1
    if drop > 0:
        parts = parts[: max(0, len(parts) - drop)]
    return ".".join(parts)


def _importfrom_candidates(node: ast.ImportFrom, module: _ModuleInfo) -> set[str]:
    candidates: set[str] = set()
    if node.level and node.level > 0:
        base = _relative_base(module.name, module.is_init, node.level)
        full_module = f"{base}.{node.module}" if node.module else base
    else:
        full_module = node.module or ""
    if full_module:
        candidates.add(full_module)
    for alias in node.names:
        if alias.name == "*":
            continue
        target = f"{full_module}.{alias.name}" if full_module else alias.name
        candidates.add(target)
    return candidates


@dataclass
class _RepoIndex:
    """The reverse-import graph + lazy-registry index, built once per selection call."""

    #: dotted module name -> the file that defines it
    module_to_file: dict[str, Path] = field(default_factory=dict)
    #: dotted module name -> set of files that import it (direct edges only)
    importers_of: dict[str, set[Path]] = field(default_factory=dict)
    #: files that define a module-level ``__getattr__`` (PEP 562 lazy export)
    lazy_getattr_files: set[Path] = field(default_factory=set)
    #: files with module-level __getattr__ -> string literals found in their body
    #: (the lazy-registry pattern's target names)
    lazy_registry_strings: dict[Path, set[str]] = field(default_factory=dict)
    #: files that could not be parsed
    unparsable: set[Path] = field(default_factory=set)


def _has_module_level_getattr(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in tree.body
    )


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def build_repo_index(
    repo_root: Path, *, src_roots: tuple[str, ...], test_roots: tuple[str, ...]
) -> _RepoIndex:
    """Walk ``src_roots`` + ``test_roots`` once, building the reverse-import graph.

    Static-only, no execution. A file that cannot be parsed is recorded so its
    callers still fail open per rule 4 without crashing the whole walk.
    """
    index = _RepoIndex()
    roots = tuple(dict.fromkeys((*src_roots, *test_roots)))
    files: list[Path] = []
    seen_dirs: set[Path] = set()
    for root in roots:
        root_dir = (
            repo_root.resolve() if root in (".", "") else (repo_root / root).resolve()
        )
        if root_dir in seen_dirs or not root_dir.is_dir():
            continue
        seen_dirs.add(root_dir)
        files.extend(_iter_python_files(root_dir))

    parsed: dict[Path, tuple[ast.Module, _ModuleInfo | None]] = {}
    for file_path in files:
        module = _module_info_for(repo_root, file_path, src_roots + test_roots)
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, ValueError, OSError):
            index.unparsable.add(file_path)
            continue
        parsed[file_path] = (tree, module)
        if module is not None:
            index.module_to_file[module.name] = file_path
        if _has_module_level_getattr(tree):
            index.lazy_getattr_files.add(file_path)
            index.lazy_registry_strings[file_path] = _string_literals(tree)

    for file_path, (tree, module) in parsed.items():
        if module is None:
            continue
        for node in ast.walk(tree):
            candidates: set[str] = set()
            if isinstance(node, ast.Import):
                candidates.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                candidates.update(_importfrom_candidates(node, module))
            for candidate in candidates:
                index.importers_of.setdefault(candidate, set()).add(file_path)

    return index


@dataclass(frozen=True)
class FileVerdict:
    """What one changed file resolved to, and why — for a legible report."""

    path: str
    fallback: bool
    reason: str
    selected: tuple[str, ...] = ()


@dataclass(frozen=True)
class DifferentialSelection:
    """The result of mapping a diff onto pytest targets.

    ``full_suite=True`` is a normal, expected outcome (see module docstring) —
    callers must not treat it as an error, only as "narrower selection was not
    provable safe this time".
    """

    changed_files: tuple[str, ...]
    selected: tuple[str, ...]
    full_suite: bool
    reason: str
    verdicts: tuple[FileVerdict, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "changed_files": list(self.changed_files),
            "selected": list(self.selected),
            "full_suite": self.full_suite,
            "reason": self.reason,
            "verdicts": [
                {
                    "path": v.path,
                    "fallback": v.fallback,
                    "reason": v.reason,
                    "selected": list(v.selected),
                }
                for v in self.verdicts
            ],
        }


def _is_under(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
    except ValueError:
        return False
    return True


def _expand_conftest(repo_root: Path, conftest_path: Path) -> str:
    """The pytest target for a changed/reached conftest.py: its whole directory."""
    directory = conftest_path.parent
    rel = directory.resolve().relative_to(repo_root.resolve())
    return str(rel) if str(rel) != "." else "."


def _is_suite_wide_conftest(
    conftest_path: Path, test_root_dirs: tuple[Path, ...]
) -> bool:
    return conftest_path.parent.resolve() in {d.resolve() for d in test_root_dirs}


def _reverse_bfs_test_targets(
    module_name: str,
    index: _RepoIndex,
    *,
    repo_root: Path,
    src_roots: tuple[str, ...],
    test_roots: tuple[str, ...],
    test_root_dirs: tuple[Path, ...],
) -> tuple[set[Path], set[Path], bool]:
    """BFS the reverse-import graph from ``module_name`` to every reaching test file.

    Unbounded depth by design (rule 10): capping it would silently reintroduce
    under-selection, which is the one failure mode this whole module exists to
    avoid. Returns (reached test files, reached non-suite-wide conftest files,
    hit_suite_wide_conftest).
    """
    visited_modules: set[str] = {module_name}
    queue: deque[str] = deque([module_name])
    reached_tests: set[Path] = set()
    reached_conftests: set[Path] = set()
    suite_wide = False
    all_roots = src_roots + test_roots

    while queue:
        current = queue.popleft()
        for importer_file in index.importers_of.get(current, ()):
            if importer_file.name == "conftest.py":
                if _is_suite_wide_conftest(importer_file, test_root_dirs):
                    suite_wide = True
                else:
                    reached_conftests.add(importer_file)
                continue
            under_test_root = any(_is_under(importer_file, d) for d in test_root_dirs)
            if under_test_root:
                reached_tests.add(importer_file)
                # A test file can itself be imported by another test file
                # (shared helpers) — keep walking through it too.
            importer_module = _module_info_for(repo_root, importer_file, all_roots)
            if importer_module is None or importer_module.name in visited_modules:
                continue
            visited_modules.add(importer_module.name)
            queue.append(importer_module.name)
            if suite_wide:
                return reached_tests, reached_conftests, suite_wide

    return reached_tests, reached_conftests, suite_wide


def _is_lazily_referenced(module: _ModuleInfo, index: _RepoIndex) -> str:
    """Non-empty reason string if ``module`` is a target of a lazy ``__getattr__``
    registry somewhere in the indexed tree — the use-site half of rule 5/7."""
    leaf = module.name.rsplit(".", 1)[-1]
    for registry_file, literals in index.lazy_registry_strings.items():
        if leaf in literals or module.name in literals:
            return f"referenced as a string literal in {registry_file}'s module-level __getattr__ (lazy-import registry)"
    return ""


def select_differential_tests(
    repo: Path,
    *,
    base_ref: str = "main",
    ref: str = "HEAD",
    src_roots: tuple[str, ...] = (".",),
    test_roots: tuple[str, ...] = ("tests",),
    fanin_fallback_threshold: int = DEFAULT_FANIN_FALLBACK_THRESHOLD,
) -> DifferentialSelection:
    """Map the diff ``base_ref..ref`` (by merge-base, via the merge queue's own
    :func:`~repository_manager.merge_queue.changed_paths`) onto pytest targets.

    This is a pure, read-only static-analysis function — it never runs pytest,
    never mutates the working tree, and never calls out to git beyond the one
    reused ``changed_paths`` call. See the module docstring for the per-file
    rule order and the fail-open rationale.
    """
    repo = repo.resolve()
    changed = changed_paths(repo, base_ref, ref)
    test_root_dirs = tuple((repo / t).resolve() for t in test_roots)

    if not changed:
        return DifferentialSelection(
            changed_files=(),
            selected=(),
            full_suite=False,
            reason="no changed files relative to the merge-base with the base ref",
        )

    for rel in changed:
        name = Path(rel).name
        if name in _GOVERNING_FILENAMES:
            return DifferentialSelection(
                changed_files=tuple(changed),
                selected=(),
                full_suite=True,
                reason=f"{rel} governs how tests run/are configured — narrowing it is unsafe",
                verdicts=(FileVerdict(rel, True, "governing/config file changed"),),
            )

    index = build_repo_index(repo, src_roots=src_roots, test_roots=test_roots)

    verdicts: list[FileVerdict] = []
    selected: set[str] = set()
    any_fallback = False

    for rel in changed:
        file_path = (repo / rel).resolve()
        if file_path.suffix != ".py":
            verdicts.append(
                FileVerdict(
                    rel, False, "not a Python file — ignored for pytest selection"
                )
            )
            continue

        under_test_root = any(_is_under(file_path, d) for d in test_root_dirs)

        # Rule 3: a changed conftest.py -> its whole directory (or full suite
        # when it is the suite-wide root conftest).
        if file_path.name == "conftest.py":
            if not file_path.is_file() or _is_suite_wide_conftest(
                file_path, test_root_dirs
            ):
                any_fallback = True
                verdicts.append(
                    FileVerdict(rel, True, "suite-wide conftest.py changed/removed")
                )
                continue
            target = _expand_conftest(repo, file_path)
            selected.add(target)
            verdicts.append(
                FileVerdict(
                    rel, False, "conftest.py changed — directory selected", (target,)
                )
            )
            continue

        # Rule 2: a changed test file selects itself.
        if under_test_root and _TEST_FILE_RE.match(file_path.name):
            selected.add(rel)
            verdicts.append(
                FileVerdict(rel, False, "test file changed — selects itself", (rel,))
            )
            continue

        # Rule 4: unparsable.
        if not file_path.is_file():
            any_fallback = True
            verdicts.append(
                FileVerdict(rel, True, "changed file no longer exists on this ref")
            )
            continue
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, ValueError, OSError) as exc:
            any_fallback = True
            verdicts.append(FileVerdict(rel, True, f"unparsable: {exc}"))
            continue

        # Rule 5: the changed file itself defines a lazy __getattr__.
        if _has_module_level_getattr(tree):
            any_fallback = True
            verdicts.append(
                FileVerdict(
                    rel,
                    True,
                    "defines a module-level __getattr__ (PEP 562 lazy export)",
                )
            )
            continue

        # Rule 6: known hub basename.
        if file_path.name in _HUB_BASENAMES:
            any_fallback = True
            verdicts.append(FileVerdict(rel, True, f"hub basename {file_path.name!r}"))
            continue

        module = _module_info_for(repo, file_path, src_roots + test_roots)
        if module is None:
            any_fallback = True
            verdicts.append(
                FileVerdict(
                    rel,
                    True,
                    "not resolvable to a module under any configured src/test root",
                )
            )
            continue

        # Rule 7: referenced from someone else's lazy registry.
        lazy_reason = _is_lazily_referenced(module, index)
        if lazy_reason:
            any_fallback = True
            verdicts.append(FileVerdict(rel, True, lazy_reason))
            continue

        # Rule 8: measured fan-in hub.
        fanin = len(index.importers_of.get(module.name, ()))
        if fanin > fanin_fallback_threshold:
            any_fallback = True
            verdicts.append(
                FileVerdict(
                    rel,
                    True,
                    f"fan-in {fanin} exceeds threshold {fanin_fallback_threshold}",
                )
            )
            continue

        reached_tests, reached_conftests, suite_wide = _reverse_bfs_test_targets(
            module.name,
            index,
            repo_root=repo,
            src_roots=src_roots,
            test_roots=test_roots,
            test_root_dirs=test_root_dirs,
        )
        if suite_wide:
            any_fallback = True
            verdicts.append(
                FileVerdict(rel, True, "transitively reaches a suite-wide conftest.py")
            )
            continue

        targets = {str(p.relative_to(repo)) for p in reached_tests}
        targets |= {_expand_conftest(repo, c) for c in reached_conftests}

        # Rule 9: zero importers found — the dangerous silent-under-selection case.
        if not targets:
            any_fallback = True
            verdicts.append(
                FileVerdict(
                    rel,
                    True,
                    "no test file imports this module (directly or transitively) by static analysis",
                )
            )
            continue

        selected |= targets
        verdicts.append(
            FileVerdict(
                rel,
                False,
                f"{len(targets)} test target(s) reached transitively",
                tuple(sorted(targets)),
            )
        )

    if any_fallback:
        fallback_files = [v.path for v in verdicts if v.fallback]
        return DifferentialSelection(
            changed_files=tuple(changed),
            selected=tuple(sorted(selected)),
            full_suite=True,
            reason=(
                f"{len(fallback_files)}/{len(changed)} changed file(s) could not be narrowed safely "
                f"({', '.join(fallback_files[:5])}{'…' if len(fallback_files) > 5 else ''}) — "
                "failing open to the full suite per the fail-open contract"
            ),
            verdicts=tuple(verdicts),
        )

    return DifferentialSelection(
        changed_files=tuple(changed),
        selected=tuple(sorted(selected)),
        full_suite=False,
        reason=f"every changed file resolved to a bounded set of test targets ({len(selected)} target(s))",
        verdicts=tuple(verdicts),
    )


def pytest_argv_for_selection(
    selection: DifferentialSelection,
    *,
    base_command: tuple[str, ...],
    full_suite_targets: tuple[str, ...] = ("tests/unit",),
) -> list[str]:
    """Build the pytest argv `rm_gates(stage=heavy)` should run for ``selection``.

    ``base_command`` is the repo's OWN declared prefix — e.g. the
    ``targeted-tests`` gate's ``[".venv/bin/python", "-m", "pytest", "-q",
    "-p", "no:randomly", "-rfE", "--timeout", "120"]`` from ``.mergequeue.yaml``
    — so this function contributes ONLY the target selection, never the
    flags/comparator. The resulting argv is meant to be run through the
    merge queue's OWN ``run_gate``/``compute_gate_baseline`` (``compare:
    pytest-ids``) for judging — this function does not judge pass/fail itself,
    it only narrows *what* runs. That differential judgment against a base-ref
    baseline is reused verbatim, not reimplemented (see module docstring).
    """
    targets = (
        list(full_suite_targets) if selection.full_suite else list(selection.selected)
    )
    if not targets:
        targets = list(full_suite_targets)
    return [*base_command, *targets]
