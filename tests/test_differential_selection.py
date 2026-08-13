"""Tests for the differential pre-push test selector (CONCEPT:RM-DIFF-SELECT).

Every test drives a REAL git repository (same discipline as ``test_merge_queue.py``)
so the merge-base-relative diff computation (reused from
:func:`repository_manager.merge_queue.changed_paths`) is exercised for real, not
mocked. The headline tests are the two the task explicitly asks for:

* :func:`test_correctness_leaf_module_change_selects_its_failing_test` — a leaf
  module changes, the selector picks exactly the test that imports it, and
  ACTUALLY RUNNING that selected test with real pytest fails for the expected
  reason (proves the selection is not merely plausible but sufficient).
* :func:`test_correctness_indirect_change_still_selected_via_transitive_import`
  — the diff touches a module two hops away from the test (through an
  intermediate module), and the selector still finds it via the unbounded BFS
  (or legitimately falls back to the full suite) — never silently drops it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from repository_manager import differential_selection as ds


def _run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git init -q", repo)
    _run("git config user.email test@example.com", repo)
    _run("git config user.name Test", repo)
    return repo


def _commit_all(repo: Path, message: str) -> None:
    _run("git add -A", repo)
    _run(f"git commit -q -m {message!r}", repo)


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small synthetic package: pkg.leaf <- pkg.middle <- pkg.hub, tests/ mirrors it."""
    repo = _git_repo(tmp_path)
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/leaf.py", "def compute():\n    return 1\n")
    _write(
        repo,
        "pkg/middle.py",
        """
        from pkg.leaf import compute


        def wrapped():
            return compute() + 1
        """,
    )
    _write(
        repo,
        "pkg/hub.py",
        """
        from pkg.middle import wrapped


        def entry():
            return wrapped()
        """,
    )
    _write(repo, "pkg/orphan.py", "def unused():\n    return 42\n")
    _write(repo, "pkg/config.py", "VALUE = 1\n")
    _write(
        repo,
        "pkg/lazy.py",
        """
        def __getattr__(name):
            if name == "Secret":
                from pkg.secret_impl import Secret
                return Secret
            raise AttributeError(name)
        """,
    )
    _write(repo, "pkg/secret_impl.py", "class Secret:\n    pass\n")
    _write(repo, "tests/__init__.py", "")
    _write(repo, "tests/conftest.py", "import pytest\n")
    _write(
        repo,
        "tests/test_leaf.py",
        """
        from pkg.leaf import compute


        def test_compute():
            assert compute() == 1
        """,
    )
    _write(
        repo,
        "tests/test_hub.py",
        """
        from pkg.hub import entry


        def test_entry():
            assert entry() == 2
        """,
    )
    _write(repo, "tests/sub/__init__.py", "")
    _write(repo, "tests/sub/conftest.py", "import pytest\n")
    _write(
        repo,
        "tests/sub/test_helper_user.py",
        """
        from tests.sub.helper import build


        def test_build():
            assert build() == "built"
        """,
    )
    _write(repo, "tests/sub/helper.py", 'def build():\n    return "built"\n')
    _commit_all(repo, "initial")
    _run("git branch -f main", repo)
    return repo


def _select(repo: Path, **kwargs) -> ds.DifferentialSelection:
    kwargs.setdefault("test_roots", ("tests",))
    kwargs.setdefault("src_roots", (".",))
    return ds.select_differential_tests(repo, base_ref="main", ref="HEAD", **kwargs)


# --------------------------------------------------------------------------- #
# Rule-by-rule behaviour
# --------------------------------------------------------------------------- #


def test_leaf_module_change_selects_transitive_importer(project: Path) -> None:
    _write(project, "pkg/leaf.py", "def compute():\n    return 2  # changed\n")
    _commit_all(project, "change leaf")
    result = _select(project)
    assert not result.full_suite, result.reason
    assert "tests/test_leaf.py" in result.selected


def test_two_hop_transitive_change_still_selects_hub_test(project: Path) -> None:
    """pkg.leaf changes; tests/test_hub.py reaches it only via pkg.middle -> pkg.hub."""
    _write(project, "pkg/leaf.py", "def compute():\n    return 3  # changed\n")
    _commit_all(project, "change leaf again")
    result = _select(project)
    assert not result.full_suite, result.reason
    assert "tests/test_hub.py" in result.selected, result.as_dict()
    assert "tests/test_leaf.py" in result.selected


def test_test_only_change_selects_itself(project: Path) -> None:
    _write(
        project,
        "tests/test_leaf.py",
        """
        from pkg.leaf import compute


        def test_compute():
            assert compute() == 1
            assert True
        """,
    )
    _commit_all(project, "touch test")
    result = _select(project)
    assert not result.full_suite, result.reason
    assert result.selected == ("tests/test_leaf.py",)


def test_conftest_change_selects_whole_subdirectory_not_full_suite(
    project: Path,
) -> None:
    _write(project, "tests/sub/conftest.py", "import pytest\n# changed\n")
    _commit_all(project, "touch sub conftest")
    result = _select(project)
    assert not result.full_suite, result.reason
    assert "tests/sub" in result.selected


def test_root_conftest_change_falls_back_to_full_suite(project: Path) -> None:
    _write(project, "tests/conftest.py", "import pytest\n# changed\n")
    _commit_all(project, "touch root conftest")
    result = _select(project)
    assert result.full_suite, result.as_dict()


def test_widely_imported_basename_falls_back(project: Path) -> None:
    _write(project, "pkg/config.py", "VALUE = 2\n")
    _commit_all(project, "change config")
    result = _select(project)
    assert result.full_suite, result.as_dict()
    assert any("hub basename" in v.reason for v in result.verdicts if v.fallback)


def test_orphan_module_with_zero_importers_fails_open(project: Path) -> None:
    """The dangerous silent-under-selection case: no importer found at all."""
    _write(project, "pkg/orphan.py", "def unused():\n    return 43\n")
    _commit_all(project, "change orphan")
    result = _select(project)
    assert result.full_suite, result.as_dict()
    assert any(
        "no test file imports" in v.reason for v in result.verdicts if v.fallback
    )


def test_module_defining_lazy_getattr_falls_back(project: Path) -> None:
    _write(
        project,
        "pkg/lazy.py",
        """
        def __getattr__(name):
            if name == "Secret":
                from pkg.secret_impl import Secret
                return Secret
            if name == "Other":
                return 1
            raise AttributeError(name)
        """,
    )
    _commit_all(project, "change lazy registry")
    result = _select(project)
    assert result.full_suite, result.as_dict()
    assert any("__getattr__" in v.reason for v in result.verdicts if v.fallback)


def test_lazy_registry_target_falls_back_even_though_unimported_statically(
    project: Path,
) -> None:
    """pkg.secret_impl has NO static importer at all — only referenced via
    pkg.lazy's __getattr__ string literal. Both rule 7 (registry reference) and
    rule 9 (zero importers) would fail this open; either is a correct verdict."""
    _write(project, "pkg/secret_impl.py", "class Secret:\n    x = 2\n")
    _commit_all(project, "change secret impl")
    result = _select(project)
    assert result.full_suite, result.as_dict()


def test_unparsable_change_falls_back(project: Path) -> None:
    _write(project, "pkg/leaf.py", "def compute(:\n    this is not python\n")
    _commit_all(project, "break syntax")
    result = _select(project)
    assert result.full_suite, result.as_dict()
    assert any("unparsable" in v.reason for v in result.verdicts if v.fallback)


def test_high_fanin_module_falls_back_even_with_unremarkable_basename(
    project: Path,
) -> None:
    repo = project
    for i in range(30):
        _write(
            repo,
            f"tests/test_fan_{i}.py",
            f"""
            from pkg.leaf import compute


            def test_fan_{i}():
                assert compute() == 1
            """,
        )
    _commit_all(repo, "add many importers")
    _write(repo, "pkg/leaf.py", "def compute():\n    return 99  # changed\n")
    _commit_all(repo, "change high-fanin leaf")
    result = _select(repo, fanin_fallback_threshold=25)
    assert result.full_suite, result.as_dict()
    assert any("fan-in" in v.reason for v in result.verdicts if v.fallback)


def test_governing_file_change_falls_back(project: Path) -> None:
    _write(project, "pyproject.toml", "[project]\nname = 'pkg'\n")
    _commit_all(project, "add pyproject")
    result = _select(project)
    assert result.full_suite, result.as_dict()


def test_no_changes_selects_nothing_without_full_suite(project: Path) -> None:
    _commit_all_noop = project  # nothing changed since HEAD == main
    result = ds.select_differential_tests(
        project, base_ref="main", ref="HEAD", test_roots=("tests",)
    )
    assert result.selected == ()
    assert not result.full_suite


# --------------------------------------------------------------------------- #
# The required correctness check: under-selection must never happen
# --------------------------------------------------------------------------- #


def test_correctness_leaf_module_change_selects_its_failing_test(
    project: Path, tmp_path: Path
) -> None:
    """Introduce a real bug in pkg.leaf, confirm the selector picks
    tests/test_leaf.py, then ACTUALLY RUN that selected test and prove it fails
    for the expected reason — the selection is not just plausible, it is
    sufficient to catch the regression."""
    _write(project, "pkg/leaf.py", "def compute():\n    return 999  # BUG\n")
    _commit_all(project, "introduce bug in leaf")

    result = _select(project)
    assert not result.full_suite, result.as_dict()
    assert "tests/test_leaf.py" in result.selected

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *result.selected],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "test_compute" in proc.stdout
    assert "assert 999 == 1" in proc.stdout or "assert compute() == 1" in proc.stdout


def test_correctness_indirect_change_still_selected_via_transitive_import(
    project: Path,
) -> None:
    """The diff touches pkg.leaf, two hops from tests/test_hub.py
    (leaf <- middle <- hub <- test_hub.py). Confirm the selector still finds
    it (transitive BFS) rather than silently dropping it because it is not a
    DIRECT importer."""
    _write(project, "pkg/leaf.py", "def compute():\n    return -1  # indirect bug\n")
    _commit_all(project, "indirect bug")

    result = _select(project)
    assert not result.full_suite, result.as_dict()
    assert "tests/test_hub.py" in result.selected, (
        "under-selection: the indirectly-related test was NOT selected -- "
        f"{result.as_dict()}"
    )

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_hub.py"],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_correctness_unrelated_change_does_not_select_unrelated_test(
    project: Path,
) -> None:
    """A negative control: changing pkg.leaf must NOT select
    tests/sub/test_helper_user.py, which has no import relationship to it."""
    _write(project, "pkg/leaf.py", "def compute():\n    return 4  # changed\n")
    _commit_all(project, "change leaf, unrelated to helper")
    result = _select(project)
    assert not result.full_suite, result.as_dict()
    assert "tests/sub/test_helper_user.py" not in result.selected
    assert "tests/sub" not in result.selected


def test_pytest_argv_for_selection_narrow(project: Path) -> None:
    _write(project, "pkg/leaf.py", "def compute():\n    return 5\n")
    _commit_all(project, "narrow change")
    result = _select(project)
    argv = ds.pytest_argv_for_selection(
        result,
        base_command=("python3", "-m", "pytest", "-q"),
        full_suite_targets=("tests",),
    )
    assert argv[:4] == ["python3", "-m", "pytest", "-q"]
    assert "tests/test_leaf.py" in argv


def test_pytest_argv_for_selection_full_suite(project: Path) -> None:
    _write(project, "pkg/config.py", "VALUE = 3\n")
    _commit_all(project, "hub change")
    result = _select(project)
    argv = ds.pytest_argv_for_selection(
        result,
        base_command=("python3", "-m", "pytest", "-q"),
        full_suite_targets=("tests",),
    )
    assert argv == ["python3", "-m", "pytest", "-q", "tests"]
