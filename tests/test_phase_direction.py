"""Tests for the maintenance-phase dependency-direction check (RF-ADR-009 §3).

Every fixture is a real on-disk workspace: a ``workspace.yml`` whose
``maintenance.phases`` mirror the canonical seven-phase order, plus checkouts
with ``pyproject.toml`` files and Python sources. Each failing case asserts the
exact edge or status that caused the failure and that nothing else did, and
each is paired with the same fixture minus that one cause passing.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess  # nosec B404 - fixed argv only, never shell=True
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml  # type: ignore[import-untyped]

from repository_manager import dependency_readiness as dr
from repository_manager import gate_runner
from repository_manager.cli_commands.context import CliRuntime
from repository_manager.cli_commands.parser import run
from repository_manager.cli_commands.phase_direction import (
    run_phase_direction_cli,
    run_phase_direction_here_cli,
)
from tests.conftest import isolated_git_subprocess_env

AU = "agent-packages/agent-utilities"
REPO_ROOT = Path(__file__).resolve().parents[1]

_MANIFEST = """
repositories:
  - url: https://example.invalid/org/pipelines.git
{extra_top}
subdirectories:
  agent-packages:
    repositories:
      - url: https://example.invalid/org/epistemic-graph.git
      - url: https://example.invalid/org/agent-utilities.git
      - url: https://example.invalid/org/graph-os.git
    subdirectories:
      agents:
        repositories:
          - url: https://example.invalid/org/agent-a.git
          - url: https://example.invalid/org/agent-b.git
  services:
    repositories:
      - url: https://example.invalid/org/langfuse.git
maintenance:
  phases:
    - {{phase: 1, projects: [pipelines]}}
    - {{phase: 2, projects: [epistemic-graph]}}
    - {{phase: 4, projects: [agent-utilities{extra_phase_project}]}}
    - {{phase: 5, projects: [graph-os]}}
    - {{phase: 7, bulk_push: true}}
"""


def _manifest(root: Path, *, extra_top: str = "", extra_phase_project: str = "") -> Path:
    path = root / "workspace.yml"
    path.write_text(
        _MANIFEST.format(extra_top=extra_top, extra_phase_project=extra_phase_project)
    )
    return path


def _pyproject(root: Path, identifier: str, name: str, body: str = "") -> Path:
    checkout = root / identifier
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n{body}')
    return checkout


def _source(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _check(root: Path, manifest: Path, *repositories: str) -> dr.PhaseDirectionReport:
    return dr.check_phase_direction(
        manifest, workspace_root=root, repositories=list(repositories) or None
    )


# Inference fixtures -- real, disposable git repos (never the actual host
# checkout: every repo below lives under `tmp_path`). `isolated_git_subprocess_env`
# is the same helper `test_git_env_leak_guard.py` uses, for the same GOC-71
# reason: this repo's own `pytest` hook can itself run as a child of a real
# `git commit`/`git push`. ------------------------------------------------------


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        env=isolated_git_subprocess_env(),
    )


def _init_repo_with_remote(path: Path, remote_url: str) -> Path:
    """A real, minimal, one-commit git repo with an ``origin`` remote."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "phase-direction-tests@example.invalid"], path)
    _git(["config", "user.name", "phase-direction-tests"], path)
    (path / "README.md").write_text("fixture\n")
    _git(["add", "README.md"], path)
    _git(["commit", "-q", "-m", "init"], path)
    _git(["remote", "add", "origin", remote_url], path)
    return path


def _here(
    root: Path | None, manifest: Path, start: Path
) -> dr.PhaseDirectionReport:
    return dr.check_phase_direction_here(manifest, workspace_root=root, start=start)


# (a) --------------------------------------------------------------------------


def test_phase4_runtime_dependency_on_phase5_package_fails(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n')

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert report.errors == []
    assert [r.status for r in report.repositories] == ["checked"]
    assert [e.as_dict() for e in report.violations] == [
        {
            "from_repository": AU,
            "from_phase": 4,
            "to_package": "graph-os",
            "to_phase": 5,
            "edge_class": "runtime",
            "group": None,
            "location": f"{AU}/pyproject.toml",
            "blocking": True,
        }
    ]
    payload = report.as_dict()
    assert payload["ok"] is False
    assert payload["blocking_violation_count"] == 1
    assert payload["violation_counts"]["runtime"] == 1

    # The same repository without that one edge passes: the edge was the cause.
    _pyproject(tmp_path, AU, "agent-utilities", "dependencies = []\n")
    assert _check(tmp_path, manifest, "agent-utilities").ok is True


# (b) --------------------------------------------------------------------------


def test_same_phase_and_earlier_phase_edges_pass(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities", 'dependencies = ["epistemic-graph[full]>=2"]\n')
    _pyproject(
        tmp_path,
        "agent-packages/agents/agent-a",
        "agent-a",
        'dependencies = ["agent-utilities[mcp]>=1", "agent-b>=0.1", "requests"]\n',
    )
    _source(
        tmp_path,
        "agent-packages/agents/agent-a/agent_a/server.py",
        "import agent_b\nfrom agent_utilities.mcp import serve\nimport epistemic_graph.client\n",
    )
    _pyproject(tmp_path, "agent-packages/agents/agent-b", "agent-b")
    # A `services/` stack is infra, never in scope — even with no phase.
    (tmp_path / "services" / "langfuse").mkdir(parents=True)

    report = _check(tmp_path, manifest)

    assert report.ok is True, report.as_dict()
    assert report.violations == []
    assert report.errors == []
    statuses = {r.repository: (r.phase, r.status) for r in report.repositories}
    assert statuses == {
        "pipelines": (1, "no_checkout"),
        "agent-packages/epistemic-graph": (2, "no_checkout"),
        AU: (4, "checked"),
        "agent-packages/graph-os": (5, "no_checkout"),
        "agent-packages/agents/agent-a": (7, "checked"),
        "agent-packages/agents/agent-b": (7, "checked"),
    }
    counts = report.as_dict()["edge_counts"]
    assert counts["runtime"] == 3  # au->eg (4->2), a->au (7->4), a->b (7->7)
    assert counts["import"] == 3  # a->b, a->au, a->eg


# (c) --------------------------------------------------------------------------


def test_repository_listed_in_no_phase_is_a_hard_unknown_phase_error(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path, extra_top="  - url: https://example.invalid/org/plans.git"
    )

    report = _check(tmp_path, manifest)

    unknown = [r for r in report.repositories if r.status == "unknown_phase"]
    assert [(r.repository, r.phase) for r in unknown] == [("plans", None)]
    assert report.violations == [] and report.errors == []
    assert report.ok is False  # the unknown phase alone fails it
    assert report.as_dict()["repository_counts"]["unknown_phase"] == 1

    assert _check(tmp_path, _manifest(tmp_path)).ok is True


def test_phase_project_missing_from_manifest_is_blocking(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, extra_phase_project=", ghost-sdk")

    report = _check(tmp_path, manifest)

    assert [(r.repository, r.status) for r in report.repositories][-1] == (
        "ghost-sdk",
        "not_in_manifest",
    )
    assert report.ok is False and report.errors == []


def test_requested_repository_unknown_to_manifest_is_an_error(tmp_path: Path) -> None:
    report = _check(tmp_path, _manifest(tmp_path), "no-such-repo")

    assert report.ok is False
    assert report.repositories == []
    assert "no-such-repo" in report.errors[0].detail


# (d) --------------------------------------------------------------------------


def test_later_phase_import_fails_in_production_but_is_test_class_in_tests(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities")
    _source(
        tmp_path,
        f"{AU}/agent_utilities/host.py",
        "# import graph_os  (a comment: no import)\n"
        "from .graph_os import local_module\n"
        "text = 'import graph_os'\n"
        "import os\n"
        "from graph_os.gateway import Gateway\n",
    )

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert report.errors == []
    assert [(e.edge_class, e.location, e.blocking) for e in report.violations] == [
        ("import", f"{AU}/agent_utilities/host.py:5", True)
    ]

    (tmp_path / AU / "agent_utilities" / "host.py").unlink()
    _source(tmp_path, f"{AU}/tests/unit/test_host.py", "import graph_os\n")
    _source(tmp_path, f"{AU}/conftest.py", "from graph_os import fixtures\n")

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is True, report.as_dict()
    assert sorted((e.edge_class, e.location, e.blocking) for e in report.violations) == [
        ("test-import", f"{AU}/conftest.py:1", False),
        ("test-import", f"{AU}/tests/unit/test_host.py:1", False),
    ]
    assert report.as_dict()["violation_counts"]["test-import"] == 2


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nimportlib.import_module("graph_os.gateway")\n',
        'import importlib as il\nil.import_module("graph_os.gateway")\n',
        'from importlib import import_module as load\nload("graph_os.gateway")\n',
        'import importlib\nimportlib.util.find_spec("graph_os.gateway")\n',
        'import importlib.util as iu\niu.find_spec("graph_os.gateway")\n',
        'from importlib.util import find_spec as probe\nprobe("graph_os.gateway")\n',
        'import_client("graph_os", "Client")\n',
        'loader.import_client("graph_os", "Client")\n',
    ],
)
def test_literal_dynamic_import_of_later_phase_fails(
    tmp_path: Path, source: str
) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities")
    _source(tmp_path, f"{AU}/agent_utilities/host.py", source)

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert report.errors == []
    violations = [edge for edge in report.violations if edge.to_package == "graph-os"]
    assert len(violations) == 1
    assert violations[0].edge_class == "import"
    assert violations[0].blocking is True


def test_nonliteral_dynamic_import_is_not_invented_as_an_edge(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities")
    _source(
        tmp_path,
        f"{AU}/agent_utilities/host.py",
        "import importlib\n"
        'module_name = "graph_" + "os"\n'
        "importlib.import_module(module_name)\n",
    )

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is True, report.as_dict()
    assert report.edges == []


def test_unparseable_source_naming_a_fleet_package_is_blocking(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities")
    _source(tmp_path, f"{AU}/agent_utilities/broken.py", "import graph_os(\n")
    _source(tmp_path, f"{AU}/agent_utilities/unrelated.py", "def broken(:\n")

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert [Path(e.declared_by).name for e in report.errors] == ["broken.py"]
    assert "does not parse" in report.errors[0].detail


def test_nested_checkouts_and_build_output_are_not_scanned(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities")
    _source(tmp_path, f"{AU}/vendor/other/.git/HEAD", "ref: refs/heads/main\n")
    _source(tmp_path, f"{AU}/vendor/other/mod.py", "import graph_os\n")
    for skipped in ("build", ".venv", "target-isolated", "pkg.egg-info"):
        _source(tmp_path, f"{AU}/{skipped}/mod.py", "import graph_os\n")

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is True, report.as_dict()
    assert report.edges == []


# (e) --------------------------------------------------------------------------


def test_optional_extra_and_dependency_group_edges_are_classified_and_fail(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(
        tmp_path,
        AU,
        "agent-utilities",
        "dependencies = []\n"
        "[project.optional-dependencies]\n"
        'serving = ["agent-utilities[mcp]", "graph-os>=1"]\n'
        "[dependency-groups]\n"
        'dev = ["pytest", {include-group = "lint"}, "graph-os"]\n'
        'lint = ["ruff"]\n'
        "[tool.uv]\n"
        'dev-dependencies = ["graph-os"]\n',
    )

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert report.errors == []
    assert [(e.edge_class, e.group, e.blocking) for e in report.violations] == [
        ("optional", "serving", True),
        ("dependency-group", "dev", True),
        ("dependency-group", "tool.uv.dev-dependencies", True),
    ]
    assert report.as_dict()["violation_counts"] == {
        "runtime": 0,
        "optional": 1,
        "dependency-group": 2,
        "import": 0,
        "test-import": 0,
    }


def test_malformed_dependency_group_is_typed_blocking_state(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities", '[dependency-groups]\ndev = "pytest"\n')

    report = _check(tmp_path, manifest, "agent-utilities")

    assert report.ok is False
    assert "dependency group 'dev' must be a list" in report.errors[0].detail


@pytest.mark.parametrize(
    ("phases", "detail"),
    [
        ("  phases: []\n", "declares no phases"),
        (
            "  phases:\n    - {phase: 1, projects: [a]}\n    - {phase: 1, projects: [b]}\n",
            "duplicate maintenance phase number",
        ),
        (
            "  phases:\n    - {phase: 1, projects: [a]}\n    - {phase: 2, projects: [a]}\n",
            "more than one phase",
        ),
        ("  phases:\n    - {phase: 1}\n    - {phase: 2}\n", "ambiguous bulk phase"),
        ("  phases:\n    - {phase: true, projects: [a]}\n", "positive integers"),
    ],
)
def test_malformed_phase_plan_is_an_error_not_a_pass(
    tmp_path: Path, phases: str, detail: str
) -> None:
    manifest = tmp_path / "workspace.yml"
    manifest.write_text("repositories: []\nmaintenance:\n" + phases)

    report = _check(tmp_path, manifest)

    assert report.ok is False
    assert detail in report.errors[0].detail


# Surfaces ---------------------------------------------------------------------


def _violating_workspace(tmp_path: Path) -> Path:
    manifest = _manifest(tmp_path)
    _pyproject(tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n')
    return manifest


def test_dispatch_and_rm_gates_route_report_the_same_violation(tmp_path: Path) -> None:
    manifest = _violating_workspace(tmp_path)

    direct = dr.dispatch(
        "phase_direction",
        manifest_path=str(manifest),
        workspace_root=str(tmp_path),
        repositories=["agent-utilities"],
    )
    routed = gate_runner.dispatch(
        "phase_direction",
        repos="agent-utilities",
        manifest_path=str(manifest),
        workspace_root=str(tmp_path),
    )

    assert direct == routed
    assert direct["ok"] is False
    assert [v["to_package"] for v in direct["violations"]] == ["graph-os"]
    assert dr.dispatch("phase_direction")["ok"] is False  # manifest is required


def test_cli_adapter_prints_json_and_exits_nonzero(tmp_path: Path, capsys) -> None:
    manifest = _violating_workspace(tmp_path)
    args = argparse.Namespace(
        file=str(manifest),
        workspace=str(tmp_path),
        phase_direction_repository=["agent-utilities"],
    )

    assert run_phase_direction_cli(args) == 1
    assert json.loads(capsys.readouterr().out)["blocking_violation_count"] == 1

    _pyproject(tmp_path, AU, "agent-utilities")
    assert run_phase_direction_cli(args) == 0


def test_cli_flag_exits_before_any_git_factory_call(tmp_path: Path, capsys) -> None:
    manifest = _violating_workspace(tmp_path)
    git_factory = MagicMock()
    runtime = CliRuntime(
        git_factory=git_factory,
        version="0.0.0-test",
        default_workspace=str(tmp_path),
        default_workspace_yml=str(manifest),
        default_threads=1,
        logger=MagicMock(),
        synchronize_workspace_manifest=MagicMock(),
        manifest_error=ValueError,
    )
    argv = ["repository-manager", "--phase-direction", "-f", str(manifest)]
    argv += ["-w", str(tmp_path)]

    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
        run(runtime)

    assert exc.value.code == 1
    git_factory.assert_not_called()
    assert json.loads(capsys.readouterr().out)["ok"] is False


# Repository-inferring mode -----------------------------------------------------
#
# ``check_phase_direction_here`` resolves which manifest repository ``start``
# is, then checks ONLY that repository's own on-disk source -- the mode a
# repository's own pre-push hook needs, since it cannot name itself by
# manifest identifier the way the workspace-wide ``--phase-direction-repository``
# selector does.


# (f) -- inference from a canonical checkout ------------------------------------


def test_infer_from_canonical_checkout_matches_the_named_check(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    checkout = _pyproject(tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n')
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")

    report = _here(tmp_path, manifest, checkout)

    assert report.errors == []
    assert [
        (r.repository, r.name, r.phase, r.status) for r in report.repositories
    ] == [(AU, "agent-utilities", 4, "checked")]
    assert [e.to_package for e in report.violations] == ["graph-os"]
    assert report.ok is False

    # Matches the workspace-wide, name-selected check exactly.
    named = _check(tmp_path, manifest, "agent-utilities")
    assert report.as_dict() == named.as_dict()


def test_infer_from_canonical_checkout_with_no_violation_passes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    checkout = _pyproject(tmp_path, AU, "agent-utilities")
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")

    report = _here(tmp_path, manifest, checkout)

    assert report.ok is True
    assert report.repositories[0].status == "checked"


# (g) -- inference from a linked worktree OUTSIDE the workspace -----------------


def test_infer_from_linked_worktree_resolves_to_canonical_identity(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    manifest = _manifest(workspace_root)
    checkout = _pyproject(workspace_root, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n')
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")
    # `git worktree add` checks out the branch's COMMITTED tree, not whatever
    # untracked files happen to sit in the canonical checkout -- commit the
    # pyproject.toml so the linked worktree created below actually has it.
    _git(["add", "pyproject.toml"], checkout)
    _git(["commit", "-q", "-m", "add pyproject"], checkout)

    # Mirrors the real convention: worktrees live under a SEPARATE tree, not
    # under the workspace path (`/home/apps/worktrees/<repo>/<branch>`).
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    worktree_path = worktree_root / "agent-utilities" / "some-branch"
    _git(
        ["worktree", "add", str(worktree_path), "-b", "some-branch"],
        checkout,
    )
    assert not str(worktree_path).startswith(str(workspace_root))

    report = _here(workspace_root, manifest, worktree_path)

    assert report.errors == []
    assert report.repositories[0].repository == AU
    assert report.repositories[0].name == "agent-utilities"
    assert report.repositories[0].status == "checked"
    assert [e.to_package for e in report.violations] == ["graph-os"]


# (h) -- an unknown remote fails closed ------------------------------------------


def test_unresolvable_repository_is_a_hard_error_not_a_silent_pass(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    stray = _init_repo_with_remote(
        tmp_path / "stray-repo", "https://example.invalid/org/totally-unrelated.git"
    )

    report = _here(workspace_root, manifest, stray)

    assert report.ok is False
    assert report.repositories == []
    assert report.edges == []
    assert len(report.errors) == 1
    assert "resolves to no workspace.yml repository" in report.errors[0].detail


def test_not_a_git_checkout_is_a_hard_error(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    not_a_repo = tmp_path / "just-a-directory"
    not_a_repo.mkdir()

    report = _here(tmp_path, manifest, not_a_repo)

    assert report.ok is False
    assert report.repositories == []
    assert "cannot infer which manifest repository" in report.errors[0].detail


# (i) -- only the resolved repository's edges are reported ----------------------


def test_only_the_resolved_repositorys_edges_are_reported(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    au_checkout = _pyproject(
        tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n'
    )
    _init_repo_with_remote(au_checkout, "https://example.invalid/org/agent-utilities.git")

    # A SIBLING repository with its OWN, DIFFERENT violation (phase 5 ->
    # phase 7's agent-a) -- must never leak into a report scoped to
    # agent-utilities.
    gos_identifier = "agent-packages/graph-os"
    _pyproject(
        tmp_path, gos_identifier, "graph-os", 'dependencies = ["agent-a>=1.0"]\n'
    )

    report = _here(tmp_path, manifest, au_checkout)

    assert [r.repository for r in report.repositories] == [AU]
    assert [e.from_repository for e in report.edges] == [AU]
    assert [e.to_package for e in report.violations] == ["graph-os"]
    # The workspace-wide check DOES see graph-os's own violation too --
    # proving the inferring mode's narrower scope is the cause of the
    # difference above, not a fixture mistake.
    everything = _check(tmp_path, manifest)
    assert {e.from_repository for e in everything.violations} == {AU, gos_identifier}


# (j) -- inherited git-hook pointer env vars must not redirect resolution -------


def test_resolution_ignores_inherited_git_repository_pointer_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GOC-71 class: a real ``git push`` running this AS a hook exports
    ``GIT_DIR``/``GIT_WORK_TREE`` into the hook's process, and git honors
    those OVER an explicit ``-C <path>``. Without ``_sanitized_git_subprocess_env``
    every ``git`` call this module makes would silently resolve against
    ``decoy`` below instead of ``checkout`` -- reproduced directly by the
    same mechanism ``tests/test_git_env_leak_guard.py`` proves at the
    subprocess level.
    """
    manifest = _manifest(tmp_path)
    checkout = _pyproject(tmp_path, AU, "agent-utilities")
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")

    decoy = _init_repo_with_remote(
        tmp_path / "decoy-real-repo", "https://example.invalid/org/decoy.git"
    )
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))

    report = _here(tmp_path, manifest, checkout)

    assert report.errors == []
    assert report.repositories[0].repository == AU
    assert report.repositories[0].name == "agent-utilities"
    assert report.ok is True


# Surfaces -- CLI / dispatch / rm_gates / the packaged pre-commit hook ----------


def test_dispatch_and_rm_gates_route_report_the_same_here_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    checkout = _pyproject(
        tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n'
    )
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")

    direct = dr.dispatch(
        "phase_direction_here", manifest_path=str(manifest), start=str(checkout)
    )
    routed = gate_runner.dispatch(
        "phase_direction_here", manifest_path=str(manifest), start=str(checkout)
    )

    assert direct == routed
    assert direct["ok"] is False
    assert [v["to_package"] for v in direct["violations"]] == ["graph-os"]
    assert dr.dispatch("phase_direction_here")["ok"] is False  # manifest is required


def test_cli_here_adapter_infers_and_exits_nonzero_on_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _manifest(tmp_path)
    checkout = _pyproject(
        tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n'
    )
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")
    args = argparse.Namespace(file=str(manifest), phase_direction_start=str(checkout))

    assert run_phase_direction_here_cli(args) == 1
    assert json.loads(capsys.readouterr().out)["blocking_violation_count"] == 1

    _pyproject(tmp_path, AU, "agent-utilities")
    assert run_phase_direction_here_cli(args) == 0


def test_cli_here_flag_exits_before_any_git_factory_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _manifest(tmp_path)
    checkout = _pyproject(
        tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n'
    )
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")
    git_factory = MagicMock()
    runtime = CliRuntime(
        git_factory=git_factory,
        version="0.0.0-test",
        default_workspace=str(tmp_path),
        default_workspace_yml=str(manifest),
        default_threads=1,
        logger=MagicMock(),
        synchronize_workspace_manifest=MagicMock(),
        manifest_error=ValueError,
    )
    argv = [
        "repository-manager",
        "--phase-direction-here",
        "-f",
        str(manifest),
        "--phase-direction-start",
        str(checkout),
    ]

    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
        run(runtime)

    assert exc.value.code == 1
    git_factory.assert_not_called()
    assert json.loads(capsys.readouterr().out)["blocking_violation_count"] == 1


# (k) -- the packaged pre-commit hook entry actually runs -----------------------


def test_packaged_pre_commit_hook_entry_runs() -> None:
    """Parses the REAL ``.pre-commit-hooks.yaml`` this repo ships, then
    invokes its literal ``entry`` command through the CLI -- proving the
    exact string another repository's ``.pre-commit-config.yaml`` would
    adopt actually resolves and runs, not just that ``check_phase_direction_here``
    works when called directly."""
    hooks = yaml.safe_load((REPO_ROOT / ".pre-commit-hooks.yaml").read_text())
    hook = next(h for h in hooks if h["id"] == "phase-direction")

    assert hook["stages"] == ["pre-push"]
    assert hook["pass_filenames"] is False
    assert hook["always_run"] is True
    assert hook["language"] == "python"

    argv = shlex.split(hook["entry"])
    assert argv == ["repository-manager", "--phase-direction-here"]


def test_packaged_hook_entry_argv_infers_and_reports_a_real_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs the hook's exact ``entry`` argv (no ``-f``/``--phase-direction-start``,
    exactly as an adopting repository's ``git push`` would invoke it) against a
    fixture repository, relying on the SAME defaults a real pre-push hook does:
    ``-f``'s manifest default and the process's current working directory."""
    hooks = yaml.safe_load((REPO_ROOT / ".pre-commit-hooks.yaml").read_text())
    hook = next(h for h in hooks if h["id"] == "phase-direction")
    argv = shlex.split(hook["entry"])

    manifest = _manifest(tmp_path)
    checkout = _pyproject(
        tmp_path, AU, "agent-utilities", 'dependencies = ["graph-os>=1.0"]\n'
    )
    _init_repo_with_remote(checkout, "https://example.invalid/org/agent-utilities.git")

    runtime = CliRuntime(
        git_factory=MagicMock(),
        version="0.0.0-test",
        default_workspace=str(tmp_path),
        default_workspace_yml=str(manifest),
        default_threads=1,
        logger=MagicMock(),
        synchronize_workspace_manifest=MagicMock(),
        manifest_error=ValueError,
    )
    monkeypatch.chdir(checkout)

    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
        run(runtime)

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["blocking_violation_count"] == 1
    assert payload["repositories"][0]["repository"] == AU
