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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repository_manager import dependency_readiness as dr
from repository_manager import gate_runner
from repository_manager.cli_commands.context import CliRuntime
from repository_manager.cli_commands.parser import run
from repository_manager.cli_commands.phase_direction import run_phase_direction_cli

AU = "agent-packages/agent-utilities"

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
