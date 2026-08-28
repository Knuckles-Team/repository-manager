"""Characterization tests for run_single_enhancer.py::run_enhancer.

run_enhancer() dynamically loads a fixed roster of analyzer scripts (plus a
report generator and an SDD-handoff generator) from
``$CODE_ENHANCER_SCRIPTS_DIR``, tolerates any of them being missing/broken,
sanitizes and persists the aggregate results, then best-effort writes a
Markdown report and an SDD handoff. These tests exercise it end-to-end
against a throwaway ``$CODE_ENHANCER_SCRIPTS_DIR``/project directory (never
the real repository's own ``.specify/``, which run_single_enhancer.py's
module-level ``PROJECT_DIR``/``SPECIFY_DIR`` would otherwise point at --
those two globals are monkeypatched to tmp_path after import) covering: a
missing analyzer script (spec load fails), a successful analyzer, an
analyzer that raises, an analyzer that returns score == -1 (excluded), and a
missing report/SDD-handoff generator. Run unmodified before and after the
extract-method refactor of run_enhancer into
_run_one_analyzer/_write_report/_write_sdd_handoff/_load_module_from_path;
results (the written results.json content and the returned/printed
behavior) must be identical.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_single_enhancer.py"


def _load_run_single_enhancer(scripts_dir: Path, monkeypatch, tmp_project: Path):
    monkeypatch.setenv("CODE_ENHANCER_SCRIPTS_DIR", str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        f"run_single_enhancer_char_{id(scripts_dir)}", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Never touch the real repo's own .specify/ -- redirect to a throwaway
    # project directory. run_enhancer() reads these module globals directly.
    module.PROJECT_DIR = tmp_project
    module.SPECIFY_DIR = tmp_project / ".specify"
    return module


_ANALYZER_NAMES = [
    "analyze_project",
    "audit_dependencies",
    "analyze_codebase",
    "analyze_security",
    "analyze_tests",
    "audit_documentation",
    "analyze_architecture",
    "trace_concepts",
    "run_linters",
    "run_precommit",
    "run_tests",
    "analyze_directory_density",
    "analyze_ui",
    "analyze_version_sync",
    "audit_changelog",
    "grade_pytest",
    "scan_env_vars",
]


@pytest.fixture
def tmp_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    return project


def test_all_analyzers_missing_produces_f_grade_fallback_for_each(
    tmp_path, monkeypatch, tmp_project
):
    """No analyzer scripts exist -> exec_module raises FileNotFoundError for
    each -> every analyzer gets an F-grade fallback result (not skipped)."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    module.run_enhancer()

    results_path = tmp_project / ".specify" / "results.json"
    assert results_path.is_file()
    results = json.loads(results_path.read_text())
    assert len(results) == len(_ANALYZER_NAMES)
    assert all(r["grade"] == "F" for r in results)
    assert all("FileNotFoundError" in r["findings"][0] for r in results)


def test_successful_analyzer_result_is_included(tmp_path, monkeypatch, tmp_project):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze_project.py").write_text(
        textwrap.dedent(
            """
            def analyze_project(project_dir):
                return {"domain": "Project", "score": 90, "grade": "A",
                        "findings": [], "justifications": []}
            """
        ),
        encoding="utf-8",
    )
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    module.run_enhancer()

    results = json.loads((tmp_project / ".specify" / "results.json").read_text())
    # 1 real success + 16 F-grade fallbacks for the missing analyzer scripts.
    assert len(results) == len(_ANALYZER_NAMES)
    success = [r for r in results if r["domain"] == "Project"]
    assert len(success) == 1
    assert success[0]["score"] == 90


def test_analyzer_raising_produces_fallback_f_grade_result(
    tmp_path, monkeypatch, tmp_project
):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze_project.py").write_text(
        "def analyze_project(project_dir):\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    module.run_enhancer()

    results = json.loads((tmp_project / ".specify" / "results.json").read_text())
    # The one analyzer that exists raises RuntimeError; the other 16 are
    # missing and raise FileNotFoundError -- both paths land in the except
    # branch, so all 17 come back as F-grade fallbacks.
    assert len(results) == len(_ANALYZER_NAMES)
    project_result = next(r for r in results if r["domain"] == "Analyze Project")
    assert project_result["grade"] == "F"
    assert project_result["score"] == 0
    assert "RuntimeError" in project_result["findings"][0]


def test_analyzer_returning_score_negative_one_is_excluded(
    tmp_path, monkeypatch, tmp_project
):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "analyze_project.py").write_text(
        textwrap.dedent(
            """
            def analyze_project(project_dir):
                return {"domain": "Project", "score": -1, "grade": "N/A",
                        "findings": [], "justifications": []}
            """
        ),
        encoding="utf-8",
    )
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    module.run_enhancer()

    results = json.loads((tmp_project / ".specify" / "results.json").read_text())
    # analyze_project itself returns score == -1 -> excluded; the other 16
    # missing analyzers still produce F-grade fallbacks (raise != return -1).
    assert len(results) == len(_ANALYZER_NAMES) - 1
    assert not any(r["domain"] == "Project" for r in results)


def test_missing_report_and_sdd_generators_do_not_raise(
    tmp_path, monkeypatch, tmp_project
):
    """generate_report.py / generate_sdd_handoff.py absent -> caught, no crash."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    # Must not raise.
    module.run_enhancer()
    assert (tmp_project / ".specify" / "results.json").is_file()
    assert not (tmp_project / ".specify" / "reports" / "code_enhancement_report.md").exists()


def test_report_and_sdd_generators_are_invoked_when_present(
    tmp_path, monkeypatch, tmp_project
):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "generate_report.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            def generate_report(results, project_name, output_path):
                Path(output_path).write_text(f"report for {project_name}: {len(results)} results")
            """
        ),
        encoding="utf-8",
    )
    (scripts_dir / "generate_sdd_handoff.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            def generate_sdd_handoff(results, project_name, output_dir):
                marker = Path(output_dir) / "sdd_handoff_marker.txt"
                marker.write_text(f"{project_name}:{len(results)}")
            """
        ),
        encoding="utf-8",
    )
    module = _load_run_single_enhancer(scripts_dir, monkeypatch, tmp_project)

    module.run_enhancer()

    report_path = tmp_project / ".specify" / "reports" / "code_enhancement_report.md"
    assert report_path.is_file()
    assert "report for project" in report_path.read_text()
    assert (tmp_project / "sdd_handoff_marker.txt").is_file()
