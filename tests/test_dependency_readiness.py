"""Tests for the dependency-readiness gate (repository_manager.dependency_readiness).

Covers both layers off the SAME primitives -- no network in these tests: every
test supplies a fake :class:`~repository_manager.dependency_readiness.IndexBackend`
so behavior is proven deterministically, and the live-index proof (does this
correctly fail against agent-utilities' real, currently-unsatisfiable
``epistemic-graph`` constraint, and pass for a satisfiable one) is demonstrated
separately, out of band, against the real PyPI index.
"""

from __future__ import annotations

import json

from repository_manager import dependency_readiness as dr
from repository_manager.scan_models import HookResult, RepoScanResult


class _FakeBackend:
    """A scripted `IndexBackend`: package -> versions, or an exception to raise."""

    def __init__(self, table: dict[str, list[str] | Exception], calls: list | None = None):
        self.table = table
        self.calls = calls if calls is not None else []

    def available_versions(self, package, *, index_url, timeout=10.0):
        self.calls.append((package, index_url))
        entry = self.table.get(package, [])
        if isinstance(entry, Exception):
            raise entry
        versions = entry
        return dr.AvailableVersions(
            package=package,
            index_url=index_url,
            versions=versions,
            latest=versions[-1] if versions else None,
        )


def _constraint(package="epistemic-graph", spec=">=2.23.2,<3.0.0", declared_by="pyproject.toml"):
    return dr.DeclaredConstraint(
        package=package,
        raw_requirement=f"{package}{spec}",
        specifier=spec,
        extras=(),
        declared_by=declared_by,
    )


# --------------------------------------------------------------------------- #
# check_constraint
# --------------------------------------------------------------------------- #


def test_check_constraint_satisfied():
    backend = _FakeBackend({"epistemic-graph": ["2.22.0", "2.23.0", "2.23.2", "2.24.0"]})
    result = dr.check_constraint(
        _constraint(), index_urls=["https://pypi.org/simple"], backend=backend
    )
    assert result.status == "satisfied"
    assert result.satisfied is True
    assert result.matching_version == "2.24.0"


def test_check_constraint_unsatisfied_names_package_constraint_and_available():
    """The live-instance shape: 2.23.0 published, >=2.23.2 declared."""
    backend = _FakeBackend({"epistemic-graph": ["2.22.4", "2.23.0"]})
    result = dr.check_constraint(
        _constraint(), index_urls=["https://pypi.org/simple"], backend=backend
    )
    assert result.status == "unsatisfied"
    assert result.satisfied is False
    assert result.latest_available == "2.23.0"
    # Message must name the package, the declared constraint, and what's available.
    assert "epistemic-graph" in result.detail
    assert "2.23.2" in result.detail
    assert "2.23.0" in result.detail


def test_check_constraint_index_unreachable_is_distinguished_from_unsatisfied():
    backend = _FakeBackend(
        {"epistemic-graph": dr.IndexUnreachableError("connection refused")}
    )
    result = dr.check_constraint(
        _constraint(),
        index_urls=["https://pypi.example/simple"],
        backend=backend,
        retries=1,
    )
    assert result.status == "index_unreachable"
    assert result.status != "unsatisfied"  # different problem, different message
    assert "connection refused" in result.detail


def test_check_constraint_retries_transient_unreachable_then_succeeds():
    """Absorbs the publish-but-not-yet-CDN-visible window."""
    calls = {"n": 0}

    class _FlakyBackend:
        def available_versions(self, package, *, index_url, timeout=10.0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise dr.IndexUnreachableError("503 momentarily")
            return dr.AvailableVersions(
                package=package, index_url=index_url, versions=["2.23.2"], latest="2.23.2"
            )

    result = dr.check_constraint(
        _constraint(),
        index_urls=["https://pypi.org/simple"],
        backend=_FlakyBackend(),
        retries=2,
        retry_delay_s=0,
    )
    assert result.status == "satisfied"
    assert calls["n"] == 2


def test_check_constraint_second_index_used_when_first_lacks_package():
    class _SecondIndexHasIt(_FakeBackend):
        def available_versions(self, package, *, index_url, timeout=10.0):
            if index_url == "https://primary/simple":
                raise dr.PackageUnknownError("not here")
            return dr.AvailableVersions(
                package=package, index_url=index_url, versions=["2.23.2"], latest="2.23.2"
            )

    result = dr.check_constraint(
        _constraint(),
        index_urls=["https://primary/simple", "https://secondary/simple"],
        backend=_SecondIndexHasIt({}),
    )
    assert result.status == "satisfied"
    assert result.index_urls_checked == ["https://primary/simple", "https://secondary/simple"]


# --------------------------------------------------------------------------- #
# declared_fleet_constraints -- scoping
# --------------------------------------------------------------------------- #


def test_declared_fleet_constraints_scopes_to_fleet_packages_only(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "some-repo"
dependencies = [
  "epistemic-graph[full]>=2.23.2,<3.0.0",
  "requests>=2.34.2",
]
[project.optional-dependencies]
extra = ["skill-graphs>=1.1.0"]
"""
    )
    constraints = dr.declared_fleet_constraints(
        tmp_path, fleet_packages={"epistemic-graph", "skill-graphs"}
    )
    names = {c.package for c in constraints}
    assert names == {"epistemic-graph", "skill-graphs"}
    assert "requests" not in names  # not a fleet package -- out of scope by design

    eg = next(c for c in constraints if c.package == "epistemic-graph")
    assert eg.extras == ("full",)
    assert eg.specifier == "<3.0.0,>=2.23.2"


def test_declared_fleet_constraints_excludes_own_package_name(tmp_path):
    """A repo never gates on a constraint that merely happens to name itself."""
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "epistemic-graph"
dependencies = ["epistemic-graph-stubs>=1.0.0"]
"""
    )
    constraints = dr.declared_fleet_constraints(
        tmp_path, fleet_packages={"epistemic-graph", "epistemic-graph-stubs"}
    )
    assert [c.package for c in constraints] == ["epistemic-graph-stubs"]


def test_fleet_package_names_excludes_infra_subdirectories(tmp_path):
    """A `services/`-tree repo (e.g. a Docker stack literally named `langfuse`)
    must never collide with an unrelated third-party PyPI package of the same
    name -- this was caught live: `services/langfuse.git` (a Docker Swarm
    stack) vs. the third-party `langfuse` SDK agent-utilities depends on."""
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        """
repositories: []
subdirectories:
  agent-packages:
    repositories:
      - url: https://example.com/org/epistemic-graph.git
  services:
    repositories:
      - url: https://example.com/org/langfuse.git
  images:
    repositories:
      - url: https://example.com/org/some-base-image.git
"""
    )
    names = dr.fleet_package_names(manifest)
    assert "epistemic-graph" in names
    assert "langfuse" not in names
    assert "some-base-image" not in names


def test_fleet_package_names_trusts_maintenance_phases_regardless_of_location():
    """`maintenance.phases[].projects` names are trusted unconditionally --
    being in the managed publish plan is itself proof of package-ness."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "workspace.yml"
        manifest.write_text(
            """
repositories: []
maintenance:
  phases:
    - phase: 1
      name: solo
      project: agent-utilities
"""
        )
        names = dr.fleet_package_names(manifest)
        assert "agent-utilities" in names


# --------------------------------------------------------------------------- #
# check_tree -- Layer 1, the pre-push hook entrypoint
# --------------------------------------------------------------------------- #


def test_check_tree_fails_on_unsatisfiable_fleet_constraint(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "agent-utilities"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    backend = _FakeBackend({"epistemic-graph": ["2.22.4", "2.23.0"]})
    report = dr.check_tree(
        tmp_path, fleet_packages={"epistemic-graph"}, backend=backend
    )
    assert report.ok is False
    assert len(report.checks) == 1
    assert report.checks[0].status == "unsatisfied"


def test_check_tree_passes_for_satisfiable_constraint(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "agent-utilities"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    backend = _FakeBackend({"epistemic-graph": ["2.23.2", "2.24.0"]})
    report = dr.check_tree(
        tmp_path, fleet_packages={"epistemic-graph"}, backend=backend
    )
    assert report.ok is True
    assert report.checks[0].status == "satisfied"


def test_check_tree_no_fleet_constraints_declared_is_ok(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "solo-repo"
dependencies = ["requests>=2.0"]
"""
    )
    report = dr.check_tree(tmp_path, fleet_packages={"epistemic-graph"})
    assert report.ok is True
    assert report.checks == []


# --------------------------------------------------------------------------- #
# hook_declared -- lets Layer 2 (await_gate_readiness) tell "not yet adopted"
# apart from "adopted and genuinely failing".
# --------------------------------------------------------------------------- #


def test_hook_declared_true_when_present(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "repos:\n- repo: local\n  hooks:\n  - id: dependency-readiness\n"
        "    entry: python -m repository_manager.dependency_readiness .\n"
    )
    assert dr.hook_declared(tmp_path) is True


def test_hook_declared_false_when_absent(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "repos:\n- repo: local\n  hooks:\n  - id: ruff\n"
    )
    assert dr.hook_declared(tmp_path) is False


def test_hook_declared_false_when_no_config_file(tmp_path):
    assert dr.hook_declared(tmp_path) is False


# --------------------------------------------------------------------------- #
# The loud override -- never a silent skip.
# --------------------------------------------------------------------------- #


def test_override_bypasses_failure_but_is_never_silent(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "agent-utilities"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    (tmp_path / ".git").mkdir()
    backend = _FakeBackend({"epistemic-graph": ["2.23.0"]})
    monkeypatch.setenv(dr.OVERRIDE_ENV_VAR, "shipping an urgent fix, eg 2.23.2 not yet cut")

    report = dr.check_tree(tmp_path, fleet_packages={"epistemic-graph"}, backend=backend)

    assert report.ok is True  # the override lets the push proceed
    assert report.overridden is True
    assert report.override_reason == "shipping an urgent fix, eg 2.23.2 not yet cut"

    printed = capsys.readouterr().err
    assert "OVERRIDDEN" in printed
    assert "shipping an urgent fix" in printed

    log_path = tmp_path / ".git" / "dependency-readiness-overrides.log"
    assert log_path.exists()
    record = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert record["reason"] == "shipping an urgent fix, eg 2.23.2 not yet cut"
    assert record["failures"][0]["package"] == "epistemic-graph"


def test_no_override_env_var_means_no_bypass(tmp_path, monkeypatch):
    monkeypatch.delenv(dr.OVERRIDE_ENV_VAR, raising=False)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "agent-utilities"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    backend = _FakeBackend({"epistemic-graph": ["2.23.0"]})
    report = dr.check_tree(tmp_path, fleet_packages={"epistemic-graph"}, backend=backend)
    assert report.ok is False
    assert report.overridden is False


# --------------------------------------------------------------------------- #
# await_gate_readiness -- Layer 2, the phased_push wave barrier. Gate-driven:
# each retry RUNS the downstream repo's own pre-push gate (scripted via an
# injected `run_gate`, never a real pre-commit invocation here) instead of
# polling the package index directly -- see test_gates.py /
# test_pre_push_gate.py for real, unmocked `run_gate_stage` proof, and
# test_phased_push.py for the wave-level (multi-phase) proof.
# --------------------------------------------------------------------------- #


class _Clock:
    """Deterministic fake clock + sleep so a 30-minute barrier is provable
    without waiting 30 real minutes."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def _gate_result(*, success: bool, detail: str = "") -> RepoScanResult:
    """A scripted `run_gate_stage`-shaped result for the HOOK_ID hook."""
    output = f"  [{'OK' if success else 'UNSATISFIED'}] {detail}" if detail else ""
    return RepoScanResult(
        repo_path="irrelevant",
        success=success,
        exit_code=0 if success else 1,
        hooks=[HookResult(hook_id=dr.HOOK_ID, passed=success, output=output)],
        stage="heavy",
    )


def test_await_gate_readiness_resumes_early_once_satisfied(monkeypatch):
    """The downstream repo's gate passes on the SECOND attempt -- must return
    well before the deadline, not wait out the full `wait_minutes` budget."""
    clock = _Clock()
    monkeypatch.setattr(dr, "hook_declared", lambda repo_path: True)
    attempts = {"n": 0}

    def run_gate(repo_path):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return _gate_result(
                success=False,
                detail="epistemic-graph declares >=2.23.2 but only 2.23.0 is available",
            )
        return _gate_result(success=True)

    outcome = dr.await_gate_readiness(
        [("agent-utilities", "/fake/agent-utilities")],
        wait_minutes=30,
        poll_interval_s=30,
        run_gate=run_gate,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert outcome.ok is True
    assert clock.t < 30 * 60  # resumed early, nowhere near the 30-minute budget
    assert attempts["n"] == 2
    assert outcome.targets_checked == ["agent-utilities"]


def test_await_gate_readiness_aborts_when_deadline_passes_unsatisfied(monkeypatch):
    """The gate names the failing repo AND the underlying constraint, never
    just 'failed'."""
    clock = _Clock()
    monkeypatch.setattr(dr, "hook_declared", lambda repo_path: True)

    def run_gate(repo_path):
        return _gate_result(
            success=False,
            detail="epistemic-graph declares >=2.23.2 but only 2.23.0 is available",
        )

    outcome = dr.await_gate_readiness(
        [("agent-utilities", "/fake/agent-utilities")],
        wait_minutes=2,
        poll_interval_s=1,
        run_gate=run_gate,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert outcome.ok is False
    assert outcome.overridden is False
    assert outcome.failures[0].repo_name == "agent-utilities"
    assert "epistemic-graph" in outcome.failures[0].detail
    assert clock.t >= 2 * 60  # genuinely waited out the full ceiling before giving up


def test_await_gate_readiness_override_bypasses_abort_loudly(tmp_path, monkeypatch, capsys):
    clock = _Clock()
    monkeypatch.setattr(dr, "hook_declared", lambda repo_path: True)
    monkeypatch.setenv(dr.OVERRIDE_ENV_VAR, "operator accepted the risk")

    def run_gate(repo_path):
        return _gate_result(success=False, detail="still unsatisfiable")

    outcome = dr.await_gate_readiness(
        [("agent-utilities", "/fake/agent-utilities")],
        wait_minutes=1,
        poll_interval_s=1,
        run_gate=run_gate,
        sleep=clock.sleep,
        now=clock.now,
        audit_repo_path=tmp_path,
    )
    assert outcome.ok is True
    assert outcome.overridden is True
    printed = capsys.readouterr().err
    assert "OVERRIDDEN" in printed
    assert "operator accepted the risk" in printed


def test_await_gate_readiness_no_targets_returns_immediately():
    clock = _Clock()
    outcome = dr.await_gate_readiness([], wait_minutes=30, sleep=clock.sleep, now=clock.now)
    assert outcome.ok is True
    assert outcome.waited_s == 0.0
    assert clock.slept == []  # never even ran the gate once


def test_await_gate_readiness_blocks_when_hook_not_adopted(monkeypatch):
    """A downstream repo that declares a fleet constraint but has not adopted
    the dependency-readiness hook cannot be gate-verified -- it must BLOCK,
    never silently pass (fleet rollout of the hook is gradual)."""
    clock = _Clock()
    monkeypatch.setattr(dr, "hook_declared", lambda repo_path: False)
    calls = {"n": 0}

    def run_gate(repo_path):
        calls["n"] += 1
        return _gate_result(success=True)

    outcome = dr.await_gate_readiness(
        [("legacy-agent", "/fake/legacy-agent")],
        wait_minutes=0.02,  # a ~1.2s ceiling is enough to prove it never blocks forever
        poll_interval_s=1,
        run_gate=run_gate,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert outcome.ok is False
    assert "has not adopted" in outcome.failures[0].detail
    assert calls["n"] == 0  # never even asked to run the gate for an unverifiable repo


def test_await_gate_readiness_backs_off_exponentially(monkeypatch):
    """Retry intervals double (bounded by max_interval_s) rather than
    hammering every downstream repo's gate every `poll_interval_s`."""
    clock = _Clock()
    monkeypatch.setattr(dr, "hook_declared", lambda repo_path: True)

    def run_gate(repo_path):
        return _gate_result(success=False, detail="still unsatisfiable")

    dr.await_gate_readiness(
        [("agent-utilities", "/fake/agent-utilities")],
        wait_minutes=10,
        poll_interval_s=30,
        max_interval_s=120,
        run_gate=run_gate,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert clock.slept[0] == 30
    assert clock.slept[1] == 60
    assert clock.slept[2] == 120
    assert clock.slept[3] == 120  # capped, not 240


# --------------------------------------------------------------------------- #
# resolve_index_urls -- honors configured index, never hardcodes pypi.org.
# --------------------------------------------------------------------------- #


def test_resolve_index_urls_honors_env_over_default(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_INDEX_URL", "https://private.example/simple")
    urls = dr.resolve_index_urls(tmp_path)
    assert urls[0] == "https://private.example/simple"
    assert dr.DEFAULT_INDEX_URL in urls  # default still present as the final fallback


def test_resolve_index_urls_honors_pyproject_uv_index(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_INDEX_URL", raising=False)
    monkeypatch.delenv("PIP_INDEX_URL", raising=False)
    (tmp_path / "pyproject.toml").write_text(
        """
[[tool.uv.index]]
name = "internal"
url = "https://gitlab.example.internal/api/v4/pypi/simple"
default = true
"""
    )
    urls = dr.resolve_index_urls(tmp_path)
    assert urls[0] == "https://gitlab.example.internal/api/v4/pypi/simple"


def test_resolve_index_urls_falls_back_to_pypi_when_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_INDEX_URL", raising=False)
    monkeypatch.delenv("PIP_INDEX_URL", raising=False)
    urls = dr.resolve_index_urls(tmp_path)
    assert urls == [dr.DEFAULT_INDEX_URL]


# --------------------------------------------------------------------------- #
# main() / CLI exit codes -- the pre-commit hook contract.
# --------------------------------------------------------------------------- #


def test_main_exits_nonzero_on_unsatisfiable(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "agent-utilities"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    monkeypatch.setattr(
        dr, "declared_fleet_constraints", lambda *a, **k: [_constraint()]
    )
    monkeypatch.setattr(
        dr,
        "check_constraint",
        lambda *a, **k: dr.ConstraintCheck(
            package="epistemic-graph",
            raw_requirement="epistemic-graph>=2.23.2,<3.0.0",
            specifier=">=2.23.2,<3.0.0",
            declared_by=str(tmp_path / "pyproject.toml"),
            status="unsatisfied",
            latest_available="2.23.0",
            detail="epistemic-graph declares >=2.23.2 but only 2.23.0 is available",
        ),
    )
    rc = dr.main([str(tmp_path)])
    assert rc == 1


def test_main_exits_zero_when_no_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "declared_fleet_constraints", lambda *a, **k: [])
    rc = dr.main([str(tmp_path)])
    assert rc == 0
