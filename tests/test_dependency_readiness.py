"""Tests for the dependency-readiness gate (repository_manager.dependency_readiness).

Covers both layers off the SAME primitives -- no network in these tests: every
test supplies a fake :class:`~repository_manager.dependency_readiness.IndexBackend`
so behavior is proven deterministically, and the live-index proof (does this
correctly fail against scion-sdk' real, currently-unsatisfiable
``epistemic-graph`` constraint, and pass for a satisfiable one) is demonstrated
separately, out of band, against the real PyPI index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from repository_manager import dependency_readiness as dr


class _FakeBackend:
    """A scripted `IndexBackend`: package -> versions, or an exception to raise."""

    def __init__(
        self, table: dict[str, list[str] | Exception], calls: list | None = None
    ):
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


def _constraint(
    package="epistemic-graph", spec=">=2.23.2,<3.0.0", declared_by="pyproject.toml"
):
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
    backend = _FakeBackend(
        {"epistemic-graph": ["2.22.0", "2.23.0", "2.23.2", "2.24.0"]}
    )
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
                package=package,
                index_url=index_url,
                versions=["2.23.2"],
                latest="2.23.2",
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
                package=package,
                index_url=index_url,
                versions=["2.23.2"],
                latest="2.23.2",
            )

    result = dr.check_constraint(
        _constraint(),
        index_urls=["https://primary/simple", "https://secondary/simple"],
        backend=_SecondIndexHasIt({}),
    )
    assert result.status == "satisfied"
    assert result.index_urls_checked == [
        "https://primary/simple",
        "https://secondary/simple",
    ]


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
    constraints = cast(
        list[dr.DeclaredConstraint],
        dr.declared_fleet_constraints(
            tmp_path, fleet_packages={"epistemic-graph", "skill-graphs"}
        ),
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
    constraints = cast(
        list[dr.DeclaredConstraint],
        dr.declared_fleet_constraints(
            tmp_path,
            fleet_packages={"epistemic-graph", "epistemic-graph-stubs"},
        ),
    )
    assert [c.package for c in constraints] == ["epistemic-graph-stubs"]


@pytest.mark.parametrize(
    "content",
    [
        b"\xff\xfe",
        b"[project\nname = 'broken'",
        b"[project]\nname = 7\ndependencies = []\n",
        b"[project]\nname = 'broken'\ndependencies = 'requests'\n",
        (
            b"[project]\nname = 'broken'\ndependencies = []\n"
            b"[project.optional-dependencies]\nextra = 'requests'\n"
        ),
        b"[project]\nname = 'broken'\ndependencies = ['not a requirement @@@']\n",
    ],
)
def test_invalid_dependency_metadata_returns_typed_blocking_state(tmp_path, content):
    (tmp_path / "pyproject.toml").write_bytes(content)

    constraints = dr.declared_fleet_constraints(
        tmp_path, fleet_packages={"epistemic-graph"}
    )

    assert len(constraints) == 1
    invalid = constraints[0]
    assert isinstance(invalid, dr.InvalidDependencyMetadata)
    result = dr.check_constraint(invalid, index_urls=[])
    assert result.status == "invalid_metadata"
    assert result.satisfied is False


def test_invalid_dependency_metadata_blocks_even_with_empty_fleet_scope(tmp_path):
    (tmp_path / "pyproject.toml").write_bytes(b"\xff")

    constraints = dr.declared_fleet_constraints(tmp_path, fleet_packages=set())

    assert len(constraints) == 1
    assert isinstance(constraints[0], dr.InvalidDependencyMetadata)


def test_fleet_package_names_excludes_infra_subdirectories(tmp_path):
    """A `services/`-tree repo (e.g. a Docker stack literally named `langfuse`)
    must never collide with an unrelated third-party PyPI package of the same
    name -- this was caught live: `services/langfuse.git` (a Docker Swarm
    stack) vs. the third-party `langfuse` SDK scion-sdk depends on."""
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
      project: scion-sdk
"""
        )
        names = dr.fleet_package_names(manifest)
        assert "scion-sdk" in names


@pytest.mark.parametrize(
    "manifest_body",
    [
        "subdirectories: invalid\n",
        "maintenance:\n  phases:\n    - null\n",
        "maintenance:\n  phases:\n    - projects: scion-sdk\n",
        (
            "subdirectories:\n"
            "  agent-packages:\n"
            "    repositories:\n"
            "      - url: https://example.invalid/org/../epistemic-graph.git\n"
        ),
        (
            "subdirectories:\n"
            "  agent-packages:\n"
            "    repositories:\n"
            "      - url: https://example.invalid/org/EPİSTEMIC-GRAPH.git\n"
        ),
        (
            "subdirectories:\n"
            "  agent-packages:\n"
            "    repositories:\n"
            "      - url: ' https://example.invalid/org/epistemic-graph.git'\n"
        ),
        "maintenance:\n  phases:\n    - projects: [org/scion-sdk]\n",
    ],
)
def test_malformed_manifest_is_typed_blocking_state(
    tmp_path, manifest_body: str
) -> None:
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(manifest_body)

    assert dr.fleet_package_names(manifest) == set()
    constraints = dr.declared_fleet_constraints(
        tmp_path,
        workspace_yml_path=manifest,
    )

    assert len(constraints) == 1
    invalid = constraints[0]
    assert isinstance(invalid, dr.InvalidDependencyMetadata)
    assert invalid.declared_by == str(manifest)
    assert "workspace manifest is malformed" in invalid.detail
    assert dr.check_constraint(invalid, index_urls=[]).status == "invalid_metadata"


def test_non_distribution_repository_is_a_valid_manifest_entry(tmp_path) -> None:
    """An org ``.github`` repository is tracked but can never be a dependency."""

    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    repositories:\n"
        "      - url: https://example.invalid/org/.github.git\n"
        "      - url: https://example.invalid/org/epistemic-graph.git\n"
    )

    assert dr.fleet_package_names(manifest) == {".github", "epistemic-graph"}
    assert not any(
        isinstance(item, dr.InvalidDependencyMetadata)
        for item in dr.declared_fleet_constraints(tmp_path, workspace_yml_path=manifest)
    )


def test_external_and_broken_pyproject_symlinks_are_typed_blockers(tmp_path) -> None:
    external = tmp_path / "external-pyproject.toml"
    external.write_text("[project]\nname='repo'\ndependencies=[]\n")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.symlink_to(external)

    constraints = dr.declared_fleet_constraints(tmp_path, fleet_packages=set())
    assert len(constraints) == 1
    assert isinstance(constraints[0], dr.InvalidDependencyMetadata)
    assert "must not be a symlink" in constraints[0].detail

    real_repo = tmp_path / "real-repo"
    real_repo.mkdir()
    (real_repo / "pyproject.toml").write_text(
        "[project]\nname='repo'\ndependencies=[]\n"
    )
    linked_repo = tmp_path / "repo-link"
    linked_repo.symlink_to(real_repo, target_is_directory=True)
    constraints = dr.declared_fleet_constraints(linked_repo, fleet_packages=set())
    assert len(constraints) == 1
    assert isinstance(constraints[0], dr.InvalidDependencyMetadata)
    assert "symlink component" in constraints[0].detail

    pyproject.unlink()
    pyproject.symlink_to(tmp_path / "missing-pyproject.toml")
    constraints = dr.declared_fleet_constraints(tmp_path, fleet_packages=set())
    assert len(constraints) == 1
    assert isinstance(constraints[0], dr.InvalidDependencyMetadata)
    assert "must not be a symlink" in constraints[0].detail


def test_broken_manifest_symlink_is_typed_blocker(tmp_path) -> None:
    manifest = tmp_path / "workspace.yml"
    manifest.symlink_to(tmp_path / "missing-workspace.yml")

    assert dr.fleet_package_names(manifest) == set()
    constraints = dr.declared_fleet_constraints(
        tmp_path,
        workspace_yml_path=manifest,
    )

    assert len(constraints) == 1
    assert isinstance(constraints[0], dr.InvalidDependencyMetadata)
    assert "manifest must not be a symlink" in constraints[0].detail


# --------------------------------------------------------------------------- #
# check_tree -- Layer 1, the pre-push hook entrypoint
# --------------------------------------------------------------------------- #


def test_check_tree_fails_on_unsatisfiable_fleet_constraint(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "scion-sdk"
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
name = "scion-sdk"
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
# hook_declared -- lets a caller tell "not yet adopted"
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
name = "scion-sdk"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    (tmp_path / ".git").mkdir()
    backend = _FakeBackend({"epistemic-graph": ["2.23.0"]})
    monkeypatch.setenv(
        dr.OVERRIDE_ENV_VAR, "shipping an urgent fix, eg 2.23.2 not yet cut"
    )

    report = dr.check_tree(
        tmp_path, fleet_packages={"epistemic-graph"}, backend=backend
    )

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
name = "scion-sdk"
dependencies = ["epistemic-graph[full]>=2.23.2,<3.0.0"]
"""
    )
    backend = _FakeBackend({"epistemic-graph": ["2.23.0"]})
    report = dr.check_tree(
        tmp_path, fleet_packages={"epistemic-graph"}, backend=backend
    )
    assert report.ok is False
    assert report.overridden is False


def test_repository_readiness_hook_is_manual_only():
    """A routine push cannot become a package-index availability check."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] == dr.HOOK_ID
    )
    assert hook["stages"] == ["manual"]


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
url = "https://example-gitlab.internal/api/v4/pypi/simple"
default = true
"""
    )
    urls = dr.resolve_index_urls(tmp_path)
    assert urls[0] == "https://example-gitlab.internal/api/v4/pypi/simple"


def test_resolve_index_urls_falls_back_to_pypi_when_nothing_configured(
    tmp_path, monkeypatch
):
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
name = "scion-sdk"
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


# --------------------------------------------------------------------------- #
# _version_file_summary / PyPISimpleIndexBackend -- the wheel rule
# (pipelines PIPE-RELEASE-R008): a compatible wheel makes a version available.
# --------------------------------------------------------------------------- #

_PURE = "py3-none-any"
_FILES = {
    "files": [
        {"filename": f"epistemic_graph-2.23.2-{_PURE}.whl", "yanked": False},
        {"filename": "epistemic_graph-2.23.2.tar.gz", "yanked": False},
        {"filename": f"epistemic_graph-2.23.3-{_PURE}.whl", "yanked": False},
        {"filename": "epistemic_graph-2.23.4.tar.gz", "yanked": False},
        {"filename": "epistemic_graph-2.23.5-cp399-cp399-win_arm64.whl"},
        {"filename": f"epistemic_graph-2.23.6-{_PURE}.whl", "yanked": True},
        {"filename": "epistemic_graph-2.23.6.tar.gz", "yanked": True},
    ]
}


@pytest.mark.spec("RM-RELEASE-R005")
def test_version_file_summary_judges_by_compatible_wheel():
    summary = dr._version_file_summary(_FILES)
    assert summary["2.23.2"].available and not summary["2.23.2"].sdist_missing
    assert summary["2.23.3"].available and summary["2.23.3"].sdist_missing
    assert summary["2.23.4"].source_only and not summary["2.23.4"].available
    assert summary["2.23.5"].incompatible and not summary["2.23.5"].available
    assert summary["2.23.6"].all_yanked and not summary["2.23.6"].available


class _FakeHTTPResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


@pytest.mark.spec("RM-RELEASE-R005")
def test_pypi_backend_buckets_versions_by_the_wheel_rule(monkeypatch):
    """The real PyPISimpleIndexBackend path: a wheel-only release is available
    (with sdist_missing), an sdist-only one is source_only, and a wheel for
    another platform is incompatible -- never folded into `versions`."""
    monkeypatch.setattr(
        dr.requests, "get", lambda *a, **k: _FakeHTTPResponse(200, _FILES)
    )
    backend = dr.PyPISimpleIndexBackend()
    available = backend.available_versions(
        "epistemic-graph", index_url="https://pypi.org/simple"
    )
    assert available.versions == ["2.23.2", "2.23.3"]
    assert available.latest == "2.23.3"
    assert available.sdist_missing_versions == ["2.23.3"]
    assert available.source_only_versions == ["2.23.4"]
    assert available.incompatible_versions == ["2.23.5"]
    assert available.yanked_versions == ["2.23.6"]


@pytest.mark.spec("RM-RELEASE-R005")
def test_pypi_backend_honours_the_target_tags(monkeypatch):
    from packaging.tags import Tag

    monkeypatch.setattr(
        dr.requests, "get", lambda *a, **k: _FakeHTTPResponse(200, _FILES)
    )
    backend = dr.PyPISimpleIndexBackend([Tag("cp399", "cp399", "win_arm64")])
    available = backend.available_versions(
        "epistemic-graph", index_url="https://pypi.org/simple"
    )
    assert available.versions == ["2.23.5"]


# --------------------------------------------------------------------------- #
# check_constraint -- sdist_missing advisory, source_only, allow-source-build,
# partial_publish only while the release run is in progress.
# --------------------------------------------------------------------------- #


class _RichFakeBackend:
    """Returns a caller-supplied AvailableVersions verbatim."""

    def __init__(self, available):
        self.available = available

    def available_versions(self, package, *, index_url, timeout=10.0):
        return self.available


def _available(**buckets):
    return dr.AvailableVersions(
        package="epistemic-graph",
        index_url="https://pypi.org/simple",
        versions=buckets.pop("versions", ["2.22.4"]),
        latest=buckets.pop("latest", "2.22.4"),
        **buckets,
    )


def _check(available, **kwargs):
    return dr.check_constraint(
        _constraint(),
        index_urls=["https://pypi.org/simple"],
        backend=_RichFakeBackend(available),
        **kwargs,
    )


@pytest.mark.spec("RM-RELEASE-R005")
def test_wheel_without_sdist_passes_with_the_sdist_missing_advisory():
    result = _check(
        _available(
            versions=["2.22.4", "2.23.2"],
            latest="2.23.2",
            sdist_missing_versions=["2.23.2"],
        )
    )
    assert result.status == "satisfied" and result.satisfied
    assert result.matching_version == "2.23.2"
    assert result.advisories == [dr.SDIST_MISSING]
    assert "sdist_missing" in result.detail


@pytest.mark.spec("RM-RELEASE-R005")
def test_wheel_with_sdist_passes_without_advisory():
    result = _check(_available(versions=["2.23.2"], latest="2.23.2"))
    assert result.satisfied and result.advisories == []


@pytest.mark.spec("RM-RELEASE-R005")
def test_sdist_only_version_fails():
    result = _check(
        _available(source_only_versions=["2.23.2"]),
        release_run_state=lambda package, version: "completed",
    )
    assert result.status == "source_only"
    assert result.satisfied is False
    assert result.unavailable_version == "2.23.2"
    assert result.partial_publish_version is None
    assert "allow-source-build" in result.detail


@pytest.mark.spec("RM-RELEASE-R005")
def test_sdist_only_version_on_the_allow_list_passes():
    result = _check(
        _available(source_only_versions=["2.23.2"]),
        allow_source_build=["Epistemic_Graph"],
    )
    assert result.satisfied
    assert result.matching_version == "2.23.2"
    assert "allow-source-build" in result.detail


@pytest.mark.spec("RM-RELEASE-R005")
def test_no_compatible_wheel_fails():
    result = _check(_available(incompatible_versions=["2.23.2"]))
    assert result.status == "no_compatible_wheel"
    assert result.satisfied is False


@pytest.mark.spec("RM-RELEASE-R005")
def test_in_progress_release_run_reports_partial_publish():
    seen = []

    def probe(package, version):
        seen.append((package, version))
        return "in_progress"

    result = _check(
        _available(source_only_versions=["2.23.2"]), release_run_state=probe
    )
    assert result.status == "partial_publish"
    assert result.satisfied is False
    assert result.partial_publish_version == "2.23.2"
    assert "in flight" in result.detail
    assert seen == [("epistemic-graph", "2.23.2")]


@pytest.mark.spec("RM-RELEASE-R005")
def test_concluded_release_run_is_judged_by_the_wheel_rule():
    result = _check(
        _available(source_only_versions=["2.23.2"]),
        release_run_state=lambda package, version: "completed",
    )
    assert result.status == "source_only"


@pytest.mark.spec("RM-RELEASE-R005")
def test_concluded_run_with_a_wheel_passes_without_asking_the_forge():
    def probe(package, version):
        raise AssertionError("a satisfied check must not query the forge")

    result = _check(
        _available(
            versions=["2.23.2"], latest="2.23.2", sdist_missing_versions=["2.23.2"]
        ),
        release_run_state=probe,
    )
    assert result.satisfied and result.advisories == [dr.SDIST_MISSING]


@pytest.mark.spec("RM-RELEASE-R005")
def test_failing_probe_degrades_to_the_wheel_rule():
    def probe(package, version):
        raise RuntimeError("forge down")

    result = _check(
        _available(source_only_versions=["2.23.2"]), release_run_state=probe
    )
    assert result.status == "source_only"


@pytest.mark.spec("RM-RELEASE-R005")
def test_allow_source_build_reads_exact_names_only(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pipelines]\nallow-source-build = ["Uptime_Kuma.API", ":all:"]\n'
    )
    assert dr.allow_source_build(tmp_path) == frozenset({"uptime-kuma-api"})
    assert dr.allow_source_build(tmp_path / "missing") == frozenset()


@pytest.mark.spec("RM-RELEASE-R005")
def test_check_tree_applies_the_allow_list(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "consumer"\n'
        '[tool.pipelines]\nallow-source-build = ["epistemic-graph"]\n'
    )
    monkeypatch.setattr(
        dr, "declared_fleet_constraints", lambda *a, **k: [_constraint()]
    )
    report = dr.check_tree(
        tmp_path,
        backend=_RichFakeBackend(_available(source_only_versions=["2.23.2"])),
        release_run_state=lambda package, version: "completed",
    )
    assert report.ok
    assert report.checks[0].matching_version == "2.23.2"


def test_check_constraint_reports_yanked_but_would_match_on_unsatisfied():
    available = dr.AvailableVersions(
        package="epistemic-graph",
        index_url="https://pypi.org/simple",
        versions=["2.22.4"],
        latest="2.22.4",
        yanked_versions=["2.23.2"],
    )
    result = dr.check_constraint(
        _constraint(),
        index_urls=["https://pypi.org/simple"],
        backend=_RichFakeBackend(available),
    )
    assert result.status == "unsatisfied"
    assert result.yanked_but_would_match == ["2.23.2"]
    assert "2.23.2" in result.detail
    assert "yanked" in result.detail


# --------------------------------------------------------------------------- #
# RM-GOVERNANCE-R018: path resolution must not depend on running from the
# primary repository location. repository-manager's own pre-commit/pre-push
# hook entry (the self-hosting exception documented in
# scripts/sweep_dependency_readiness_hook.py's module docstring: this repo
# already has itself installed locally, so its hook invokes
# `python -m repository_manager.dependency_readiness .` directly rather than
# resolving a sibling checkout through `uv run --with`) never walks $PWD for
# a sibling repo -- but _find_workspace_manifest's own upward walk from
# `start` could still be fooled by an isolated branch checkout living outside
# the canonical workspace tree (a linked git worktree routinely lives under a
# separate, unrelated scratch directory), if the packaged copy it checks
# first were ever missing or bypassed. These tests pin that it is not: the
# packaged copy resolves
# first, from a `start` that has no workspace.yml anywhere above it.
# --------------------------------------------------------------------------- #


def test_find_workspace_manifest_resolves_from_a_location_with_no_ancestor_copy(
    tmp_path,
):
    """An isolated checkout's `start` (no workspace.yml anywhere above it,
    unlike the primary repository location under agent-packages/) still
    resolves -- the packaged copy next to this installed module is checked
    before any upward walk from `start`."""
    outside_any_workspace = tmp_path / "isolated-worktree" / "nested" / "deep"
    outside_any_workspace.mkdir(parents=True)

    manifest = dr._find_workspace_manifest(outside_any_workspace)

    assert manifest is not None
    assert manifest == Path(dr.__file__).parent / "workspace.yml"
    assert manifest.is_file()


def test_dependency_readiness_hook_exits_zero_from_an_isolated_worktree(tmp_path):
    """The actual hook entry point, run with an isolated-checkout-shaped
    path (not the primary repository location), resolves and reports --
    never a path-resolution crash. No declared fleet constraint this repo's
    own pyproject.toml carries is exercised against a live index here (that
    proof is the documented out-of-band one at the top of this file); an
    empty constraint set after real path resolution is exit 0."""
    isolated_worktree = tmp_path / "isolated-worktree"
    isolated_worktree.mkdir()

    rc = dr.main(["--json", str(isolated_worktree)])

    assert rc == 0
