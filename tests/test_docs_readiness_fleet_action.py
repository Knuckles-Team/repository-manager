"""Focused contracts for the NE137 documentation-readiness fleet action."""

from __future__ import annotations

import asyncio
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from repository_manager import docs_readiness
from repository_manager.cli_commands.docs_readiness import run_docs_readiness_cli
from repository_manager.mcp_tools.context import McpToolContext
from repository_manager.mcp_tools.docs_readiness import (
    register_docs_readiness_tools,
)

PRODUCTION_FLEET_COUNT = 75


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Readiness Test"],
        check=True,
    )
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def _commit_generated_artifacts(path: Path) -> None:
    (path / "llms.txt").write_text("fixture\n", encoding="utf-8")
    (path / "agent-readiness-manifest.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "add",
            "llms.txt",
            "agent-readiness-manifest.json",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "generated artifacts"],
        check=True,
    )


def _ignore_local_build_dirs(path: Path) -> None:
    (path / ".git" / "info" / "exclude").write_text(
        ".venv/\ntarget/\n", encoding="utf-8"
    )


def _manifest(root: Path) -> None:
    lines = ["name: Fixture", "path: .", "repositories:"]
    lines.extend(
        [
            "  - url: https://example.invalid/pipelines.git",
            "    description: pipelines",
            "subdirectories:",
            "  agent-packages:",
            "    subdirectories:",
            "      agents:",
            "        repositories:",
            "          - url: https://example.invalid/provider.git",
            "            description: provider",
            "  services:",
            "    repositories:",
            "      - url: https://example.invalid/shared-scaffold.git",
            "        description: shared scaffolding",
        ]
    )
    (root / "workspace.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_generator(calls: list[tuple[Path, bool]]):
    def generate(
        root: Path,
        *,
        check: bool,
        adopt_existing: bool,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        assert adopt_existing is (output_dir is not None)
        calls.append((root, check))
        if not check:
            destination = output_dir or root
            (destination / "llms.txt").write_text("fixture\n", encoding="utf-8")
            (destination / "agent-readiness-manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
        return {
            "schema_version": "agent-readiness/v1",
            "generator_version": "fixture-1",
            "generated": ["llms.txt"],
            "planned": ["llms.txt"] if check and not calls else [],
            "pruned": [],
            "provenance": {"generator_version": "fixture-1"},
        }

    return generate


def _valid_readiness() -> dict[str, Any]:
    return {
        "schema_version": "agent-readiness/v1",
        "project": {"name": "provider", "kind": "package"},
        "applicability": {
            "content": True,
            "discoverability": True,
            "access_policy": True,
            "capabilities": True,
            "errors": False,
            "provenance": True,
            "measurement": False,
            "deployment": False,
        },
        "standards": [
            {"id": "RFC 3986", "kind": "rfc", "level": "normative"},
            {"id": "Docs Draft", "kind": "draft", "level": "draft"},
            {"id": "Concept IDs", "kind": "convention", "level": "advisory"},
        ],
        "content_signals": {"policy": "unset"},
        "budgets": {"curated_chars": 12000, "summary_chars": 500, "full_chars": 8000},
        "capabilities": {
            "api": {"applicable": False},
            "mcp": {"applicable": False},
            "a2a": {"applicable": False},
            "skills": {"applicable": True, "path": "skills"},
        },
    }


@pytest.fixture
def fixture_workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    repo = root / "agent-packages" / "agents" / "provider"
    repo.mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "docs" / "index.md").write_text(
        "# Home\n\nFixture docs.\n", encoding="utf-8"
    )
    (repo / "docs" / "agent-readiness.json").write_text(
        json.dumps(_valid_readiness()) + "\n", encoding="utf-8"
    )
    (repo / "skills" / "fixture").mkdir(parents=True)
    (repo / "skills" / "fixture" / "SKILL.md").write_text(
        "---\nname: fixture\ndescription: Fixture skill\n---\n", encoding="utf-8"
    )
    (repo / "mkdocs.yml").write_text(
        "site_name: Fixture\nsite_url: https://docs.example.invalid/\n"
        "nav:\n  - Home: index.md\n",
        encoding="utf-8",
    )
    _init_repo(repo)
    _manifest(root)
    return root, repo


@pytest.fixture(autouse=True)
def _one_repository_fleet_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep small fixtures explicit while production remains pinned to 75."""

    monkeypatch.setattr(docs_readiness, "EXPECTED_AGENT_FLEET_COUNT", 1)


def _fixture_dispatch(action: str = "preview", **kwargs: Any) -> dict[str, Any]:
    """Run the one-repository fixture under its explicit selection contract."""

    return docs_readiness.dispatch(action, **kwargs)


def test_default_generator_is_the_canonical_universal_skills_builder(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace
    result = _fixture_dispatch(
        workspace_root=root,
        repository="agent-packages/agents/provider",
    )

    assert result["ok"] is True
    row = result["repositories"][0]
    assert row["status"] == "planned"
    assert row["generator_version"] == "1.0.0"
    assert str(root) not in json.dumps(result)
    assert not (root / "agent-packages" / "agents" / "provider" / "llms.txt").exists()


def test_default_apply_verifies_real_generator_output_dir_contract(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace

    result = _fixture_dispatch(
        "apply",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        confirm=True,
    )

    assert result["ok"] is True
    assert result["repositories"][0]["status"] == "applied"
    assert (repo / "llms.txt").is_file()
    assert (repo / "agent-readiness-manifest.json").is_file()


def test_preview_is_read_only_manifest_scoped_and_excludes_scaffolding(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    calls: list[tuple[Path, bool]] = []
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    result = _fixture_dispatch(
        workspace_root=root,
        _generator=_fake_generator(calls),
    )

    assert result["ok"] is True
    assert [row["status"] for row in result["repositories"]] == ["planned"]
    assert calls == [(repo, True)]
    assert (
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == before
    )
    assert str(root) not in json.dumps(result)


def test_apply_requires_exact_identity_and_confirmation(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace
    calls: list[tuple[Path, bool]] = []
    generator = _fake_generator(calls)

    missing_selection = _fixture_dispatch(
        "apply", workspace_root=root, confirm=True, _generator=generator
    )
    no_confirmation = _fixture_dispatch(
        "apply",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=generator,
    )
    applied = _fixture_dispatch(
        "apply",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        confirm=True,
        _generator=generator,
    )

    assert missing_selection["error_code"] == "apply-requires-exact-repository"
    assert no_confirmation["error_code"] == "apply-confirmation-required"
    assert applied["ok"] is True
    assert applied["repositories"][0]["status"] == "applied"
    assert calls[0] == (root / "agent-packages" / "agents" / "provider", False)
    assert len(calls) == 2 and calls[1][1] is False


def test_dirty_repo_is_refused_before_generator(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    (repo / "uncommitted.txt").write_text("wip\n", encoding="utf-8")
    calls: list[tuple[Path, bool]] = []

    result = _fixture_dispatch(
        "preview",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=_fake_generator(calls),
    )

    assert result["ok"] is False
    assert result["repositories"][0]["reason"] == "repository-dirty"
    assert calls == []


def test_missing_applicability_is_a_privacy_safe_block(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    subprocess.run(
        ["git", "-C", str(repo), "rm", "-q", "docs/agent-readiness.json"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "remove applicability"],
        check=True,
    )
    calls: list[tuple[Path, bool]] = []
    result = _fixture_dispatch(
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=_fake_generator(calls),
    )

    assert result["ok"] is False
    assert result["repositories"][0]["reason"] == "applicability-containment"
    assert calls == []
    assert str(repo) not in json.dumps(result)


def test_manifest_secret_fields_are_refused_before_selection(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace
    (root / "workspace.yml").write_text(
        "name: Fixture\npath: .\napi_key: should-not-be-accepted\n",
        encoding="utf-8",
    )

    result = _fixture_dispatch(
        workspace_root=root,
        _generator=_fake_generator([]),
    )

    assert result["ok"] is False
    assert result["error_code"] == "workspace-manifest-invalid"
    assert "should-not-be-accepted" not in json.dumps(result)


def test_exact_identity_rejects_traversal_and_out_of_manifest(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace
    for value, expected in (
        ("../provider", "repository-identity-invalid"),
        ("agent-packages/./agents/provider", "repository-identity-invalid"),
        ("agent-packages/agents/other", "repository-not-in-manifest"),
        ("services/shared-scaffold", "repository-not-in-manifest"),
        (
            str(root / "agent-packages" / "agents" / "provider"),
            "repository-identity-invalid",
        ),
    ):
        result = _fixture_dispatch(
            workspace_root=root,
            repository=value,
            _generator=_fake_generator([]),
        )
        assert result["error_code"] == expected


def test_symlinked_manifest_target_fails_closed(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    real = repo.with_name("provider-real")
    repo.rename(real)
    repo.symlink_to(real, target_is_directory=True)

    result = _fixture_dispatch(
        workspace_root=root,
        _generator=_fake_generator([]),
    )

    assert result["ok"] is False
    assert result["error_code"] == "repository-path-symlink"
    assert str(real) not in json.dumps(result)


def test_ambiguous_canonical_authority_is_refused(
    fixture_workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = fixture_workspace

    def ambiguous() -> Any:
        raise docs_readiness.DocsReadinessError("generator-authority-ambiguous")

    monkeypatch.setattr(docs_readiness, "_canonical_generator", ambiguous)
    result = _fixture_dispatch(
        workspace_root=root,
        repository="agent-packages/agents/provider",
    )

    assert result == {
        "ok": False,
        "action": "preview",
        "error_code": "generator-authority-ambiguous",
    }


def test_generator_exception_cannot_publish_paths_or_exception_text(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace

    def fail(root: Path, *, check: bool, adopt_existing: bool) -> dict[str, Any]:
        del check, adopt_existing
        raise ValueError(f"{root}/credential-value")

    result = _fixture_dispatch(
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=fail,
    )

    assert result["ok"] is False
    assert result["repositories"][0]["reason"] == "generator-failed"
    assert str(root) not in json.dumps(result)
    assert "credential-value" not in json.dumps(result)


def test_verify_requires_current_idempotent_generator_plan(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    _commit_generated_artifacts(repo)

    def current(
        generator_root: Path,
        *,
        check: bool,
        adopt_existing: bool,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        assert check is False
        assert adopt_existing is (output_dir is not None)
        destination = output_dir or generator_root
        (destination / "llms.txt").write_text("fixture\n", encoding="utf-8")
        (destination / "agent-readiness-manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "schema_version": "agent-readiness/v1",
            "generator_version": "fixture-1",
            "generated": ["llms.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {"generator_version": "fixture-1"},
        }

    result = _fixture_dispatch(
        "verify",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=current,
    )
    assert result["ok"] is True
    assert result["repositories"][0]["status"] == "verified"
    assert result["repositories"][0]["generator_version"] == "fixture-1"


def test_apply_failure_rolls_back_only_owned_artifacts(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    _commit_generated_artifacts(repo)
    before_llms = (repo / "llms.txt").read_bytes()
    before_manifest = (repo / "agent-readiness-manifest.json").read_bytes()

    def fail(
        generator_root: Path, *, check: bool, adopt_existing: bool
    ) -> dict[str, Any]:
        del check, adopt_existing
        (generator_root / "llms.txt").write_bytes(b"partial publication\n")
        (generator_root / "agent-readiness-manifest.json").write_bytes(
            b"partial publication\n"
        )
        raise ValueError("publication failed")

    result = _fixture_dispatch(
        "apply",
        workspace_root=root,
        repository="agent-packages/agents/provider",
        confirm=True,
        _generator=fail,
    )

    assert result["ok"] is False
    assert result["repositories"][0]["reason"] == "generator-failed"
    assert (repo / "llms.txt").read_bytes() == before_llms
    assert (repo / "agent-readiness-manifest.json").read_bytes() == before_manifest
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout == ""


def test_generator_output_escape_is_rejected_without_durable_path(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, _ = fixture_workspace

    def escape(root: Path, *, check: bool, adopt_existing: bool) -> dict[str, Any]:
        del root, check, adopt_existing
        return {
            "generated": ["../secret.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {},
        }

    result = _fixture_dispatch(
        workspace_root=root,
        repository="agent-packages/agents/provider",
        _generator=escape,
    )
    assert result["ok"] is False
    assert result["repositories"][0]["reason"] == "generator-outputs-invalid"
    assert str(root) not in json.dumps(result)


def test_default_fleet_is_exact_manifest_agent_packages_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    identities = docs_readiness._manifest_repositories(
        root / "repository_manager" / "workspace.yml", root
    )
    selected = docs_readiness._select_repositories(
        identities, None, expected_count=PRODUCTION_FLEET_COUNT
    )
    selected_ids = {item.identifier for item in selected}

    assert len(selected) == 75
    assert selected_ids == {
        item.identifier
        for item in identities
        if item.identifier.startswith("agent-packages/")
        and item.identifier != "agent-packages/agents/tests"
    }
    assert "agent-packages/skills/universal-skills" in selected_ids
    assert "agent-packages/skills/skill-graphs" in selected_ids
    assert "agent-packages/agents/tests" not in selected_ids


def test_manifest_selection_drift_fails_closed_before_generator(
    fixture_workspace: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = fixture_workspace
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        docs_readiness, "EXPECTED_AGENT_FLEET_COUNT", PRODUCTION_FLEET_COUNT
    )

    result = _fixture_dispatch(
        workspace_root=root,
        _generator=_fake_generator(calls),
    )

    assert result == {
        "ok": False,
        "action": "preview",
        "error_code": "fleet-selection-drift",
    }
    assert calls == []


def test_manifest_checkout_drift_fails_closed_before_generator(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    repo.rename(repo.with_name("provider-parked"))
    calls: list[tuple[Path, bool]] = []

    result = _fixture_dispatch(
        workspace_root=root,
        _generator=_fake_generator(calls),
    )

    assert result == {
        "ok": False,
        "action": "preview",
        "error_code": "fleet-selection-drift",
    }
    assert calls == []


def test_default_fleet_does_not_discover_unlisted_sibling(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "agent-packages" / "agents" / "ghost").mkdir(parents=True)
    repo = root / "agent-packages" / "agents" / "provider"
    repo.mkdir(parents=True)
    manifest = root / "workspace.yml"
    manifest.write_text(
        "name: Fixture\npath: .\nrepositories: []\nsubdirectories:\n"
        "  agent-packages:\n    subdirectories:\n      agents:\n"
        "        repositories:\n          - url: https://example.invalid/provider.git\n",
        encoding="utf-8",
    )

    identities = docs_readiness._manifest_repositories(manifest, root)
    selected = docs_readiness._select_repositories(identities, None)

    assert [item.identifier for item in selected] == ["agent-packages/agents/provider"]


def test_verify_uses_bounded_staging_and_does_not_copy_or_mutate_source(
    fixture_workspace: tuple[Path, Path],
) -> None:
    root, repo = fixture_workspace
    _ignore_local_build_dirs(repo)
    (repo / ".venv").mkdir()
    (repo / ".venv" / "ignored.bin").write_bytes(b"ignored")
    (repo / "target").mkdir()
    (repo / "target" / "ignored.bin").write_bytes(b"ignored")
    _commit_generated_artifacts(repo)
    calls: list[tuple[Path, Path]] = []

    def generate(
        generator_root: Path,
        *,
        output_dir: Path,
        check: bool,
        adopt_existing: bool,
    ) -> dict[str, Any]:
        assert generator_root == repo
        assert output_dir != generator_root
        assert check is False and adopt_existing is True
        calls.append((generator_root, output_dir))
        (output_dir / "llms.txt").write_text("fixture\n", encoding="utf-8")
        (output_dir / "agent-readiness-manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "schema_version": "agent-readiness/v1",
            "generator_version": "fixture-1",
            "generated": ["llms.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {"generator_version": "fixture-1"},
        }

    result = docs_readiness._verify_current(generate, repo)

    assert result["ok"] is True
    assert len(calls) == 1
    assert not any(path.name == "repository" for path in calls[0][1].parents)
    assert (repo / ".venv" / "ignored.bin").read_bytes() == b"ignored"


def test_verify_refuses_generator_source_tree_mutation(
    fixture_workspace: tuple[Path, Path],
) -> None:
    _, repo = fixture_workspace
    _commit_generated_artifacts(repo)

    def mutate(
        generator_root: Path,
        *,
        output_dir: Path,
        check: bool,
        adopt_existing: bool,
    ) -> dict[str, Any]:
        del check, adopt_existing
        (generator_root / "README.md").write_text("unexpected\n", encoding="utf-8")
        (output_dir / "llms.txt").write_text("fixture\n", encoding="utf-8")
        (output_dir / "agent-readiness-manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "generated": ["llms.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {},
        }

    result = docs_readiness._verify_current(mutate, repo)

    assert result == {"ok": False, "error_code": "verification-mutated-target"}


def test_verify_rejects_undeclared_staging_output(
    fixture_workspace: tuple[Path, Path],
) -> None:
    _, repo = fixture_workspace

    def generate(
        generator_root: Path,
        *,
        output_dir: Path,
        check: bool,
        adopt_existing: bool,
    ) -> dict[str, Any]:
        del generator_root, check, adopt_existing
        (output_dir / "secret.txt").write_text("not an artifact\n", encoding="utf-8")
        return {
            "generated": ["llms.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {},
        }

    result = docs_readiness._verify_current(generate, repo)

    assert result == {"ok": False, "error_code": "output-outside-contract"}


@pytest.mark.parametrize(
    "written",
    ({"agent-readiness-manifest.json"}, {"llms.txt"}),
    ids=("missing-declared-output", "missing-provenance-manifest"),
)
def test_verify_requires_exact_declared_staging_outputs(
    fixture_workspace: tuple[Path, Path], written: set[str]
) -> None:
    _, repo = fixture_workspace

    def generate(
        generator_root: Path,
        *,
        output_dir: Path,
        check: bool,
        adopt_existing: bool,
    ) -> dict[str, Any]:
        del generator_root, check, adopt_existing
        for relative in written:
            (output_dir / relative).write_text("fixture\n", encoding="utf-8")
        return {
            "generated": ["llms.txt"],
            "planned": [],
            "pruned": [],
            "provenance": {},
        }

    result = docs_readiness._verify_current(generate, repo)

    assert result == {"ok": False, "error_code": "output-contract-incomplete"}


def test_artifact_snapshot_has_file_and_byte_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    sections = output / "llms-sections" / "section"
    sections.mkdir(parents=True)
    for index in range(docs_readiness.MAX_OUTPUTS + 1):
        (sections / f"{index}.txt").write_bytes(b"x")

    with pytest.raises(
        docs_readiness.DocsReadinessError, match="output-count-exceeded"
    ):
        docs_readiness._artifact_files(output)

    small = tmp_path / "small"
    small.mkdir()
    (small / "llms.txt").write_bytes(b"four")
    monkeypatch.setattr(docs_readiness, "MAX_OUTPUT_BYTES", 3)
    with pytest.raises(docs_readiness.DocsReadinessError, match="output-oversize"):
        docs_readiness._artifact_files(small)


def test_mcp_and_cli_register_one_shared_action_surface(
    fixture_workspace: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root, _ = fixture_workspace
    observed: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
        observed.append((action, kwargs))
        return {"ok": True, "action": action}

    monkeypatch.setattr(docs_readiness, "dispatch", fake_dispatch)
    cli_args = Namespace(
        docs_readiness="preview",
        docs_readiness_repository="agent-packages/agents/provider",
        docs_readiness_confirm=False,
        workspace=str(root),
        file="workspace.yml",
    )
    assert run_docs_readiness_cli(cli_args) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "preview"

    from fastmcp import FastMCP

    context = McpToolContext(
        get_git_instance=lambda: None,
        resolve_repo_dir=lambda *_: "",
        resolve_commit_code_target=lambda *_: (None, None),
        submit_job=lambda *_args, **_kwargs: {},
        cancel_job=lambda _job: {},
        get_job_status=lambda *_args, **_kwargs: {},
        last_failed_repos=lambda: [],
        wait_for_jobs_and_run=lambda *_args, **_kwargs: None,
        jobs={},
        jobs_lock=__import__("threading").RLock(),
        default_workspace=str(root),
        default_workspace_yml="workspace.yml",
    )
    mcp = FastMCP("fixture")
    register_docs_readiness_tools(mcp, context=context)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "rm_docs_readiness" in names
    assert observed[0][0] == "preview"
