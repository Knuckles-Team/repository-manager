"""Characterization tests for repository_manager.cli_commands.parser.run.

``run`` (CCN 82) is a CLI dispatcher: argparse construction followed by a long
chain of ``if args.X:`` verb dispatches with a few sequential dependencies
(``has_errors`` flows from the gate dispatch into --bump/--maintain/--push;
``--validate`` resets ``args.bump``/``args.push`` so those verbs do not also
fire). These tests pin exactly the branches this lane's pure extract-method
refactor touches: the priority order of the immediate-exit verbs, the
manifest gate's validation/early-return/exit-1 paths, the missing-workspace-
file exit path, the repositories filter, --commit's missing-message guard,
the --gate / --gate-retest dispatch (including failure counting), and the
--validate -> --bump/--maintain/--push has_errors + reset chain.

The existing ``tests/test_repository_manager_cli.py`` already covers
--maintain/--clone/--install through the packaged ``main()`` entrypoint and
is reused as-is; this file exercises the branches it does not reach, calling
``run(runtime)`` directly against a constructed ``CliRuntime`` so each test
can inject a mock git object without going through the real ``Git`` factory.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from repository_manager.cli_commands.context import CliRuntime
from repository_manager.cli_commands.parser import run


class _ManifestError(Exception):
    pass


def _make_runtime(git: MagicMock, *, tmp_workspace_yml: str) -> tuple[CliRuntime, MagicMock]:
    logger = MagicMock()
    git_factory = MagicMock(return_value=git)
    synchronize = MagicMock()
    runtime = CliRuntime(
        git_factory=git_factory,
        version="0.0.0-test",
        default_workspace="/nonexistent/workspace",
        default_workspace_yml=tmp_workspace_yml,
        default_threads=1,
        logger=logger,
        synchronize_workspace_manifest=synchronize,
        manifest_error=_ManifestError,
    )
    return runtime, git_factory


@pytest.fixture
def git(tmp_path):
    g = MagicMock()
    g.project_map = {}
    g.path = str(tmp_path)
    g.generate_markdown_summary.return_value = "summary"
    return g


@pytest.fixture
def runtime_and_factory(git, tmp_path):
    # ``-f/--file`` defaults to ``runtime.default_workspace_yml``, so this
    # must be an EXISTING path or every test would silently hit the
    # missing-workspace-file exit(2) path via the default, not the flag.
    default_yml = tmp_path / "default-workspace.yml"
    default_yml.write_text("projects: {}\n")
    return _make_runtime(git, tmp_workspace_yml=str(default_yml))


def _argv(*args: str):
    return patch.object(sys, "argv", ["repository-manager", *args])


# --------------------------------------------------------------------------
# Immediate-exit verb priority (merge-queue / lane / etc. before bulk verbs)
# --------------------------------------------------------------------------


def test_merge_queue_dispatches_before_any_git_factory_call(runtime_and_factory):
    runtime, git_factory = runtime_and_factory
    with (
        patch(
            "repository_manager.cli_commands.parser.run_merge_queue_cli",
            return_value=3,
        ) as mock_mq,
        _argv("--merge-queue", "status"),
    ):
        with pytest.raises(SystemExit) as exc:
            run(runtime)
        assert exc.value.code == 3
        mock_mq.assert_called_once()
    git_factory.assert_not_called()


def test_lane_dispatches_before_any_git_factory_call(runtime_and_factory):
    runtime, git_factory = runtime_and_factory
    with (
        patch(
            "repository_manager.cli_commands.parser.run_lane_cli", return_value=0
        ) as mock_lane,
        _argv("--lane", "doctor"),
    ):
        with pytest.raises(SystemExit) as exc:
            run(runtime)
        assert exc.value.code == 0
        mock_lane.assert_called_once()
    git_factory.assert_not_called()


# --------------------------------------------------------------------------
# Manifest gate: validation errors, success path, exit-1 on unsynced check
# --------------------------------------------------------------------------


def test_manifest_check_without_source_errors_before_git_factory(runtime_and_factory):
    runtime, git_factory = runtime_and_factory
    with _argv("--manifest-check"), pytest.raises(SystemExit) as exc:
        run(runtime)
    assert exc.value.code == 2
    git_factory.assert_not_called()


def test_manifest_dry_run_without_sync_errors(runtime_and_factory):
    runtime, git_factory = runtime_and_factory
    with (
        _argv("--manifest-dry-run", "--manifest-source", "x"),
        pytest.raises(SystemExit) as exc,
    ):
        run(runtime)
    assert exc.value.code == 2
    git_factory.assert_not_called()


def test_manifest_check_success_returns_without_touching_git_factory(
    runtime_and_factory, capsys
):
    runtime, git_factory = runtime_and_factory
    report = MagicMock()
    report.as_dict.return_value = {"synchronized": True}
    report.synchronized = True
    runtime.synchronize_workspace_manifest.return_value = report
    with _argv("--manifest-check", "--manifest-source", "x"):
        run(runtime)  # must return normally, not sys.exit
    runtime.synchronize_workspace_manifest.assert_called_once()
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"synchronized": True}
    git_factory.assert_not_called()


def test_manifest_check_unsynchronized_exits_1(runtime_and_factory, capsys):
    runtime, git_factory = runtime_and_factory
    report = MagicMock()
    report.as_dict.return_value = {"synchronized": False}
    report.synchronized = False
    runtime.synchronize_workspace_manifest.return_value = report
    with (
        _argv("--manifest-check", "--manifest-source", "x"),
        pytest.raises(SystemExit) as exc,
    ):
        run(runtime)
    assert exc.value.code == 1
    git_factory.assert_not_called()


def test_manifest_sync_error_from_runtime_calls_parser_error(runtime_and_factory):
    runtime, git_factory = runtime_and_factory
    runtime.synchronize_workspace_manifest.side_effect = _ManifestError("boom")
    with (
        _argv("--manifest-sync", "--manifest-source", "x"),
        pytest.raises(SystemExit) as exc,
    ):
        run(runtime)
    assert exc.value.code == 2
    git_factory.assert_not_called()


# --------------------------------------------------------------------------
# Missing workspace file
# --------------------------------------------------------------------------


def test_missing_workspace_file_exits_2(runtime_and_factory, tmp_path):
    runtime, git_factory = runtime_and_factory
    missing = str(tmp_path / "does-not-exist.yml")
    with _argv("--file", missing), pytest.raises(SystemExit) as exc:
        run(runtime)
    assert exc.value.code == 2
    # git_factory IS called before the file-existence check in the original
    # ordering (git is constructed first, then the file is loaded into it).
    git_factory.assert_called_once()


# --------------------------------------------------------------------------
# Repositories filter
# --------------------------------------------------------------------------


def test_repositories_filter_narrows_existing_project_map(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    git.project_map = {
        "https://example.com/foo.git": "/w/foo",
        "https://example.com/bar.git": "/w/bar",
        "https://example.com/baz.git": "/w/baz",
    }
    with _argv("--repositories", "foo, bar"):
        run(runtime)
    assert set(git.project_map) == {
        "https://example.com/foo.git",
        "https://example.com/bar.git",
    }


def test_repositories_filter_builds_map_when_empty(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    git.project_map = {}
    with _argv("--repositories", "some-org/some-repo, bare-name"):
        run(runtime)
    assert "some-org/some-repo" in git.project_map
    assert any(k.endswith("bare-name") for k in git.project_map if "github.com" in k)


# --------------------------------------------------------------------------
# --commit missing message
# --------------------------------------------------------------------------


def test_commit_without_message_exits_1(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    with _argv("--commit"), pytest.raises(SystemExit) as exc:
        run(runtime)
    assert exc.value.code == 1
    git.commit_projects.assert_not_called()


def test_commit_with_message_calls_commit_projects(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    with _argv("--commit", "-m", "msg"):
        run(runtime)
    git.commit_projects.assert_called_once_with(message="msg")


# --------------------------------------------------------------------------
# --gate / --gate-retest dispatch
# --------------------------------------------------------------------------


def test_gate_clean_status_warns_and_does_not_set_has_errors(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    with (
        patch(
            "repository_manager.gate_runner.dispatch",
            return_value={"status": "clean"},
        ) as mock_dispatch,
        _argv("--gate", "fast"),
    ):
        run(runtime)
    mock_dispatch.assert_called_once()
    assert any(
        "No projects found" in str(call.args[0])
        for call in runtime.logger.warning.call_args_list
    )


def test_gate_failure_count_marks_has_errors_and_skips_push(runtime_and_factory, git):
    runtime, _ = runtime_and_factory

    def fake_dispatch(action, **kwargs):
        submit_one = kwargs["submit_one"]
        job_id = submit_one("repo1", "/w/repo1")
        return {"status": "ok", "queued_count": 1, "jobs": {"repo1": job_id}}

    with (
        patch(
            "repository_manager.gate_runner.dispatch", side_effect=fake_dispatch
        ),
        patch(
            "repository_manager.gate_runner.LocalJobStore"
        ) as MockStore,
        _argv("--gate", "fast", "--push"),
    ):
        store = MockStore.return_value
        store.submit_job.return_value = "job-1"
        store.jobs_lock.__enter__ = MagicMock()
        store.jobs_lock.__exit__ = MagicMock(return_value=False)
        store.jobs = {"job-1": {"error": "boom"}}
        run(runtime)
    assert any(
        "gate failed" in str(call.args[0])
        for call in runtime.logger.error.call_args_list
    )
    assert any(
        "Skipping push" in str(call.args[0])
        for call in runtime.logger.error.call_args_list
    )


def test_gate_retest_reports_nothing_to_retest_for_missing_job_id(
    runtime_and_factory, git
):
    runtime, _ = runtime_and_factory
    retest_result = {
        "message": "done",
        "targets": {"repo1": {"baseline": "PASS", "retest_job_id": None}},
    }
    with (
        patch("repository_manager.gate_runner.dispatch", return_value=retest_result),
        patch("repository_manager.gate_ledger.default_gate_ledger"),
        _argv("--gate-retest", "heavy"),
    ):
        run(runtime)
    assert any(
        "nothing to retest" in str(call.args[0])
        for call in runtime.logger.info.call_args_list
    )


# --------------------------------------------------------------------------
# --validate resets --bump/--push and has_errors gates them
# --------------------------------------------------------------------------


def test_validate_failure_prevents_bump_and_push(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    git.validate_and_release.return_value = {"passed": False}
    with _argv("--validate", "--bump", "patch", "--push"):
        run(runtime)
    runtime.logger.error.assert_any_call("Validation failed with errors.")
    # --validate resets args.bump/args.push to None/False, so bump_version
    # and phased_push must never be called even though has_errors alone
    # would also have blocked them.
    git.bump_version.assert_not_called()
    git.phased_push.assert_not_called()


def test_validate_success_still_clears_bump_and_push_flags(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    git.validate_and_release.return_value = {"passed": True}
    with _argv("--validate", "--bump", "patch", "--push"):
        run(runtime)
    runtime.logger.info.assert_any_call(
        "Validation and subsequent operations completed successfully."
    )
    git.bump_version.assert_not_called()
    git.phased_push.assert_not_called()


def test_bump_runs_per_project_and_tracks_errors(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    git.project_map = {"u1": "/w/one", "u2": "/w/two"}
    ok = MagicMock(status="success")
    err = MagicMock(status="error")
    git.bump_version.side_effect = [ok, err]
    with _argv("--bump", "patch"):
        run(runtime)
    assert git.bump_version.call_count == 2


def test_maintain_config_load_failure_exits_1(runtime_and_factory, tmp_path):
    runtime, _ = runtime_and_factory
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{not json")
    with (
        _argv("--maintain", "--config", str(bad_config)),
        pytest.raises(SystemExit) as exc,
    ):
        run(runtime)
    assert exc.value.code == 1


def test_push_config_load_failure_exits_1(runtime_and_factory, tmp_path):
    runtime, _ = runtime_and_factory
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{not json")
    with (
        _argv("--push", "--config", str(bad_config)),
        pytest.raises(SystemExit) as exc,
    ):
        run(runtime)
    assert exc.value.code == 1


def test_push_skipped_when_has_errors_from_failed_maintain(runtime_and_factory, git):
    runtime, _ = runtime_and_factory
    err = MagicMock(status="error")
    git.phased_bumpversion.return_value = [err]
    with _argv("--maintain", "--push"):
        run(runtime)
    git.phased_push.assert_not_called()
    runtime.logger.error.assert_any_call(
        "Skipping push due to preceding validation or bump errors."
    )
