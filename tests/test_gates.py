"""Tests for the single gate-execution engine (repository_manager.gates).

Covers what the old ``repository_manager.scanner`` prototype tested (parsing,
"no parseable hook results" surfacing) PLUS the thing that module could never
prove: that ``stage="heavy"`` actually reaches ``--hook-stage pre-push`` and
therefore fires a hook declared ONLY at that stage, while ``stage="fast"``
does not.
"""

import subprocess
from unittest import mock

from repository_manager import gates
from tests.conftest import isolated_git_subprocess_env


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[
            "pre-commit",
            "run",
            "--hook-stage",
            "pre-commit",
            "--all-files",
            "--verbose",
        ],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_init_failure_surfaces_raw_output(tmp_path):
    """A non-zero pre-commit with no parseable hooks must report the real error."""
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    boom = "An error has occurred: InvalidConfigError\nfailed to install hook env"

    with mock.patch.object(
        gates, "_run_pre_commit", return_value=_completed(1, stderr=boom)
    ):
        result = gates.run_gate_stage(str(tmp_path), "fast")

    assert result.success is False
    assert result.exit_code == 1
    assert result.hooks == []
    assert result.error is not None
    assert "InvalidConfigError" in result.error
    assert result.stage == "fast"


def test_real_hook_failure_not_overridden(tmp_path):
    """When a hook genuinely fails, error stays None and the hook is captured."""
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = (
        "ruff.....................................................................Failed\n"
        "- hook id: ruff\n"
        "- duration: 0.42s\n\n"
        "E501 line too long\n"
    )

    with mock.patch.object(
        gates, "_run_pre_commit", return_value=_completed(1, stdout=output)
    ):
        result = gates.run_gate_stage(str(tmp_path), "fast")

    assert result.success is False
    failed = [h for h in result.hooks if not h.passed]
    assert len(failed) == 1
    assert failed[0].hook_id == "ruff"
    assert failed[0].duration_s == 0.42
    assert result.error is None


def test_success_passes_cleanly(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = "ruff.....................................................................Passed\n- hook id: ruff\n- duration: 0.1s\n"

    with mock.patch.object(
        gates, "_run_pre_commit", return_value=_completed(0, stdout=output)
    ):
        result = gates.run_gate_stage(str(tmp_path), "fast")

    assert result.success is True
    assert result.error is None
    assert result.hooks[0].duration_s == 0.1


def test_no_config_is_a_clean_skip(tmp_path):
    result = gates.run_gate_stage(str(tmp_path), "fast")
    assert result.success is True
    assert result.hooks == []
    assert "No .pre-commit-config.yaml" in (result.error or "")


def test_unknown_stage_rejected(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    try:
        gates.run_gate_stage(str(tmp_path), "medium")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "fast" in str(e) and "heavy" in str(e)


def test_run_pre_commit_passes_the_hook_stage_flag(tmp_path):
    """The literal fix: --hook-stage must reach the subprocess argv."""
    with mock.patch("subprocess.run", return_value=_completed(0)) as run:
        gates._run_pre_commit(str(tmp_path), "pre-push", files=["a.py"])
    argv = run.call_args.args[0]
    assert "--hook-stage" in argv
    assert argv[argv.index("--hook-stage") + 1] == "pre-push"
    assert "--files" in argv and "a.py" in argv


def test_explain_gate_result_condenses_to_failures(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = (
        "ruff.....................................................................Failed\n"
        "- hook id: ruff\n- duration: 0.2s\n\nE501 line too long at foo.py:12\n"
    )
    with mock.patch.object(
        gates, "_run_pre_commit", return_value=_completed(1, stdout=output)
    ):
        result = gates.run_gate_stage(str(tmp_path), "fast")

    explanation = gates.explain_gate_result(result)
    assert "FAILED" in explanation
    assert "ruff" in explanation
    assert "E501" in explanation


def test_profile_gate_result_sorts_slowest_first(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = (
        "fast-hook................................................................Passed\n"
        "- hook id: fast-hook\n- duration: 0.1s\n\n"
        "slow-hook................................................................Passed\n"
        "- hook id: slow-hook\n- duration: 3.5s\n\n"
    )
    with mock.patch.object(
        gates, "_run_pre_commit", return_value=_completed(0, stdout=output)
    ):
        result = gates.run_gate_stage(str(tmp_path), "fast")

    profile = gates.profile_gate_result(result)
    assert [p["hook_id"] for p in profile] == ["slow-hook", "fast-hook"]
    assert profile[0]["duration_s"] == 3.5


# ---------------------------------------------------------------------------
# Live proof: a hook declared ONLY at pre-push fires at stage="heavy" and not
# at stage="fast". No mocking of pre-commit itself -- this is the actual CLI.
# ---------------------------------------------------------------------------


def _init_repo_with_tiered_hooks(tmp_path):
    env = isolated_git_subprocess_env()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.email", "a@b.c"], cwd=tmp_path, check=True, env=env
    )  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=tmp_path, check=True, env=env
    )  # nosec B603 B607
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "default_stages: [pre-commit]\n"
        "repos:\n"
        "- repo: local\n"
        "  hooks:\n"
        "  - id: fast-only\n"
        "    name: fast-only\n"
        "    entry: python3 -c \"print('FAST_ONLY_RAN')\"\n"
        "    language: system\n"
        "    always_run: true\n"
        "    pass_filenames: false\n"
        "  - id: heavy-only\n"
        "    name: heavy-only\n"
        "    entry: python3 -c \"print('HEAVY_ONLY_RAN')\"\n"
        "    language: system\n"
        "    always_run: true\n"
        "    pass_filenames: false\n"
        "    stages: [pre-push, manual]\n"
    )
    (tmp_path / "file.txt").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True, env=env
    )  # nosec B603 B607


def test_stage_fast_does_not_fire_the_pre_push_only_hook(tmp_path):
    _init_repo_with_tiered_hooks(tmp_path)
    result = gates.run_gate_stage(str(tmp_path), "fast")
    ran = {h.hook_id for h in result.hooks}
    assert "fast-only" in ran
    assert "heavy-only" not in ran
    assert "HEAVY_ONLY_RAN" not in result.raw_output


def test_stage_heavy_fires_the_pre_push_only_hook(tmp_path):
    """The actual proof requested: heavy reaches a hook fast never can."""
    _init_repo_with_tiered_hooks(tmp_path)
    result = gates.run_gate_stage(str(tmp_path), "heavy")
    ran = {h.hook_id for h in result.hooks}
    assert "heavy-only" in ran
    assert "HEAVY_ONLY_RAN" in result.raw_output
    # default_stages: [pre-commit] means the fast-tier hook does NOT re-run
    # at pre-push -- confirms stage scoping is exact, not additive/leaky.
    assert "fast-only" not in ran
