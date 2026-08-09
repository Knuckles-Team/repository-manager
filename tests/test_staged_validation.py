"""RMDD-11 staged policy, evidence, and runner adversarial coverage."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
    ValidationStage,
)
from repository_manager.development.serialization import canonical_digest
from repository_manager.execution.fakes import FakeExecutor
from repository_manager.validation import (
    BaselineCache,
    BaselineMode,
    BaselineObservation,
    EvidenceError,
    EvidenceOutcome,
    FakeValidationJobAuthority,
    GateMode,
    LocalTestAdmission,
    PathSelection,
    TimeoutPolicy,
    ValidationCertificate,
    ValidationFailureClass,
    ValidationGate,
    ValidationPolicyError,
    ValidationProfile,
    ValidationProfileRegistry,
    ValidationRequest,
    ValidationRunner,
    ValidationRunResult,
    builtin_profiles,
    compare_failure_signals,
    profile_from_merge_config,
    verify_certificate,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (repo / "removed.txt").write_text("remove me\n", encoding="utf-8")
    _git(repo, "add", "-A")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=RMDD test",
            "-c",
            "user.email=rmdd@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )
    return repo, _git(repo, "rev-parse", "HEAD")


def _digest(value: str) -> str:
    return canonical_digest(value)


def _gate(
    name: str,
    stage: ValidationStage,
    *,
    baseline: BaselineMode = BaselineMode.DISABLED,
    include: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    mode: GateMode = GateMode.BLOCKING,
) -> ValidationGate:
    return ValidationGate(
        name=name,
        command=("true",),
        stage=stage,
        baseline_mode=baseline,
        path_selection=PathSelection(include=include),
        artifact_dependencies=dependencies,
        mode=mode,
    )


def _request(
    repo: Path,
    sha: str,
    profile: ValidationProfile,
    stages: tuple[ValidationStage, ...],
    *,
    base_sha: str | None = None,
    generation_id: str | None = None,
    changed_paths: tuple[str, ...] | None = None,
    snapshot_dirty_lane_tree: bool = True,
) -> ValidationRequest:
    return ValidationRequest(
        request_id="request:test",
        repository_id="repo:test",
        tree_sha=sha,
        tree_path=str(repo),
        profile=profile,
        stages=stages,
        config_digest=_digest("config"),
        toolchain_digest=_digest("toolchain"),
        target_host="host:test",
        generation_id=generation_id,
        base_sha=base_sha,
        changed_paths=changed_paths,
        snapshot_dirty_lane_tree=snapshot_dirty_lane_tree,
    )


def _result(
    *,
    outcome: ExecutionOutcome = ExecutionOutcome.SUCCEEDED,
    exit_code: int | None = 0,
    failure_class: FailureClass | None = None,
    stdout: str = "",
    stderr: str = "",
    command_id: str = "command:test",
    fence: str = "fence:test",
) -> ExecutionResult:
    now = datetime.now(UTC)
    return ExecutionResult(
        command_id=command_id,
        outcome=outcome,
        exit_code=exit_code,
        started_at=now,
        finished_at=now,
        duration_ms=0,
        worker_id="worker:test",
        fence=fence,
        failure_class=failure_class,
        stdout_tail=stdout,
        stderr_tail=stderr,
    )


def test_profile_selection_and_override_are_validated() -> None:
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate("python", ValidationStage.FEEDBACK, include=("src/**",)),
            _gate("docs", ValidationStage.INTEGRATION, include=("docs/**",)),
        ),
    )
    assert [gate.name for gate in profile.gates_for(("src/a.py",))] == ["python"]
    assert [gate.name for gate in profile.gates_for(("docs/a.md",))] == ["docs"]
    assert profile.gates_for(("other.txt",)) == ()
    assert profile.gates_for(()) == ()

    override = profile_from_merge_config(
        {
            "schema_version": 2,
            "gates": [
                {
                    "name": "lint",
                    "command": ["ruff", "check", "."],
                    "stage": "feedback",
                    "resources": {"cpu_weight": 2, "memory_mb": 512},
                }
            ],
        },
        family="repository",
    )
    assert override.gates[0].resources.cpu_weight == 2
    with pytest.raises(ValidationPolicyError):
        profile_from_merge_config(
            {"schema_version": 2, "unsafe": True, "gates": []},
            family="repository",
        )
    assert "python" in ValidationProfileRegistry().resolve("python").family


def test_plan_seals_exact_sha_and_dependency_linked_jobs(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate("feedback", ValidationStage.FEEDBACK),
            _gate(
                "certify",
                ValidationStage.CERTIFICATION,
                dependencies=("feedback",),
            ),
        ),
    )
    request = _request(
        repo,
        sha,
        profile,
        (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
        base_sha=sha,
        generation_id="generation:test",
    )
    plan = ValidationRunner().plan(request)
    assert all(job.tree_sha == sha for job in plan.jobs)
    assert plan.jobs[1].dependencies == (plan.jobs[0].job_id,)
    assert plan.plan_digest == ValidationRunner().plan(request).plan_digest
    assert all(job.worktree_path == str(repo) for job in plan.jobs)


def test_dirty_feedback_tree_uses_safe_commit_and_evaluates_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AdmittedSubprocessExecutor:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            argv = tuple(command.argv)
            self.commands.append(argv)
            started = datetime.now(UTC)
            completed = subprocess.run(
                list(argv),
                cwd=command.workdir,
                capture_output=True,
                text=True,
                check=False,
                timeout=command.timeout_seconds,
            )
            finished = datetime.now(UTC)
            return ExecutionResult(
                command_id=str(kwargs["command_id"]),
                outcome=(
                    ExecutionOutcome.SUCCEEDED
                    if completed.returncode == 0
                    else ExecutionOutcome.FAILED
                ),
                exit_code=completed.returncode,
                started_at=started,
                finished_at=finished,
                duration_ms=max(0, int((finished - started).total_seconds() * 1000)),
                worker_id=str(kwargs["worker_id"]),
                fence=str(kwargs["fence"]),
                failure_class=(
                    None
                    if completed.returncode == 0
                    else FailureClass.VALIDATION_CANDIDATE_FAILURE
                ),
                stdout_tail=completed.stdout[-64_000:],
                stderr_tail=completed.stderr[-64_000:],
            )

    repo, sha = _repo(tmp_path)
    (repo / "removed.txt").unlink()
    marker = tmp_path / "staged-status.txt"
    hook = tmp_path / "record_staged.py"
    hook.write_text(
        "import os, subprocess\n"
        "from pathlib import Path\n"
        "status = subprocess.check_output(\n"
        "    ['git', 'diff', '--name-status', 'HEAD^', 'HEAD'], text=True\n"
        ")\n"
        "Path(os.environ['RMDD_STAGED_MARKER']).write_text(status, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n"
        "- repo: local\n"
        "  hooks:\n"
        "  - id: record-staged\n"
        "    name: record staged\n"
        f"    entry: python {hook}\n"
        "    language: system\n"
        "    pass_filenames: false\n"
        "    always_run: true\n",
        encoding="utf-8",
    )
    unscheduled_marker = tmp_path / "unscheduled-hook.txt"
    installed_hook = repo / ".git" / "hooks" / "pre-commit"
    installed_hook.write_text(
        f"#!/bin/sh\nprintf 'unscheduled\\n' > '{unscheduled_marker}'\nexit 91\n",
        encoding="utf-8",
    )
    installed_hook.chmod(0o755)
    profile = ValidationProfile(
        "custom",
        1,
        (
            ValidationGate(
                name="feedback",
                command=("pre-commit", "run", "--all-files"),
                stage=ValidationStage.FEEDBACK,
                baseline_mode=BaselineMode.DISABLED,
            ),
        ),
    )
    calls: list[Path] = []
    safe_results: list[dict[str, Any]] = []

    def safe_commit_spy(path: Path, message: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(path)
        from repository_manager.safe_commit import safe_commit

        result = safe_commit(path, message, **kwargs)
        safe_results.append(result)
        return result

    monkeypatch.setenv("GIT_AUTHOR_NAME", "RMDD test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "rmdd@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "RMDD test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "rmdd@example.invalid")
    monkeypatch.setenv("RMDD_STAGED_MARKER", str(marker))
    authority = FakeValidationJobAuthority()
    admission = LocalTestAdmission()
    executor = AdmittedSubprocessExecutor()
    runner = ValidationRunner(
        job_authority=authority,
        resource_admission=admission,
        executor=executor,
        safe_commit_fn=safe_commit_spy,
    )
    result = runner.run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    assert result.preparation_error is None
    assert result.ok
    assert result.snapshot_gate_deferred is True
    assert result.evidence[0].snapshot_gate_deferred is True
    assert calls == [repo]
    assert result.request.tree_sha != sha
    assert _git(repo, "status", "--porcelain") == ""
    assert safe_results[0]["gate_invoked"] is False
    assert safe_results[0]["gate_deferred"] is True
    assert safe_results[0]["gate_stage"] == "deferred"
    assert authority.jobs[0].gate_name == "feedback"
    assert executor.commands[0] == ("pre-commit", "run", "--all-files")
    assert not unscheduled_marker.exists()
    staged_status = marker.read_text(encoding="utf-8")
    assert any(
        line.startswith("D\t") and line.endswith("removed.txt")
        for line in staged_status.splitlines()
    )
    with pytest.raises(subprocess.CalledProcessError):
        _git(repo, "show", f"{result.request.tree_sha}:removed.txt")


def test_changed_path_selection_is_derived_not_trusted_from_request(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "new.py").write_text("print('new')\n", encoding="utf-8")
    profile = ValidationProfile(
        "paths",
        1,
        (_gate("src-check", ValidationStage.FEEDBACK, include=("src/**",)),),
    )
    request = _request(
        repo,
        sha,
        profile,
        (ValidationStage.FEEDBACK,),
        base_sha=sha,
        changed_paths=("README.md",),
    )
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(request)
    assert result.preparation_error is None
    assert result.request.changed_paths == ("src/new.py",)
    assert [item.gate_name for item in result.evidence] == ["src-check"]


def test_deferred_snapshot_can_run_only_selected_lightweight_gate(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    (repo / "tracked.txt").write_text("lightweight\n", encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    profile = ValidationProfile(
        "lightweight",
        1,
        (_gate("quick", ValidationStage.FEEDBACK),),
    )
    executor = FakeExecutor()
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=executor,
    ).run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    assert result.ok
    assert result.snapshot_gate_deferred is True
    assert result.evidence[0].snapshot_gate_deferred is True
    assert executor.commands[0].argv == ("true",)


def test_certification_requires_a_passed_selected_precommit_replay(
    tmp_path: Path,
) -> None:
    def configured_repo(root: Path) -> tuple[Path, str]:
        root.mkdir()
        repo, sha = _repo(root)
        (repo / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
        (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        return repo, sha

    def profile(cert_command: tuple[str, ...]) -> ValidationProfile:
        return ValidationProfile(
            "certification",
            1,
            (
                ValidationGate(
                    name="feedback",
                    command=("true",),
                    stage=ValidationStage.FEEDBACK,
                    baseline_mode=BaselineMode.DISABLED,
                ),
                ValidationGate(
                    name="certify",
                    command=cert_command,
                    stage=ValidationStage.CERTIFICATION,
                    baseline_mode=BaselineMode.ABSOLUTE,
                    artifact_dependencies=("feedback",),
                ),
            ),
        )

    no_replay_repo, no_replay_sha = configured_repo(tmp_path / "no-replay")
    no_replay = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            no_replay_repo,
            no_replay_sha,
            profile(("true",)),
            (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
            base_sha=no_replay_sha,
            generation_id="generation:no-replay",
        )
    )
    assert no_replay.ok
    assert no_replay.snapshot_gate_deferred is True
    assert no_replay.certificate is None
    assert no_replay.evidence[-1].snapshot_gate_replayed is False

    replay_repo, replay_sha = configured_repo(tmp_path / "replay")
    replay = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            replay_repo,
            replay_sha,
            profile(("pre-commit", "run", "--all-files")),
            (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
            base_sha=replay_sha,
            generation_id="generation:replay",
        )
    )
    assert replay.certificate is not None
    assert replay.evidence[-1].snapshot_gate_replayed is True
    assert replay.landable

    spoof_repo, spoof_sha = configured_repo(tmp_path / "spoof")
    spoof = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            spoof_repo,
            spoof_sha,
            profile(("echo", "pre-commit", "run", "--all-files")),
            (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
            base_sha=spoof_sha,
            generation_id="generation:spoof",
        )
    )
    assert spoof.ok
    assert spoof.certificate is None
    assert spoof.evidence[-1].snapshot_gate_replayed is False


def test_precommit_replay_proof_accepts_only_the_exact_full_repo_argv() -> None:
    def runs(command: tuple[str, ...]) -> bool:
        return ValidationGate(
            name="certify",
            command=command,
            stage=ValidationStage.CERTIFICATION,
            baseline_mode=BaselineMode.ABSOLUTE,
        ).runs_precommit

    assert runs(("pre-commit", "run", "--all-files"))
    assert runs(("python3.14", "-m", "pre_commit", "run", "--all-files"))
    assert not runs(("pre-commit", "run", "hook-id", "--all-files"))
    assert not runs(("pre-commit", "run", "--files", "tracked.py", "--all-files"))
    assert not runs(("echo", "pre-commit", "run", "--all-files"))
    assert not runs(("pre-commit-wrapper", "run", "--all-files"))


@pytest.mark.parametrize(
    ("family", "changed_file"),
    [("rust", "lib.rs"), ("frontend", "app.ts")],
)
def test_builtin_build_certification_requires_hook_replay_when_snapshot_deferred(
    tmp_path: Path,
    family: str,
    changed_file: str,
) -> None:
    repo, sha = _repo(tmp_path)
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    (repo / changed_file).write_text("changed\n", encoding="utf-8")
    builtin = builtin_profiles()[family]
    profile = replace(
        builtin,
        gates=tuple(
            replace(gate, baseline_mode=BaselineMode.ABSOLUTE) for gate in builtin.gates
        ),
    )
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
            base_sha=sha,
            generation_id=f"generation:{family}",
        )
    )
    assert result.ok
    assert result.snapshot_gate_deferred is True
    certification = [
        item for item in result.evidence if item.stage is ValidationStage.CERTIFICATION
    ]
    assert certification
    assert all(item.snapshot_gate_replayed is False for item in certification)
    assert result.certificate is None


def test_changed_path_derivation_preserves_deletion_rename_and_odd_names(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    odd_name = "odd\tname.py"
    (repo / odd_name).write_text("odd\n", encoding="utf-8")
    (repo / "removed.txt").unlink()
    profile = ValidationProfile(
        "paths",
        1,
        (_gate("deleted", ValidationStage.FEEDBACK, include=("removed.txt",)),),
    )
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.FEEDBACK,),
            base_sha=sha,
            changed_paths=("unrelated.txt",),
        )
    )
    assert result.preparation_error is None
    assert result.request.changed_paths == (odd_name, "removed.txt")
    assert [item.gate_name for item in result.evidence] == ["deleted"]

    rename_root = tmp_path / "rename"
    rename_root.mkdir()
    rename_repo, rename_sha = _repo(rename_root)
    (rename_repo / "removed.txt").rename(rename_repo / odd_name)
    rename_profile = ValidationProfile(
        "rename",
        1,
        (_gate("old-name", ValidationStage.FEEDBACK, include=("removed.txt",)),),
    )
    renamed = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            rename_repo,
            rename_sha,
            rename_profile,
            (ValidationStage.FEEDBACK,),
            base_sha=rename_sha,
            changed_paths=("unrelated.txt",),
        )
    )
    assert renamed.preparation_error is None
    assert renamed.request.changed_paths == (odd_name, "removed.txt")
    assert [item.gate_name for item in renamed.evidence] == ["old-name"]


def test_git_inspection_terminates_on_bounded_status_and_diff_output(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    for index in range(40):
        (repo / f"untracked-{index:03d}.txt").write_text("x\n", encoding="utf-8")
    status = ValidationRunner._bounded_git(
        ["status", "--porcelain", "-z"], repo, max_bytes=32
    )
    assert status is None

    for index in range(40):
        (repo / f"committed-{index:03d}.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=RMDD test",
            "-c",
            "user.email=rmdd@example.invalid",
            "commit",
            "-qm",
            "many files",
        ],
        cwd=repo,
        check=True,
    )
    new_sha = _git(repo, "rev-parse", "HEAD")
    diff = ValidationRunner._bounded_git(
        ["diff", "--name-status", "-z", sha, new_sha, "--"],
        repo,
        max_bytes=32,
    )
    assert diff is None


def test_differential_baseline_is_fail_closed_and_cache_identity_is_complete(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate(
                "certify",
                ValidationStage.CERTIFICATION,
                baseline=BaselineMode.DIFFERENTIAL,
            ),
        ),
    )
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.CERTIFICATION,),
            base_sha=sha,
            generation_id="generation:test",
        )
    )
    assert not result.ok
    assert result.certificate is None
    assert result.evidence[0].outcome is EvidenceOutcome.REFUSED
    assert (
        result.evidence[0].failure_class is ValidationFailureClass.BASELINE_UNPRODUCIBLE
    )

    cache = BaselineCache()
    identity = {
        "base_sha": sha,
        "gate_config_digest": _digest("gate"),
        "command_digest": _digest("command"),
        "toolchain_digest": _digest("toolchain"),
        "target_host": "host:test",
    }
    observation = BaselineObservation(readable=True, tree_sha=sha, exit_code=0)
    cache.put(observation, **identity)
    assert cache.get(**identity) == observation
    changed = dict(identity, toolchain_digest=_digest("new-toolchain"))
    assert cache.get(**changed) is None


def test_differential_compare_allows_only_preexisting_signals() -> None:
    baseline = BaselineObservation(
        readable=True,
        tree_sha="a" * 40,
        exit_code=1,
        failure_ids=("old",),
    )
    allowed = compare_failure_signals(
        mode=BaselineMode.DIFFERENTIAL,
        baseline=baseline,
        candidate_exit_code=1,
        candidate_failure_ids=("old",),
    )
    denied = compare_failure_signals(
        mode=BaselineMode.DIFFERENTIAL,
        baseline=baseline,
        candidate_exit_code=1,
        candidate_failure_ids=("old", "new"),
    )
    unreadable = compare_failure_signals(
        mode=BaselineMode.DIFFERENTIAL,
        baseline=BaselineObservation(readable=False, tree_sha="a" * 40),
        candidate_exit_code=0,
    )
    assert allowed.ok and allowed.pre_existing_failure_ids == ("old",)
    assert not denied.ok and denied.new_failure_ids == ("new",)
    assert not unreadable.ok and not unreadable.baseline_readable
    unitemized = compare_failure_signals(
        mode=BaselineMode.DIFFERENTIAL,
        baseline=BaselineObservation(readable=True, tree_sha="a" * 40, exit_code=1),
        candidate_exit_code=1,
    )
    assert not unitemized.ok
    assert unitemized.new_failure_ids == ("<unitemized-failure>",)
    with pytest.raises(EvidenceError, match="failure IDs exceed the bounded count"):
        BaselineObservation(
            readable=True,
            tree_sha="a" * 40,
            exit_code=1,
            failure_ids=tuple(str(index) for index in range(257)),
        )
    with pytest.raises(EvidenceError, match="failure IDs exceed the bounded count"):
        compare_failure_signals(
            mode=BaselineMode.DIFFERENTIAL,
            baseline=baseline,
            candidate_exit_code=1,
            candidate_failure_ids=tuple(str(index) for index in range(257)),
        )
    with pytest.raises(EvidenceError, match="failure ID size"):
        BaselineObservation(
            readable=True,
            tree_sha="a" * 40,
            exit_code=1,
            failure_ids=("x" * 1025,),
        )


def test_resource_refusal_happens_before_executor(tmp_path: Path) -> None:
    class CountingExecutor(FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            self.calls += 1
            raise AssertionError("executor must not run after resource refusal")

    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom", 1, (_gate("feedback", ValidationStage.FEEDBACK),)
    )
    admission = LocalTestAdmission(allow=False)
    executor = CountingExecutor()
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=admission,
        executor=executor,
    ).run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    assert result.evidence[0].outcome is EvidenceOutcome.DEFERRED
    assert result.evidence[0].failure_class is ValidationFailureClass.RESOURCE
    assert executor.calls == 0


def test_cancelled_and_timed_out_jobs_release_reservations(tmp_path: Path) -> None:
    class OutcomeExecutor(FakeExecutor):
        def __init__(self, outcome: ExecutionOutcome) -> None:
            super().__init__()
            self.outcome = outcome

        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            return _result(
                outcome=self.outcome,
                exit_code=None,
                command_id=str(kwargs["command_id"]),
                fence=str(kwargs["fence"]),
            )

    repo, sha = _repo(tmp_path)
    cancelled_profile = ValidationProfile(
        "cancelled", 1, (_gate("cancel", ValidationStage.FEEDBACK),)
    )
    cancelled_admission = LocalTestAdmission()
    cancelled = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=cancelled_admission,
        executor=OutcomeExecutor(ExecutionOutcome.CANCELLED),
    ).run(_request(repo, sha, cancelled_profile, (ValidationStage.FEEDBACK,)))
    assert cancelled.evidence[0].outcome is EvidenceOutcome.CANCELLED
    assert len(cancelled_admission.released) == 1

    timeout_gate = ValidationGate(
        name="timeout",
        command=("true",),
        stage=ValidationStage.CERTIFICATION,
        baseline_mode=BaselineMode.DISABLED,
        timeout_policy=TimeoutPolicy.DEFER,
    )
    timeout_profile = ValidationProfile("timeout", 1, (timeout_gate,))
    timeout_admission = LocalTestAdmission()
    timed_out = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=timeout_admission,
        executor=OutcomeExecutor(ExecutionOutcome.TIMED_OUT),
    ).run(
        _request(
            repo,
            sha,
            timeout_profile,
            (ValidationStage.CERTIFICATION,),
            generation_id="generation:timeout",
        )
    )
    assert not timed_out.ok
    assert timed_out.certificate is None
    assert timed_out.evidence[0].outcome is EvidenceOutcome.DEFERRED
    assert timed_out.evidence[0].failure_class is ValidationFailureClass.TIMEOUT
    assert len(timeout_admission.released) == 1
    assert timeout_admission.released[0][1] is EvidenceOutcome.DEFERRED


def test_executor_environment_failure_still_releases_resource(tmp_path: Path) -> None:
    class RaisingExecutor:
        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            raise OSError("worker disappeared")

    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "environment", 1, (_gate("check", ValidationStage.FEEDBACK),)
    )
    admission = LocalTestAdmission()
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=admission,
        executor=RaisingExecutor(),
    ).run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    assert result.evidence[0].outcome is EvidenceOutcome.FAILED
    assert result.evidence[0].failure_class is ValidationFailureClass.ENVIRONMENT
    assert len(admission.released) == 1
    assert admission.released[0][1] is EvidenceOutcome.FAILED


def test_blocking_failure_prevents_dependent_gate_execution_in_order(
    tmp_path: Path,
) -> None:
    class OrderedExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("dependent gate must remain blocked")
            return _result(
                outcome=ExecutionOutcome.FAILED,
                exit_code=1,
                failure_class=FailureClass.VALIDATION_CANDIDATE_FAILURE,
                command_id=str(kwargs["command_id"]),
                fence=str(kwargs["fence"]),
            )

    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "ordering",
        1,
        (
            _gate("feedback", ValidationStage.FEEDBACK),
            _gate(
                "certify",
                ValidationStage.CERTIFICATION,
                baseline=BaselineMode.ABSOLUTE,
                dependencies=("feedback",),
            ),
        ),
    )
    executor = OrderedExecutor()
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=executor,
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.FEEDBACK, ValidationStage.CERTIFICATION),
            generation_id="generation:ordering",
        )
    )
    assert executor.calls == 1
    assert [item.outcome for item in result.evidence] == [
        EvidenceOutcome.FAILED,
        EvidenceOutcome.SKIPPED,
    ]
    assert result.evidence[1].failure_class is ValidationFailureClass.DEPENDENCY
    assert result.certificate is None


def test_symlink_worktree_and_result_fence_mismatch_are_refused(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(repo, target_is_directory=True)
    profile = ValidationProfile(
        "custom", 1, (_gate("check", ValidationStage.FEEDBACK),)
    )
    linked = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(_request(link, sha, profile, (ValidationStage.FEEDBACK,)))
    assert linked.preparation_error is not None

    class WrongFenceExecutor:
        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            return _result()

    mismatched = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=WrongFenceExecutor(),
    ).run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    assert mismatched.evidence[0].outcome is EvidenceOutcome.REFUSED
    assert mismatched.evidence[0].failure_class is ValidationFailureClass.STALE_FENCE


def test_reservation_on_different_host_is_refused_before_executor(
    tmp_path: Path,
) -> None:
    class CountingExecutor(FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
            self.calls += 1
            return super().run(command, **kwargs)

    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "remote", 1, (_gate("check", ValidationStage.FEEDBACK),)
    )
    admission = LocalTestAdmission(host_id="host:elsewhere")
    executor = CountingExecutor()
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=admission,
        executor=executor,
    ).run(_request(repo, sha, profile, (ValidationStage.FEEDBACK,)))
    evidence = result.evidence[0]
    assert evidence.outcome is EvidenceOutcome.REFUSED
    assert evidence.failure_class is ValidationFailureClass.STALE_FENCE
    assert executor.calls == 0
    assert admission.released[0][1] is EvidenceOutcome.REFUSED


def test_local_and_inventory_results_share_the_same_aggregate_schema(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "portable",
        1,
        (
            _gate(
                "certify", ValidationStage.CERTIFICATION, baseline=BaselineMode.ABSOLUTE
            ),
        ),
    )
    base_request = _request(
        repo,
        sha,
        profile,
        (ValidationStage.CERTIFICATION,),
        generation_id="generation:portable",
    )
    local = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(host_id="host:test"),
        executor=FakeExecutor(),
    ).run(base_request)
    remote_request = replace(base_request, target_host="inventory:build-a")
    remote = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(host_id="inventory:build-a"),
        executor=FakeExecutor(),
    ).run(remote_request)

    def aggregate(result: ValidationRunResult) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            sorted(
                (
                    item.gate_name,
                    item.stage.value,
                    item.outcome.value,
                    item.failure_class.value if item.failure_class else "",
                )
                for item in result.evidence
            )
        )

    assert local.ok and local.landable
    assert remote.ok and remote.landable
    assert (
        aggregate(local)
        == aggregate(remote)
        == (("certify", "certification", "passed", ""),)
    )
    assert local.evidence[0].target_host == "host:test"
    assert remote.evidence[0].target_host == "inventory:build-a"
    assert local.certificate is not None
    assert remote.certificate is not None
    assert verify_certificate(local.certificate, local.evidence).valid
    assert verify_certificate(remote.certificate, remote.evidence).valid


def test_baseline_provider_tree_mismatch_refuses_even_for_green_command(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate(
                "certify",
                ValidationStage.CERTIFICATION,
                baseline=BaselineMode.DIFFERENTIAL,
            ),
        ),
    )

    class WrongBaseline:
        def run(self, job: object, command: ExecutionCommand) -> BaselineObservation:
            return BaselineObservation(readable=True, tree_sha="b" * 40, exit_code=0)

    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
        baseline_provider=WrongBaseline(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.CERTIFICATION,),
            base_sha=sha,
            generation_id="generation:test",
        )
    )
    assert result.evidence[0].outcome is EvidenceOutcome.REFUSED
    assert (
        result.evidence[0].failure_class is ValidationFailureClass.BASELINE_UNPRODUCIBLE
    )


def test_handoff_is_typed_and_has_no_landing_or_push_effect(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile("custom", 1, (_gate("smoke", ValidationStage.SMOKE),))
    request = _request(
        repo, sha, profile, (ValidationStage.SMOKE,), generation_id="generation:test"
    )
    handoff = ValidationRunner.post_land_smoke_handoff(
        request, tree_sha=sha, evidence=()
    )
    assert handoff.from_stage is ValidationStage.SMOKE
    assert handoff.next_stage is ValidationStage.RELEASE
    assert handoff.tree_sha == sha


def test_certificate_changes_when_evidence_changes(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate(
                "certify", ValidationStage.CERTIFICATION, baseline=BaselineMode.ABSOLUTE
            ),
        ),
    )
    first = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.CERTIFICATION,),
            generation_id="generation:test",
        )
    )
    assert first.certificate is not None
    changed = replace(first.evidence[0], detail="different evidence")
    second = ValidationCertificate.issue(
        certificate_id="certificate:changed",
        generation_id="generation:test",
        tree_sha=sha,
        gate_config_digest=first.request.config_digest,
        toolchain_digest=first.request.toolchain_digest,
        target_host=first.request.target_host,
        resource_digest=first.request.resource_digest,
        blocking_gate_names=("certify",),
        evidence=(changed,),
        issued_at=datetime.now(UTC),
        profile_digest=first.request.profile.digest,
    )
    assert second.digest != first.certificate.digest


def test_profile_dependency_must_precede_dependent_gate() -> None:
    with pytest.raises(Exception, match="earlier stage"):
        ValidationProfile(
            "bad",
            1,
            (
                _gate(
                    "feedback",
                    ValidationStage.FEEDBACK,
                    dependencies=("certify",),
                ),
                _gate("certify", ValidationStage.CERTIFICATION),
            ),
        )


def test_certification_certificate_rejects_stage_and_identity_drift(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    profile = ValidationProfile(
        "custom",
        1,
        (
            _gate(
                "certify", ValidationStage.CERTIFICATION, baseline=BaselineMode.ABSOLUTE
            ),
        ),
    )
    result = ValidationRunner(
        job_authority=FakeValidationJobAuthority(),
        resource_admission=LocalTestAdmission(),
        executor=FakeExecutor(),
    ).run(
        _request(
            repo,
            sha,
            profile,
            (ValidationStage.CERTIFICATION,),
            generation_id="generation:test",
        )
    )
    assert result.certificate is not None
    assert result.landable
    tampered = (replace(result.evidence[0], tree_sha="b" * 40),)
    check = verify_certificate(result.certificate, tampered)
    assert not check.valid
    assert any("tree SHA" in reason for reason in check.reasons)
    for field, reason_fragment in (
        ("gate_config_digest", "config digest"),
        ("toolchain_digest", "toolchain digest"),
        ("profile_digest", "profile digest"),
        ("command_digest", "evidence digest set"),
    ):
        drift_kwargs: Any = {field: _digest(f"drift:{field}")}
        drifted = (replace(result.evidence[0], **drift_kwargs),)
        drift_check = verify_certificate(result.certificate, drifted)
        assert not drift_check.valid
        assert any(reason_fragment in reason for reason in drift_check.reasons)
    missing = verify_certificate(result.certificate, ())
    assert not missing.valid
    assert any(
        "missing blocking certification evidence" in reason
        for reason in missing.reasons
    )
    with pytest.raises(EvidenceError, match="blocking gate set"):
        ValidationCertificate.issue(
            certificate_id="certificate:no-blocking",
            generation_id="generation:test",
            tree_sha=sha,
            gate_config_digest=result.request.config_digest,
            toolchain_digest=result.request.toolchain_digest,
            target_host=result.request.target_host,
            resource_digest=result.request.resource_digest,
            blocking_gate_names=(),
            evidence=(result.evidence[0],),
            issued_at=datetime.now(UTC),
            profile_digest=result.request.profile.digest,
        )
    with pytest.raises(EvidenceError, match="evidence set"):
        ValidationCertificate.issue(
            certificate_id="certificate:no-evidence",
            generation_id="generation:test",
            tree_sha=sha,
            gate_config_digest=result.request.config_digest,
            toolchain_digest=result.request.toolchain_digest,
            target_host=result.request.target_host,
            resource_digest=result.request.resource_digest,
            blocking_gate_names=("certify",),
            evidence=(),
            issued_at=datetime.now(UTC),
            profile_digest=result.request.profile.digest,
        )
    with pytest.raises(EvidenceError):
        ValidationCertificate.issue(
            certificate_id="certificate:bad",
            generation_id="generation:test",
            tree_sha=sha,
            gate_config_digest=result.request.config_digest,
            toolchain_digest=result.request.toolchain_digest,
            target_host=result.request.target_host,
            resource_digest=result.request.resource_digest,
            blocking_gate_names=("certify",),
            evidence=(replace(result.evidence[0], stage=ValidationStage.FEEDBACK),),
            issued_at=datetime.now(UTC),
            profile_digest=result.request.profile.digest,
        )
