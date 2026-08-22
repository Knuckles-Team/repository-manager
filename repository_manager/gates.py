"""The single implementation of "run this repo's declared pre-commit hooks".

**The blocking gap this module closes (GOC-60/P0.4).** The two-tier gate model
(GOC-59) declares hooks FAST (``pre-commit`` stage: formatters/linters, no
network/tests/compilers) or HEAVY (``pre-push`` stage: pytest, cargo, ``uv lock
--check``, ...). Before this module existed, ``--hook-stage pre-push`` appeared
**nowhere** in this codebase: ``Git._gate_before_push`` — despite its name — ran
ordinary commit-stage hooks (pre-commit's own default when ``--hook-stage`` is
omitted), so a repo's declared HEAVY hooks were never exercised by anything.
The two-tier model was undrivable through the tool meant to drive it.

**Collapses two duplicate "run pre-commit, parse the result" implementations**
that used to disagree on this exact point:

* ``repository_manager.scanner`` (deleted) — a two-pass prototype used by
  ``Git.validate_single_project`` (→ ``rm_projects action=validate``) and,
  separately, by ``Git._gate_before_push``. Neither call ever passed
  ``--hook-stage``.
* This module, now the ONLY place that shells out to ``pre-commit run`` to
  *check* (not stage/commit) a repo's hooks. Both callers above, and the new
  ``rm_gates`` MCP tool, call :func:`run_gate_stage` here.

``Git.pre_commit``/``Git.pre_commit_projects`` (the interactive stage-and-commit
helper backing the CLI's ``--pre-commit`` flag and the commit-prep step of
``rm_projects validate``'s auto-bump/auto-push chain) is deliberately left as
its own implementation: it stages (``git add -A``), retries once if a
formatter reformatted files, and returns a ``GitResult`` through
``Git.git_action`` — a mutating, retrying workflow with a different contract
than "run the hooks and report what happened", not another copy of it. It now
passes ``--hook-stage pre-commit`` explicitly (previously relied on
pre-commit's own default), so the tier it runs is declared, not implied.

``merge_queue``'s gate runner is also left alone: its ``GateSpec``/
``QueueConfig`` model runs arbitrary declared commands and diffs their output
against a baseline run on the base ref (a *differential landing decision*, not
"does this repo's own hook suite pass right now"). Genuinely different
semantics — see ``merge_queue.py``'s module docstring.

**Two more things this module now does, both in service of the same fast
release loop (see :mod:`repository_manager.gate_ledger`'s module docstring
for the incident that forced them).** First, :func:`run_gate_stage` records
every run it makes into the durable :class:`repository_manager.gate_ledger
.GateLedger` (unless a caller opts out with ``record=False``) -- a job store
that dies with the process cannot answer "what failed last time?", and that
question is what turned one repository's push into six hours of re-running a
90-minute wave for a handful of steps that had actually failed. Second, a
HEAVY-tier run against a Cargo-based repo takes a ``"build"`` reservation
from :mod:`repository_manager.task_queue` before shelling out to
``pre-commit`` -- ``cargo``'s shared ``CARGO_TARGET_DIR`` corrupts concurrent
writers (phantom ``E0599``), the exact incident that class exists to
prevent, and two heavy gates racing on the same repo's Cargo build is exactly
that shape. A refused reservation is reported back as ``deferred=True``, not
as a failure: the gate never ran, so there is nothing to have found.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess  # nosec B404 - fixed argv only, never shell=True
import time
from pathlib import Path
from typing import Any, Literal

from repository_manager import build_queue, gate_ledger, merge_queue, task_queue
from repository_manager.scan_models import HookResult, RepoScanResult

logger = logging.getLogger(__name__)

GateStage = Literal["fast", "heavy"]

#: Maps the tool-facing tier name to pre-commit's own ``--hook-stage`` value.
#: This dict IS the two-tier contract at the execution layer -- the one place
#: "fast" and "heavy" become the pre-commit stages that make it real.
HOOK_STAGE_BY_GATE_STAGE: dict[str, str] = {
    "fast": "pre-commit",
    "heavy": "pre-push",
}

_HOOK_LINE = re.compile(r"^(.+?)\.{5,}(Passed|Failed|Skipped)\s*$")
_DURATION_LINE = re.compile(r"^-\s*duration:\s*([\d.]+)\s*s?\s*$", re.IGNORECASE)

# Per-tier wall-clock ceilings. The FAST tier is diff-scoped lint/format work and
# genuinely should finish in minutes. The HEAVY tier runs a repo's full pytest /
# cargo / lockfile suite, which for the larger repos (agent-utilities,
# epistemic-graph) legitimately exceeds ten minutes -- a 600s ceiling there is not
# a safety net, it is a guaranteed timeout that reports as a gate FAILURE and so
# is indistinguishable from the gate actually finding a defect.
_DEFAULT_TIMEOUT_BY_STAGE = {"fast": 600, "heavy": 5400}
_TIMEOUT_ENV_VAR = "RM_GATE_TIMEOUT_SECONDS"


def default_gate_timeout(stage: str) -> int:
    """Wall-clock ceiling for ``stage``, overridable via ``RM_GATE_TIMEOUT_SECONDS``.

    An explicit override applies to every tier; otherwise the per-tier default
    applies. A non-numeric or non-positive override is ignored rather than
    crashing a push.
    """
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw:
        try:
            override = int(raw)
        except ValueError:
            override = 0
        if override > 0:
            return override
    return _DEFAULT_TIMEOUT_BY_STAGE.get(stage, 600)


def _run_pre_commit(
    repo_path: str,
    hook_stage: str,
    *,
    files: list[str] | None = None,
    hook_ids: list[str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """The one ``pre-commit run --hook-stage <stage>`` invocation in this codebase.

    ``files`` (non-empty) scopes per-file hooks to just those paths via
    ``--files`` instead of ``--all-files`` -- much faster on large repos;
    ``always_run`` hooks still run regardless, so nothing is skipped, only
    narrowed. ``hook_ids`` (non-empty) additionally restricts the run to those
    specific hook ids (used by ``profile`` when a caller wants to time one
    named hook in isolation).
    """
    env = os.environ.copy()
    env["SKIP"] = (
        f"{env['SKIP']},no-commit-to-branch"
        if "SKIP" in env
        else ("no-commit-to-branch")
    )

    scope = ["--files", *files] if files else ["--all-files"]
    argv = [
        "pre-commit",
        "run",
        *(hook_ids or []),
        "--hook-stage",
        hook_stage,
        *scope,
        "--verbose",
    ]

    return subprocess.run(  # nosec B603 B607
        argv,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


#: A hook whose EXECUTABLE is absent did not find a defect -- it never ran.
#: pre-commit surfaces that as an ordinary ``Failed``, so without this the two
#: are indistinguishable to every caller.
#:
#: Measured 2026-08-21: the repository-manager MCP pod ships git and python but
#: no Rust toolchain, so all SIX of epistemic-graph's cargo/maturin pre-push
#: hooks "failed" in 46 seconds and the push was refused with
#: "Pre-push gate failed (<six hook names>); push aborted. Fix the gate." That
#: reads exactly like a quality verdict. It cost two full investigation cycles
#: before anyone checked whether `cargo` existed in that container -- and the
#: gate could never have passed there regardless of the code.
_UNRUNNABLE_MARKERS = (
    "executable not found",
    "not found in $path",
    "command not found",
    # The Python-level shape (`FileNotFoundError: [Errno 2] No such file or
    # directory: 'node'`) -- deliberately keeping the trailing `: '` so a hook
    # that merely could not open a DATA file ("cat: foo.json: No such file or
    # directory") is not miscredited to a missing toolchain.
    "no such file or directory: '",
    "is not installed",
)


def _is_unrunnable(body: str) -> bool:
    """Did this hook fail because its TOOL is missing, not because the code is?"""
    lowered = body.lower()
    if "executable `" in lowered and "` not found" in lowered:
        return True
    return any(marker in lowered for marker in _UNRUNNABLE_MARKERS)


def parse_gate_output(output: str) -> list[HookResult]:
    """Parse ``pre-commit run --verbose`` output into timed :class:`HookResult`s.

    ``--verbose`` prints, per hook::

        Hook Name................................................Passed
        - hook id: hook-id
        - duration: 1.01s

        <hook stdout/stderr, if any>

    The duration line is parsed here so every caller of :func:`run_gate_stage`
    gets real per-hook timings for free from the same run that already checked
    pass/fail -- no second invocation needed for ``profile``.
    """
    lines = output.split("\n")
    hooks: list[HookResult] = []

    current_hook_name: str | None = None
    current_status: str | None = None
    current_duration: float | None = None
    current_output: list[str] = []

    def _save() -> None:
        if not current_hook_name:
            return
        body = "\n".join(current_output).strip() if current_status == "Failed" else ""
        hooks.append(
            HookResult(
                hook_id=current_hook_name.strip(),
                passed=current_status in ("Passed", "Skipped"),
                output=body,
                duration_s=current_duration,
                unrunnable=bool(body) and _is_unrunnable(body),
            )
        )

    for line in lines:
        stripped = line.strip()
        match = _HOOK_LINE.match(stripped)
        if match:
            _save()
            current_hook_name = match.group(1).strip().rstrip(".")
            current_status = match.group(2)
            current_duration = None
            current_output = []
            continue
        duration_match = _DURATION_LINE.match(stripped)
        if duration_match and current_hook_name:
            current_duration = float(duration_match.group(1))
            continue
        if current_hook_name:
            current_output.append(line)

    _save()
    return hooks


#: The one file whose presence at a repo's root means "a HEAVY-tier run here
#: might invoke `cargo`" -- the same signal
#: :data:`repository_manager.task_queue.EXECUTION_CLASSES`'s ``"build"``
#: class exists to arbitrate (a shared ``CARGO_TARGET_DIR`` corrupts
#: concurrent writers with phantom ``E0599``).
_CARGO_MANIFEST = "Cargo.toml"


def _failing_test_ids(hook: HookResult) -> list[str]:
    """Test ids *hook*'s own captured output reports failed/errored.

    Reuses :func:`repository_manager.merge_queue._pytest_ids` -- the parser
    already proven against pytest's stable ``-rfE`` short-summary contract --
    rather than writing a second one that could quietly drift from it. A hook
    that isn't pytest at all simply yields no ids, which is correct: nothing
    here claims pytest's output contract holds for a hook that never
    promised it.
    """
    return sorted(merge_queue._pytest_ids(hook.output))


def _gate_ledger_metadata(repo_path: str) -> dict[str, Any]:
    """Best-effort environment facts folded into a gate-ledger row.

    Every fact gathered here can fail for a reason that has nothing to do
    with the gate that just ran (no git repo, no ``.mergequeue.yaml``, no
    discoverable ``.venv``, ...), and a ledger row missing one field is still
    a useful row -- so each is independently best-effort and degrades to its
    safe default rather than aborting the whole record. This function never
    raises.
    """
    repo_id = ""
    try:
        repo_id = build_queue.stable_repository_id(repo_path)
    except Exception:  # noqa: BLE001 - metadata gathering must never break a gate verdict
        logger.warning(
            "gate-ledger metadata: could not resolve a stable repo id for %r; "
            "recording the row without one rather than failing the gate",
            repo_path,
            exc_info=True,
        )

    git_sha = ""
    try:
        sha_proc = subprocess.run(  # nosec B603 B607
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if sha_proc.returncode == 0:
            git_sha = sha_proc.stdout.strip()
    except Exception:  # noqa: BLE001 - metadata gathering must never break a gate verdict
        logger.warning(
            "gate-ledger metadata: could not resolve git HEAD for %r; "
            "recording the row without a git_sha rather than failing the gate",
            repo_path,
            exc_info=True,
        )

    tree_dirty = False
    try:
        status_proc = subprocess.run(  # nosec B603 B607
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if status_proc.returncode == 0:
            tree_dirty = bool(status_proc.stdout.strip())
    except Exception:  # noqa: BLE001 - metadata gathering must never break a gate verdict
        logger.warning(
            "gate-ledger metadata: could not resolve git tree-dirty status for "
            "%r; recording the row with tree_dirty=False rather than failing "
            "the gate",
            repo_path,
            exc_info=True,
        )

    toolchain_fingerprint = ""
    try:
        config = merge_queue.load_config(repo_path)
        toolchain_fingerprint = merge_queue._environment_signature(
            Path(repo_path), config
        )
    except Exception:  # noqa: BLE001 - no .mergequeue.yaml degrades, never raises
        logger.warning(
            "gate-ledger metadata: could not compute a toolchain fingerprint "
            "for %r (no .mergequeue.yaml or similar); recording the row "
            "without one rather than failing the gate",
            repo_path,
            exc_info=True,
        )

    return {
        "repo_id": repo_id,
        "git_sha": git_sha,
        "tree_dirty": tree_dirty,
        "toolchain_fingerprint": toolchain_fingerprint,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def _record_gate_run(
    hooks: list[HookResult],
    *,
    repo_path: str,
    stage: str,
    scope: str,
    trigger: str,
    hook_ids_requested: list[str] | None,
    success: bool,
    exit_code: int,
    duration_s: float,
    error: str | None,
) -> str:
    """Map one gate invocation into :meth:`GateLedger.record_run` and write it.

    ``outcome`` is ``"passed"`` when the hook passed, else ``"failed"`` --
    the only two verdicts a pre-commit run itself produces here; ``unrunnable``
    carries through verbatim so a missing toolchain never masquerades as a
    real failure inside the ledger either. Every FAILING hook gets a
    ``failing_tests`` entry even when no test ids were extracted from it
    (an empty list, not a missing key) -- that is what drives the ledger's
    clear-on-improve behaviour for hooks that are not pytest at all, or that
    failed before pytest printed a single ``FAILED``/``ERROR`` line.

    Metadata gathering (:func:`_gate_ledger_metadata`) is already fully
    best-effort. What is NOT swallowed here is a ``ValueError`` from
    :meth:`GateLedger.record_run` itself -- that means THIS function handed
    it a malformed observation (a contract bug in the mapping above, not a
    storage outage), and hiding it would make record_run's own validation
    guard decorative. Every other exception -- a ledger that cannot even be
    constructed, a write that cannot land -- is a lost observation, and
    turning that into a failed gate would be worse than having no ledger.
    """
    metadata = _gate_ledger_metadata(repo_path)
    hooks_payload = [
        {
            "hook_id": h.hook_id,
            "outcome": "passed" if h.passed else "failed",
            "unrunnable": h.unrunnable,
            "duration_s": h.duration_s,
            "output_tail": h.output,
        }
        for h in hooks
    ]
    failing_tests = {h.hook_id: _failing_test_ids(h) for h in hooks if not h.passed}
    try:
        return gate_ledger.default_gate_ledger().record_run(
            repo_id=metadata["repo_id"],
            repo_path=repo_path,
            stage=stage,
            scope=scope,
            trigger=trigger,
            success=success,
            exit_code=exit_code,
            duration_s=duration_s,
            hooks=hooks_payload,
            failing_tests=failing_tests,
            hook_ids_requested=hook_ids_requested,
            git_sha=metadata["git_sha"],
            tree_dirty=metadata["tree_dirty"],
            toolchain_fingerprint=metadata["toolchain_fingerprint"],
            hostname=metadata["hostname"],
            pid=metadata["pid"],
            error=error or "",
        )
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - a ledger outage is not a gate verdict
        return ""


def run_gate_stage(
    repo_path: str,
    stage: GateStage,
    *,
    files: list[str] | None = None,
    hook_ids: list[str] | None = None,
    timeout: int | None = None,
    trigger: str = "run",
    scope: str = "full_wave",
    colocated: bool | None = None,
    record: bool = True,
) -> RepoScanResult:
    """Run one repo's declared hooks at one tier and report the result.

    This is the single execution primitive behind the ``rm_gates`` MCP tool,
    ``rm_projects action=validate`` (via ``Git.validate_single_project``), the
    pre-push gate ``Git.push_project`` runs automatically before every push
    (via ``Git._gate_before_push``), and ``Git.phased_push``'s gate-driven
    phase-transition barrier (via ``dependency_readiness.await_gate_readiness``
    -- CONCEPT:RM-DEP-READY). ``stage="fast"`` passes ``--hook-stage
    pre-commit``; ``stage="heavy"`` passes ``--hook-stage pre-push``.

    ``hook_ids`` (non-empty) narrows the run to those specific declared hook
    ids instead of every hook at ``stage`` -- e.g. the phase-transition
    barrier scopes to just ``["dependency-readiness"]`` so a retry loop reruns
    one fast network-bound check, not a repo's entire heavy suite (pytest,
    mypy, ...) on every poll. A repo whose ``.pre-commit-config.yaml`` does
    not declare a requested hook id reports as a failure with no parseable
    hooks (pre-commit's own "No hook with ID" error surfaces via ``error``) --
    never silently treated as "nothing to check".

    Returns a :class:`RepoScanResult` whose ``hooks`` carry per-hook pass/fail
    AND timing (``HookResult.duration_s``), and whose top-level ``duration_s``
    is the measured wall-clock time of the whole invocation.

    ``trigger``/``scope`` are recorded facts, not behaviour switches -- they
    answer "why did this run happen" (``"run"``/``"validate"``/``"pre-push"``,
    ...) and "how much of the suite did it cover" (``"full_wave"`` the
    default, or a caller-defined narrower scope such as a retest) for the
    gate ledger (see below); they do not change what pre-commit is asked to
    do. ``colocated`` and ``record`` gate two independent pieces of new
    behaviour:

    * **Build reservation.** When ``stage="heavy"`` and the repo has a
      ``Cargo.toml`` at its root, this function is about to invoke
      ``pre-commit``'s pre-push hooks, which may run ``cargo build`` --
      contending for the same shared ``CARGO_TARGET_DIR`` corruption hazard
      :data:`repository_manager.task_queue.EXECUTION_CLASSES`'s ``"build"``
      class exists to arbitrate. ``colocated=True`` takes that reservation
      (via :func:`repository_manager.task_queue.acquire`) before running the
      gate and releases it after; a refused reservation is reported back as
      ``RepoScanResult(success=False, deferred=True, ...)`` -- the gate did
      NOT run, so this is a deferral to retry, never a verdict about the
      code. ``colocated`` left at its default (``None``/not ``True``) skips
      the reservation entirely rather than erroring, exactly like
      :func:`task_queue.acquire`'s own PARTITION-policy exemption -- a repo
      with no Rust toolchain has nothing to contend over, and a caller with
      no proof of same-node execution should not be forced to reason about a
      resource it cannot safely arbitrate.
    * **Ledger recording.** With ``record=True`` (the default), the finished
      run -- every hook's outcome, any pytest ids it reported failing, and
      the toolchain/commit metadata the run happened under -- is written to
      :func:`repository_manager.gate_ledger.default_gate_ledger` so a later
      caller can ask "what failed last time?" without re-running the whole
      wave (see that module's docstring for the incident this closes). The
      returned :class:`RepoScanResult`'s ``run_id`` carries the ledger's
      identifier for that write, or ``""`` when ``record=False`` or the
      write could not complete. Recording is best-effort end to end: a
      ledger outage never changes ``success``/``exit_code``/``hooks`` below.
    """
    if stage not in HOOK_STAGE_BY_GATE_STAGE:
        raise ValueError(f"unknown gate stage {stage!r}; expected 'fast' or 'heavy'")

    if not os.path.exists(os.path.join(repo_path, ".pre-commit-config.yaml")):
        return RepoScanResult(
            repo_path=repo_path,
            success=True,
            exit_code=0,
            hooks=[],
            error="No .pre-commit-config.yaml found.",
            stage=stage,
        )

    hook_stage = HOOK_STAGE_BY_GATE_STAGE[stage]
    effective_timeout = timeout if timeout is not None else default_gate_timeout(stage)

    needs_build_reservation = (
        stage == "heavy"
        and colocated is True
        and os.path.exists(os.path.join(repo_path, _CARGO_MANIFEST))
    )

    started = time.monotonic()
    try:
        if needs_build_reservation:
            with task_queue.acquire(
                "build", operation=f"gate:{stage}", path=repo_path, colocated=True
            ):
                result = _run_pre_commit(
                    repo_path,
                    hook_stage,
                    files=files,
                    hook_ids=hook_ids,
                    timeout=effective_timeout,
                )
        else:
            result = _run_pre_commit(
                repo_path,
                hook_stage,
                files=files,
                hook_ids=hook_ids,
                timeout=effective_timeout,
            )
    except task_queue.TaskQueueError as exc:
        # The build reservation was refused (every POOL slot busy, or -- were
        # this call ever made without colocated=True -- ColocationRequired).
        # This is a DEFERRAL, not a verdict: the gate never ran, so there is
        # nothing here to have found a defect. Reporting it as an ordinary
        # failure is how transient build contention turns into a false
        # regression on whatever code happened to be checked out at the time.
        return RepoScanResult(
            repo_path=repo_path,
            success=False,
            exit_code=-1,
            deferred=True,
            error=(
                f"DEFERRED, not a verdict: the HEAVY gate for {repo_path} needs a "
                f"'build' reservation (a Cargo.toml is present) and none was free: "
                f"{exc}. The gate did not run against this code; retry once a build "
                f"slot frees rather than treating this as a hook failure."
            ),
            stage=stage,
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired:
        return RepoScanResult(
            repo_path=repo_path,
            success=False,
            exit_code=-1,
            error=(
                f"TIMEOUT after {effective_timeout}s running the {stage!r} "
                f"({hook_stage}) gate -- the gate did NOT finish, so this is "
                f"not a hook verdict. Raise {_TIMEOUT_ENV_VAR} or speed up the "
                f"suite; do not read this as the gate finding a defect."
            ),
            stage=stage,
            duration_s=time.monotonic() - started,
        )
    except Exception as e:
        return RepoScanResult(
            repo_path=repo_path,
            success=False,
            exit_code=-1,
            error=f"Error executing the {stage!r} ({hook_stage}) gate: {type(e).__name__}",
            stage=stage,
            duration_s=time.monotonic() - started,
        )
    duration_s = time.monotonic() - started

    raw_output = result.stdout + "\n" + result.stderr
    hooks = parse_gate_output(raw_output)
    success = result.returncode == 0

    # When pre-commit exits non-zero but produced NO parseable failing hook
    # lines (env/install/config error before any hook ran), the real
    # diagnostic lives only in raw_output -- surface it as `error` so the
    # failure is never silently empty.
    error: str | None = None
    if not success and not any(not h.passed for h in hooks):
        tail = "\n".join(raw_output.strip().splitlines()[-40:]).strip()
        error = (
            f"pre-commit --hook-stage {hook_stage} exited {result.returncode} "
            f"with no parseable hook results (likely an init/config/environment "
            f"error):\n{tail}"
            if tail
            else f"pre-commit --hook-stage {hook_stage} exited {result.returncode} with no output."
        )

    run_id = ""
    if record:
        run_id = _record_gate_run(
            hooks,
            repo_path=repo_path,
            stage=stage,
            scope=scope,
            trigger=trigger,
            hook_ids_requested=hook_ids,
            success=success,
            exit_code=result.returncode,
            duration_s=duration_s,
            error=error,
        )

    return RepoScanResult(
        repo_path=repo_path,
        success=success,
        exit_code=result.returncode,
        hooks=hooks,
        raw_output=raw_output,
        error=error,
        stage=stage,
        duration_s=duration_s,
        run_id=run_id,
    )


def explain_gate_result(result: RepoScanResult, *, max_lines_per_hook: int = 12) -> str:
    """Condense a :class:`RepoScanResult` to the actionable lines: what failed, why.

    Powers ``rm_gates action=explain`` -- a full ``raw_output`` dump is often
    thousands of lines (every passing hook's verbose header); this keeps only
    the failing hooks' own output, trimmed.
    """
    label = result.stage or "gate"
    if result.success:
        return f"{result.repo_path}: {label} passed ({len(result.hooks)} hook(s))."

    parts = [f"{result.repo_path}: {label} FAILED (exit {result.exit_code})"]
    failed = [h for h in result.hooks if not h.passed]
    if not failed and result.error:
        parts.append(result.error)
    for h in failed:
        body_lines = [ln for ln in h.output.splitlines() if ln.strip()]
        snippet = "\n".join(body_lines[:max_lines_per_hook])
        if len(body_lines) > max_lines_per_hook:
            snippet += (
                f"\n... ({len(body_lines) - max_lines_per_hook} more line(s) truncated)"
            )
        parts.append(
            f"- {h.hook_id}:\n{snippet}"
            if snippet
            else f"- {h.hook_id}: failed (no captured output)"
        )
    return "\n".join(parts)


def profile_gate_result(result: RepoScanResult) -> list[dict[str, Any]]:
    """Per-hook timings for one gate run, slowest first.

    Powers ``rm_gates action=profile``: without measured per-hook numbers the
    FAST tier silently rots back into a slow tier, and there is no way to see
    which hook to trim.
    """
    return [
        {
            "hook_id": h.hook_id,
            "duration_s": h.duration_s,
            "passed": h.passed,
        }
        for h in sorted(result.hooks, key=lambda h: h.duration_s or 0.0, reverse=True)
    ]
