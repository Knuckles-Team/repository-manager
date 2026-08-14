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
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - fixed argv only, never shell=True
import time
from typing import Any, Literal

from repository_manager.scan_models import HookResult, RepoScanResult

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
        hooks.append(
            HookResult(
                hook_id=current_hook_name.strip(),
                passed=current_status in ("Passed", "Skipped"),
                output=(
                    "\n".join(current_output).strip()
                    if current_status == "Failed"
                    else ""
                ),
                duration_s=current_duration,
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


def run_gate_stage(
    repo_path: str,
    stage: GateStage,
    *,
    files: list[str] | None = None,
    hook_ids: list[str] | None = None,
    timeout: int | None = None,
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
    started = time.monotonic()
    try:
        result = _run_pre_commit(
            repo_path,
            hook_stage,
            files=files,
            hook_ids=hook_ids,
            timeout=effective_timeout,
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

    return RepoScanResult(
        repo_path=repo_path,
        success=success,
        exit_code=result.returncode,
        hooks=hooks,
        raw_output=raw_output,
        error=error,
        stage=stage,
        duration_s=duration_s,
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
