"""Supervised job execution whose result cannot silently claim success.

CONCEPT:RM-JOB-OUTCOME

Three false greens in one session, all from the same root: **a result that
does not carry how the process ENDED is not a result.**

1. A ``cargo test`` stage was OOM-killed at ``MemoryMax=48G`` during linking.
   The unit had been launched with ``systemd-run --collect``, which reaps the
   unit once it exits — so the follow-up ``systemctl show`` found no unit and
   returned DEFAULTS: ``Result=success``, ``ExecMainStatus=0``. A killed build
   reported as a clean one. The only surviving evidence was the ABSENCE of a
   sentinel line in the log.
2. Several stages were chained as ``{ a; b; c; } ; echo EXIT=$?`` — which
   records only the LAST command's status. A mid-chain abort (a Rust test
   binary dying on stack overflow, SIGABRT) was masked by a later stage's
   clean exit, and the run was read as "0 failed".
3. A monitor scored a killed run by grepping its truncated log for failures,
   found none because the run never got far enough to print any, and reported
   zero.

Every one of those is the same bug: success was INFERRED from the absence of
visible failure. So the contract here is inverted — a job is ``ok`` only when
it positively proves completion:

* each stage runs and is recorded SEPARATELY, so no stage can hide behind
  another's exit code;
* completion is proven by a sentinel the runner itself writes AFTER the
  command returns, never by an exit code read back from a reaped unit;
* a missing sentinel is classified ``killed``, never ``passed``;
* resource limits and their outcome travel WITH the result, so "it OOMed" is a
  first-class answer rather than something a human has to go find in a journal.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess  # nosec B404 - fixed argv, no shell
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "COMPLETED",
    "FAILED",
    "KILLED",
    "ResourceProfile",
    "StageOutcome",
    "JobOutcome",
    "run_stages",
    "classify_log",
    "dispatch",
]

#: The stage ran to completion and reported success.
COMPLETED = "completed"
#: The stage ran to completion and reported failure. A REAL result.
FAILED = "failed"
#: The stage never finished. NOT a failure of the code — a failure to measure.
#: Must never be collapsed into either of the above.
KILLED = "killed"

#: Written by the runner immediately after the command returns. Its presence is
#: the ONLY proof a stage completed; its absence means the process died before
#: the runner regained control.
_SENTINEL = "__RM_STAGE_EXIT__"

_SENTINEL_RE = re.compile(rf"^{re.escape(_SENTINEL)}=(-?\d+)$", re.MULTILINE)


@dataclass
class ResourceProfile:
    """Declared limits for a job, and the knobs that enforce them.

    Hand-copied magic numbers are how the 48G OOM happened: the figure came
    from a handoff document, was never matched to the workload, and silently
    truncated the most expensive stage in the repository. Limits belong with
    the job definition (``workspace.yml``), reviewed like anything else.
    """

    memory_max: str = ""
    cpu_quota: str = ""
    jobs: int = 0
    #: systemd's OOM preference. Deliberately optional and OFF by default:
    #: ``ManagedOOMPreference`` is unsupported on some systemd builds in this
    #: fleet, where passing it makes the unit fail to start AT ALL — turning a
    #: protected run into no run. Opt in per host, never assume.
    oom_preference: str = ""

    def unit_args(self) -> list[str]:
        args: list[str] = []
        if self.memory_max:
            args += ["-p", f"MemoryMax={self.memory_max}"]
        if self.cpu_quota:
            args += ["-p", f"CPUQuota={self.cpu_quota}"]
        if self.oom_preference:
            args += ["-p", f"ManagedOOMPreference={self.oom_preference}"]
        return args

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageOutcome:
    """One stage's result, including HOW it ended."""

    name: str
    status: str
    exit_code: int | None = None
    duration_seconds: float = 0.0
    log_path: str = ""
    termination: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """True only when this stage produced a trustworthy verdict."""
        return self.status in (COMPLETED, FAILED)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "measured": self.measured}


@dataclass
class JobOutcome:
    """A whole job. ``ok`` requires EVERY stage to have completed successfully."""

    ok: bool
    fully_measured: bool
    stages: list[StageOutcome] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    host: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fully_measured": self.fully_measured,
            "host": self.host,
            "profile": self.profile,
            "stages": [s.as_dict() for s in self.stages],
        }


def classify_log(log_text: str) -> tuple[str, int | None]:
    """Classify a stage from its log alone: ``(status, exit_code)``.

    The sentinel is the whole contract. No sentinel means the runner never
    regained control after the command — the process was killed (OOM, SIGKILL,
    an aborting test binary taking the harness with it, a reboot). That is
    reported as :data:`KILLED`, which is neither pass nor fail: it is the
    absence of a measurement, and it must stay distinguishable from both.
    """
    # The LAST sentinel is this stage's; earlier ones belong to earlier commands
    # that wrote into the same log.
    codes = _SENTINEL_RE.findall(log_text)
    if not codes:
        return KILLED, None
    code = int(codes[-1])
    return (COMPLETED if code == 0 else FAILED), code


def _oom_evidence(unit: str) -> dict[str, Any]:
    """Ask the journal why a unit died. Best-effort; never raises."""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["journalctl", "--user", "-u", unit, "-n", "50", "--no-pager"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    text = proc.stdout
    evidence: dict[str, Any] = {}
    if "oom-kill" in text or "Failed with result 'oom-kill'" in text:
        evidence["termination"] = "oom-kill"
    peak = re.search(r"(\d+(?:\.\d+)?[KMGT]?) memory peak", text)
    if peak:
        evidence["memory_peak"] = peak.group(1)
    return evidence


def run_stages(
    stages: Sequence[tuple[str, str]],
    *,
    workdir: Path | str,
    log_dir: Path | str,
    profile: ResourceProfile | None = None,
    env: Mapping[str, str] | None = None,
    use_systemd: bool = True,
) -> dict[str, Any]:
    """Run ``(name, shell_command)`` stages, each recorded independently.

    Each stage gets its OWN unit and its OWN log, because a shared exit code is
    exactly how a mid-chain abort disappears. A stage that is killed does not
    stop later stages from running — but it does permanently mark the job as
    not fully measured, so no downstream consumer can read the job as green.
    """
    profile = profile or ResourceProfile()
    workdir = Path(workdir).resolve()
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    environ = dict(env or {})

    outcomes: list[StageOutcome] = []
    for name, command in stages:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "stage"
        unit = f"rmjob-{slug}-{uuid.uuid4().hex[:8]}"
        log_path = log_dir / f"{slug}.log"
        # The sentinel is appended by the SHELL, after the command returns.
        # If the command takes the shell down with it, no sentinel is written —
        # which is precisely the signal we want.
        wrapped = (
            f"{command} > {shlex.quote(str(log_path))} 2>&1; "
            f'echo "{_SENTINEL}=$?" >> {shlex.quote(str(log_path))}'
        )

        started = time.monotonic()
        if use_systemd:
            argv = [
                "systemd-run",
                "--user",
                "--collect",
                "--wait",
                f"--unit={unit}",
                f"--working-directory={workdir}",
                *profile.unit_args(),
                *[f"--setenv={k}={v}" for k, v in environ.items()],
                "bash",
                "-c",
                wrapped,
            ]
        else:
            argv = ["bash", "-c", wrapped]
        try:
            subprocess.run(  # nosec B603 - fixed argv, no shell
                argv,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            outcomes.append(
                StageOutcome(
                    name=name,
                    status=KILLED,
                    log_path=str(log_path),
                    termination=f"runner error: {exc}",
                    duration_seconds=time.monotonic() - started,
                )
            )
            continue

        duration = time.monotonic() - started
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        status, code = classify_log(text)
        outcome = StageOutcome(
            name=name,
            status=status,
            exit_code=code,
            duration_seconds=duration,
            log_path=str(log_path),
        )
        if status == KILLED:
            evidence = _oom_evidence(unit) if use_systemd else {}
            outcome.termination = evidence.get("termination", "no completion sentinel")
            outcome.evidence = evidence
        outcomes.append(outcome)

    fully_measured = all(o.measured for o in outcomes)
    job = JobOutcome(
        # `ok` demands positive proof from EVERY stage. An unmeasured stage can
        # never contribute to a green verdict.
        ok=bool(outcomes)
        and fully_measured
        and all(o.status == COMPLETED for o in outcomes),
        fully_measured=fully_measured,
        stages=outcomes,
        profile=profile.as_dict(),
        host=os.uname().nodename,
    )
    return job.as_dict()


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core shared by the CLI and the MCP tool so they cannot drift."""
    if action == "classify":
        path = Path(str(kwargs.get("log") or ""))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        status, code = classify_log(text)
        return {"ok": status == COMPLETED, "status": status, "exit_code": code}
    if action == "run":
        raw = kwargs.get("stages") or []
        stages = [(str(s["name"]), str(s["command"])) for s in raw]
        prof = kwargs.get("profile") or {}
        return run_stages(
            stages,
            workdir=kwargs.get("workdir") or Path.cwd(),
            log_dir=kwargs.get("log_dir") or Path.cwd(),
            profile=ResourceProfile(**prof),
            env=kwargs.get("env"),
            use_systemd=bool(kwargs.get("use_systemd", True)),
        )
    return {"ok": False, "error": f"unknown action: {action}"}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(prog="job-outcome", description=__doc__)
    parser.add_argument("log", help="a stage log to classify")
    args = parser.parse_args(argv)
    result = dispatch("classify", log=args.log)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
