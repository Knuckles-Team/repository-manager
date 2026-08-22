from pydantic import BaseModel


class HookResult(BaseModel):
    hook_id: str
    passed: bool
    output: str = ""
    #: Wall-clock seconds pre-commit reported for this hook (parsed from its own
    #: ``--verbose`` ``- duration: <n>s`` line). ``None`` when not measured/parsed.
    duration_s: float | None = None
    #: True when this hook did not FAIL, it could not RUN: its executable is
    #: absent from this environment (``Executable ... not found`` /
    #: ``command not found``). "The toolchain is missing here" and "your code is
    #: bad" are different answers and must never be reported as the same one.
    unrunnable: bool = False


class RepoScanResult(BaseModel):
    repo_path: str
    success: bool
    exit_code: int
    hooks: list[HookResult] = []
    raw_output: str = ""
    pytest_output: str | None = None
    error: str | None = None
    #: The gate tier this result was produced at ("fast"/"heavy"), or "" when
    #: the result predates tiering / doesn't apply (e.g. skipped, no config).
    stage: str = ""
    #: Total wall-clock seconds for the whole gate invocation.
    duration_s: float = 0.0
    #: The gate ledger's ``run_id`` for this invocation, when the run was
    #: recorded (:mod:`repository_manager.gate_ledger`). Empty when recording
    #: was disabled (``record=False``) or the result predates the ledger.
    run_id: str = ""
    #: True when the gate did not run because a reservation was refused --
    #: this is NOT a failing verdict about the code. It means "we don't know
    #: yet", not "the code is broken": a build reservation
    #: (:mod:`repository_manager.task_queue`) was busy, so
    #: :func:`repository_manager.gates.run_gate_stage` deferred the run
    #: rather than letting two concurrent heavy gates corrupt a shared
    #: ``target/`` directory. ``success=False`` alongside ``deferred=True``
    #: must be read as "retry", never as "fix the code" -- a caller that
    #: conflates the two turns transient contention into a false regression.
    deferred: bool = False
