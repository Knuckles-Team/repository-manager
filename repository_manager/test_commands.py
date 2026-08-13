"""Structurally correct test-runner invocations (CONCEPT:RM-TEST-COMMANDS).

**The invariant.** A ``cargo test``/``cargo nextest run`` invocation that omits
``--no-fail-fast`` stops at the FIRST failing test binary and never runs the
rest — every later failure is simply never reported. This has exactly one
correct setting (always pass it) and was previously enforced only by prose:
"always pass ``--no-fail-fast``" in a preset's comments, or in an operator's
memory. Prose is not enforcement — this program's own measurement lesson
records ``--no-fail-fast`` absent from all 8 of a prior session's eg
measurement points, silently truncating every suite number it reported.

**The fix.** The command a caller *declares* (in ``.mergequeue.yaml`` /
``.buildcache.yaml``) is not the command that runs. :func:`ensure_no_fail_fast`
is applied by the RUNNER, at the process-launch chokepoint
(:func:`repository_manager.merge_queue._timed_run`,
:func:`repository_manager.build_queue._run_build_command`), after config is
loaded and immediately before ``subprocess.run``. A declared command that
already carries the flag is left untouched (idempotent); one that omits it
gets it appended. A caller cannot construct an executed ``cargo test``/
``cargo nextest run`` invocation without it — the omission is no longer
representable in the argv that actually gets spawned.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["ensure_no_fail_fast", "is_test_suite_command"]

_FLAG = "--no-fail-fast"


def is_test_suite_command(argv: Sequence[str]) -> bool:
    """Return whether *argv* is a ``cargo test``/``cargo nextest run`` invocation."""

    if len(argv) >= 2 and argv[0] == "cargo" and argv[1] == "test":
        return True
    return bool(
        len(argv) >= 3
        and argv[0] == "cargo"
        and argv[1] == "nextest"
        and argv[2] == "run"
    )


def ensure_no_fail_fast(argv: Sequence[str]) -> list[str]:
    """Return *argv*, with ``--no-fail-fast`` appended if this is a test-suite
    command that omitted it.

    Every other command (``cargo check``, ``cargo clippy``, ``pytest``,
    anything not recognised as a full-suite test run) is returned unchanged —
    this only ever ADDS the one flag whose absence is known to silently
    truncate results, never rewrites or reorders anything else the caller
    declared.
    """

    result = list(argv)
    if is_test_suite_command(result) and _FLAG not in result:
        result.append(_FLAG)
    return result
