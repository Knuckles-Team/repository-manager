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

**Every runner truncates, not just cargo — so this now STRIPS as well as adds.**
The original docstring promised this function "only ever ADDS the one flag";
that is no longer true and the promise is withdrawn deliberately. cargo's
truncation is opt-out (omit a flag and it stops early), but pytest's and go
test's are opt-IN (``-x`` / ``--exitfirst`` / ``--maxfail=N`` / ``-failfast``),
so for those the structurally-correct invocation is produced by REMOVING what
the caller declared, not by appending. A gate that stops at the first failure
reports one defect per wave; on a 90-minute suite that is one defect per 90
minutes, which is how a single push consumed a day. Both directions serve the
same invariant — *the run reports every failure it found* — and both are
applied at the same chokepoint so neither can be bypassed per-caller.

**Scope boundary, stated so it is never mistaken for complete.** This governs
argv that *this package constructs*. It cannot govern a repository's own
``.pre-commit-config.yaml`` ``entry:`` text, because
:func:`repository_manager.gates.run_gate_stage` never builds those argv — it
shells to ``pre-commit run`` and each repo's entry line is opaque shell.
Detecting a fail-fast flag hiding in one of those is a separate, static concern
(:mod:`repository_manager.fail_fast_audit`), and that is a DETECTION, not a
prevention. Do not describe the never-stop-early guarantee as closed for
pre-commit-driven gates anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "ensure_no_fail_fast",
    "is_go_test_command",
    "is_pytest_command",
    "is_test_suite_command",
]

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


#: Launchers a pytest invocation is commonly wrapped in across this fleet's
#: pre-commit entries and scripts. Recognition walks past these so
#: ``uv run --all-extras pytest`` is seen for what it is.
_LAUNCHERS = frozenset({"uv", "uvx", "poetry", "hatch", "pdm", "nox", "tox"})


def _skip_launchers(argv: Sequence[str]) -> list[str]:
    """Drop wrapper tokens so the real program lands at index 0.

    ``uv run --all-extras pytest tests`` -> ``pytest tests``;
    ``python -m pytest tests`` -> ``pytest tests``. Options belonging to the
    wrapper are skipped, but the FIRST non-option token after a launcher is
    taken as the program, so a wrapper option that takes a separate value could
    in principle be mistaken for it. That mis-read can only cause this module to
    decline to recognise a command (leaving argv untouched), never to rewrite
    one it should not have — the failure direction is deliberate.
    """

    tokens = list(argv)
    while tokens:
        head = tokens[0]
        if head in _LAUNCHERS:
            tokens = tokens[1:]
            # `uv run ...`, `poetry run ...` — step over the subcommand too.
            if tokens and tokens[0] in {"run", "exec"}:
                tokens = tokens[1:]
            while tokens and tokens[0].startswith("-"):
                tokens = tokens[1:]
            continue
        if head in {"python", "python3"} or head.startswith("python3."):
            rest = tokens[1:]
            while rest and rest[0].startswith("-") and rest[0] != "-m":
                rest = rest[1:]
            if rest and rest[0] == "-m":
                tokens = rest[1:]
                continue
        break
    return tokens


def is_pytest_command(argv: Sequence[str]) -> bool:
    """Return whether *argv* ultimately invokes ``pytest``."""

    tokens = _skip_launchers(argv)
    if not tokens:
        return False
    program = tokens[0].rsplit("/", 1)[-1]
    return program == "pytest" or program.startswith("pytest.")


def is_go_test_command(argv: Sequence[str]) -> bool:
    """Return whether *argv* is a ``go test`` invocation."""

    tokens = _skip_launchers(argv)
    return (
        len(tokens) >= 2
        and tokens[0].rsplit("/", 1)[-1] == "go"
        and tokens[1] == "test"
    )


def _strip_pytest_fail_fast(argv: Sequence[str]) -> list[str]:
    """Remove pytest's early-exit flags, including from bundled short options.

    Handles ``-x``, ``--exitfirst``, ``--maxfail=N``, the two-token
    ``--maxfail N``, and bundled shorts such as ``-xvs`` (which becomes
    ``-vs``; a bundle that held only ``-x`` disappears entirely).
    ``--maxfail=0`` is left alone — pytest reads 0 as "no limit", so it is not
    a truncation and rewriting it would change nothing but the caller's intent.
    """

    result: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-x", "--exitfirst"}:
            continue
        if arg == "--maxfail":
            # Two-arg form: drop the flag and only drop its value if that
            # value is actually a truncating count.
            skip_next = True
            continue
        if arg.startswith("--maxfail="):
            value = arg.split("=", 1)[1]
            if value.strip() == "0":
                result.append(arg)
            continue
        if (
            len(arg) > 1
            and arg[0] == "-"
            and not arg.startswith("--")
            and "x" in arg[1:]
            and all(character.isalpha() for character in arg[1:])
        ):
            remainder = "".join(c for c in arg[1:] if c != "x")
            if remainder:
                result.append(f"-{remainder}")
            continue
        result.append(arg)
    return result


def _strip_go_fail_fast(argv: Sequence[str]) -> list[str]:
    """Remove ``go test``'s ``-failfast`` (both ``-`` and ``--`` spellings)."""

    return [token for token in argv if token not in {"-failfast", "--failfast"}]


def ensure_no_fail_fast(argv: Sequence[str]) -> list[str]:
    """Return *argv* rewritten so the run cannot stop at its first failure.

    ``cargo test``/``cargo nextest run`` gain ``--no-fail-fast`` (idempotent).
    ``pytest`` loses ``-x``/``--exitfirst``/``--maxfail=N``. ``go test`` loses
    ``-failfast``. Anything else — ``cargo check``, ``cargo clippy``, a build,
    an unrecognised program — is returned unchanged, byte for byte.

    Recognition is by argv shape, never by a caller-supplied label, so a
    command declared in one repo's config and a command assembled in code are
    treated identically.
    """

    result = list(argv)
    if is_test_suite_command(result):
        if _FLAG not in result:
            result.append(_FLAG)
        return result
    if is_pytest_command(result):
        return _strip_pytest_fail_fast(result)
    if is_go_test_command(result):
        return _strip_go_fail_fast(result)
    return result
