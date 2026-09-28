#!/usr/bin/env python3
"""What a gate does when it cannot run here.

A fresh clone may lack a tool (``docker``, ``cccc``), the locked project
environment, or the pinned ``agent-utilities`` sibling that
``scripts/bootstrap.sh`` provides. Such a gate has found nothing, so it must
not report a pass -- but it must not block a contributor's commit either:

* locally it prints ``SKIPPED (<gate>): <reason>`` and exits 0;
* in CI (``CI`` set) it prints ``CANNOT RUN`` and exits 2, because CI runs
  ``scripts/bootstrap.sh`` first and is expected to provide everything.

As a command it runs a tool-dependent command only when the tool exists::

    python3 scripts/gate_env.py --gate docker-compose-check --need docker -- \\
        docker compose ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404 - argv from the hook definition, no shell
import sys


def unavailable(gate: str, reason: str) -> int:
    """Fail closed in CI; skip visibly everywhere else."""

    if os.environ.get("CI"):
        print(f"{gate}: CANNOT RUN in CI: {reason}", file=sys.stderr)
        return 2
    print(f"SKIPPED ({gate}): {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", required=True)
    parser.add_argument("--need", action="append", default=[], metavar="TOOL")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command:
        parser.error("a command is required after --")
    for tool in options.need:
        if shutil.which(tool) is None:
            return unavailable(options.gate, f"`{tool}` is not installed")
    return subprocess.run(command).returncode  # nosec B603


if __name__ == "__main__":
    sys.exit(main())
