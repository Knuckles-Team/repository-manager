"""Test helper: write a small Python program that runs as an executable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def write_python_program(path: Path, body: str) -> Path:
    """Write ``body`` as a directly runnable program and return its path.

    POSIX gets a ``#!/usr/bin/env python3`` file with the executable bit set.
    Windows cannot execute a script file directly, so the body is written to
    ``<path>.py`` and the returned path is a ``<path>.cmd`` shim that runs it
    with this interpreter, forwarding arguments and the exit status.
    """
    if os.name != "nt":
        path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return path
    script = path.with_name(path.name + ".py")
    script.write_text(body, encoding="utf-8")
    shim = path.with_name(path.name + ".cmd")
    shim.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    return shim
