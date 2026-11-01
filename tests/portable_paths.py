"""Helpers that turn POSIX-looking fixture paths into native absolute paths."""

from __future__ import annotations

import os
import re
from pathlib import Path


def native_abs(posix_path: str) -> str:
    """Return ``posix_path`` as a canonical absolute path on this platform.

    POSIX paths pass through unchanged; on Windows ``/home/app/x`` becomes
    ``<drive>:\\home\\app\\x`` (an absolute, canonical path).
    """
    if os.name != "nt":
        return posix_path
    return Path.cwd().anchor + posix_path.lstrip("/").replace("/", "\\")


SHELL = os.name != "nt"


def _git_sh() -> str:
    import shutil

    git = shutil.which("git")
    if git is not None:
        root = Path(git).resolve().parent.parent
        for candidate in (root / "usr" / "bin" / "sh.exe", root / "bin" / "sh.exe"):
            if candidate.exists():
                return str(candidate)
    return "sh"


def sh_arg(command: str) -> str | list[str]:
    """Argument for ``subprocess.run(..., shell=SHELL)`` running a POSIX shell line.

    POSIX passes the line to the system shell.  Windows has no POSIX shell
    (``cmd.exe`` mis-parses single quotes and ``&&`` chains), so the line runs
    through the POSIX ``sh`` that ships with Git for Windows.
    """
    if SHELL:
        return command
    # sh treats a backslash as an escape; drive paths need forward slashes.
    posix_line = re.sub(
        r"[A-Za-z]:(?:\\[^\s'\"&;|]*)+",
        lambda match: match.group(0).replace("\\", "/"),
        command,
    )
    return [_git_sh(), "-c", posix_line]
