"""Helpers that turn POSIX-looking fixture paths into native absolute paths."""

from __future__ import annotations

import os
from pathlib import Path


def native_abs(posix_path: str) -> str:
    """Return ``posix_path`` as a canonical absolute path on this platform.

    POSIX paths pass through unchanged; on Windows ``/home/app/x`` becomes
    ``<drive>:\\home\\app\\x`` (an absolute, canonical path).
    """
    if os.name != "nt":
        return posix_path
    return Path.cwd().anchor + posix_path.lstrip("/").replace("/", "\\")
