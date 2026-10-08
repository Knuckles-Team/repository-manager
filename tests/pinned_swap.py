"""Test helper for attacks that rename a directory the boundary has pinned."""

from __future__ import annotations

import os
from collections.abc import Callable


def attempt_swap(action: Callable[[], None]) -> bool:
    """Run ``action`` (a rename-based swap) and report whether it happened.

    POSIX pins with descriptors, so the swap succeeds and the boundary must
    detect it.  Windows pins directories without delete sharing, so renaming a
    pinned directory fails with a sharing violation: the attack is prevented
    outright and the operation keeps acting on the pinned original.
    """
    try:
        action()
    except PermissionError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) in {5, 32}:
            return False
        raise
    return True
