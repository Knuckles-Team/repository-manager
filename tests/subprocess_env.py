"""Subprocess environments that prove what a base install sees."""

from __future__ import annotations

import os
from pathlib import Path


def base_install_env() -> dict[str, str]:
    """The current environment minus any ``PYTHONPATH`` entry that carries the RMDD-14 seam.

    Dropping ``PYTHONPATH`` wholesale also drops the locked environment's
    site-packages when CI exposes it that way, so a fresh interpreter could not
    import the SDK. Only entries holding a ``tunnel_manager/remote_execution``
    checkout are removed.
    """
    env = dict(os.environ)
    kept = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and not (Path(entry) / "tunnel_manager" / "remote_execution").exists()
    ]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    return env


def with_env(extra: dict[str, str] | None) -> dict[str, str]:
    """The current environment plus ``extra``; an extra ``PYTHONPATH`` goes in front of the inherited one."""
    env = dict(os.environ)
    env.update(extra or {})
    inherited = os.environ.get("PYTHONPATH")
    if inherited and "PYTHONPATH" in (extra or {}):
        env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], inherited])
    return env
