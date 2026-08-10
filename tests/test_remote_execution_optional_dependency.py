"""RMDD-15: the optional ``tunnel-manager`` dependency must never break the
base install.

RMDD-14's ``tunnel_manager.remote_execution`` seam exists only on
tunnel-manager's own unmerged integration branch and predates every published
PyPI release, so ``repository-manager`` cannot declare a resolvable
dependency on it yet (see ``repository_manager/remote_execution/README.md``,
"Optional dependency: tunnel-manager is required but not yet applied"). These
tests prove the package degrades honestly -- ``import
repository_manager.remote_execution`` must succeed with zero new hard
dependency, and only an actual attempt to build/run a remote dispatch must
raise a clear, named error.

Before the fix in this lane, ``import repository_manager.remote_execution``
raised a bare ``ModuleNotFoundError: No module named 'tunnel_manager'`` three
frames deep inside ``executor.py`` -- reproduced verbatim in the lane handoff.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _clean_env() -> dict[str, str]:
    """A subprocess environment with no inherited ``PYTHONPATH``.

    This test's whole point is proving what the *base install* sees, so it
    must not depend on whether the outer test run happens to have a
    tunnel-manager checkout on ``PYTHONPATH`` (e.g. the way the tunnel-manager
    dependent tests in this suite are exercised -- see
    ``test_remote_execution_executor.py``'s module docstring).  Explicitly
    strip it rather than relying on the invoking shell's environment.
    """

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_base_import_succeeds_without_tunnel_manager() -> None:
    """The package must import cleanly with no tunnel-manager on ``sys.path``.

    Run in a fresh subprocess so an already-imported ``tunnel_manager`` (e.g.
    from a sibling test module that opts into the optional dependency) cannot
    mask a regression here.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import repository_manager.remote_execution as m; "
            "assert 'tunnel_manager' not in __import__('sys').modules; "
            "print('OK', sorted(m.__all__))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout
    assert "tunnel_manager" not in completed.stderr


def test_registry_module_imports_without_tunnel_manager() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import repository_manager.remote_execution.registry; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert completed.returncode == 0, completed.stderr


def test_constructing_remote_worker_executor_without_tunnel_manager_refuses() -> None:
    from repository_manager.remote_execution.executor import (
        RemoteExecutionUnavailableError,
        RemoteWorkerExecutor,
    )

    if "tunnel_manager" in sys.modules:
        pytest.skip(
            "tunnel_manager already imported by another test in this process; "
            "the unavailable-dependency path is proven by the subprocess tests above"
        )

    with pytest.raises(RemoteExecutionUnavailableError) as excinfo:
        RemoteWorkerExecutor(target=object(), actor=None, port=object())  # type: ignore[arg-type]
    assert "tunnel-manager" in str(excinfo.value)
    # The refusal must name its cause (H-12), not swallow it.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)
