"""P0.7 — persisting the capacity ledger through ``remote_worker_actions``.

Proves the actual bad state is now impossible: ``register_worker`` used to
write only to a process-local in-memory singleton, so a restarted MCP server
forgot every registered host with no signal anything was lost. Here a
"restart" is simulated the only honest way from inside one test process —
resetting the module's own singleton globals to `None`, exactly what
happens to real module globals when the interpreter that owns the process
exits and a new one starts — and a NEW `capacity_inventory()` call still
sees the host, because it rehydrates from the durable store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from repository_manager import remote_worker_actions as rwa


@pytest.fixture(autouse=True)
def _isolated_capacity_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Give every test in this module its OWN durable store, and reset the
    process-local singletons before and after — real isolation, not a shared
    ``~/.local/state`` file leaking between test runs or module import order.
    """

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None
    yield
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None


def _register(host_id: str = "r820") -> dict:
    return rwa.dispatch(
        "register_worker",
        host_id=host_id,
        cpu_weight=32,
        memory_mib=131072,
        disk_mib=2_000_000,
        process_slots=8,
        labels=["gpu"],
        inventory_alias=host_id,
        repository_roots={"epistemic-graph": "/srv/build/epistemic-graph"},
        toolchains=["rust-1.95"],
    )


def _simulate_restart() -> None:
    """Drop every process-local singleton — a fresh process would have none."""

    rwa._CAPACITY_INVENTORY = None
    rwa._REGISTRY = None
    # `_CAPACITY_STORE` deliberately survives: it is a HANDLE to a durable
    # file, not part of the in-memory ledger being tested. A real restart
    # opens a new handle to the same path, which `_capacity_store()` already
    # does lazily the next time it is called after `_CAPACITY_STORE = None`.
    rwa._CAPACITY_STORE = None


def test_a_registered_host_survives_a_simulated_restart() -> None:
    result = _register()
    assert result["ok"] is True
    assert result["capacity_registered"] is True

    before_restart = rwa.capacity_inventory().get("r820")
    assert before_restart is not None

    _simulate_restart()

    after_restart = rwa.capacity_inventory().get("r820")
    assert after_restart is not None
    assert after_restart.host_id == "r820"
    assert after_restart.total.cpu_weight == 32
    assert after_restart.total.memory_mib == 131072


def test_the_worker_profile_still_resolves_after_restart_via_a_fresh_registration() -> (
    None
):
    """The `RemoteWorkerRegistry` profile itself (roots/toolchains) is a
    SEPARATE, still-intentionally-in-memory-only structure
    (`remote_worker_actions.py`'s own docstring: never a second job ledger).
    Only capacity is durable today; this documents that boundary rather than
    silently assuming profiles persist too — `profile()` on a fresh registry
    correctly refuses until re-registered, and re-registration succeeds
    because the durable capacity record it depends on is still there.
    """

    _register()
    _simulate_restart()

    with pytest.raises(rwa.RemoteWorkerRegistryError):
        rwa.remote_worker_registry().profile("r820")

    # Capacity survived, so re-registering the PROFILE (not capacity) is a
    # forward path an operator can take without redeclaring resource sizes.
    second = _register()
    assert second["ok"] is True
    assert rwa.remote_worker_registry().profile("r820").host_id == "r820"


def test_seed_from_inventory_action_persists_seeded_hosts_through_a_restart(
    tmp_path: Path,
) -> None:
    import textwrap

    inventory_file = tmp_path / "inventory.yaml"
    inventory_file.write_text(
        textwrap.dedent(
            """
            all:
              children:
                homelab:
                  hosts:
                    r820:
                      ansible_host: 10.0.0.13
            """
        )
    )

    result = rwa.dispatch("seed_from_inventory", path=str(inventory_file))
    assert result["ok"] is True
    assert result["seeded"] == ["r820"]

    _simulate_restart()

    rehydrated = rwa.capacity_inventory().require("r820")
    assert rehydrated.host_id == "r820"
    assert "seed" in rehydrated.labels


def test_a_second_process_style_instance_sees_the_same_host() -> None:
    """Two independent `capacity_inventory()` resolutions, with the module
    singleton reset between them (standing in for two separate processes
    against the same durable store), converge on the same accounting.
    """

    _register("r710")
    first = rwa.capacity_inventory().require("r710")

    _simulate_restart()

    second = rwa.capacity_inventory().require("r710")
    assert first.host_id == second.host_id
    assert first.total.as_dict() == second.total.as_dict()
