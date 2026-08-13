"""``capacity_seed`` — adapting tunnel-manager's SSH inventory into
placeholder ``HostCapacity`` seeds, without coupling to tunnel-manager
itself (CONCEPT:RM-CAPACITY-SEED)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from repository_manager.capacity import CapacityInventory, HostCapacity, ResourceVector
from repository_manager.capacity_seed import parse_inventory_hosts, seed_capacity

_SAMPLE_INVENTORY = textwrap.dedent(
    """
    all:
      children:
        homelab:
          hosts:
            gb10:
              ansible_host: 10.0.0.18
              role: gpu-inference-llm
            r510:
              ansible_host: 10.0.0.10
            r710:
              ansible_host: 10.0.0.11
            r820:
              ansible_host: 10.0.0.13
          vars:
            ansible_user: genius
    """
)


def test_parses_hosts_from_the_real_inventory_shape(tmp_path: Path) -> None:
    inventory_file = tmp_path / "inventory.yaml"
    inventory_file.write_text(_SAMPLE_INVENTORY)
    hosts = parse_inventory_hosts(inventory_file)
    assert {h.alias for h in hosts} == {"gb10", "r510", "r710", "r820"}
    gb10 = next(h for h in hosts if h.alias == "gb10")
    assert gb10.role == "gpu-inference-llm"
    assert gb10.group == "homelab"


def test_a_missing_inventory_file_seeds_nothing_not_an_error(tmp_path: Path) -> None:
    assert parse_inventory_hosts(tmp_path / "does-not-exist.yaml") == ()


def test_seed_capacity_registers_every_unregistered_host(tmp_path: Path) -> None:
    inventory_file = tmp_path / "inventory.yaml"
    inventory_file.write_text(_SAMPLE_INVENTORY)
    capacity = CapacityInventory()

    result = seed_capacity(capacity, path=inventory_file)

    assert sorted(result.seeded) == ["gb10", "r510", "r710", "r820"]
    assert result.already_registered == ()
    assert result.inventory_host_count == 4
    for alias in result.seeded:
        assert capacity.get(alias) is not None


def test_seed_never_overwrites_an_already_registered_host(tmp_path: Path) -> None:
    inventory_file = tmp_path / "inventory.yaml"
    inventory_file.write_text(_SAMPLE_INVENTORY)
    capacity = CapacityInventory()
    real_measured = HostCapacity(
        host_id="r820",
        total=ResourceVector(cpu_weight=64, memory_mib=524288, disk_mib=8_000_000),
        target_kind="inventory_alias",
        version=1,
    )
    capacity.register(real_measured)

    result = seed_capacity(capacity, path=inventory_file)

    assert "r820" in result.already_registered
    assert "r820" not in result.seeded
    # The real measured capacity survived untouched.
    assert capacity.require("r820").total.cpu_weight == 64


def test_a_seeded_host_is_deliberately_stale_and_cannot_admit_real_work(
    tmp_path: Path,
) -> None:
    """A seed carries no verified measurement — it must not silently pass
    for a real, live, admittable host. This is the "never fake a live
    observation" half of the invariant.
    """

    inventory_file = tmp_path / "inventory.yaml"
    inventory_file.write_text(_SAMPLE_INVENTORY)
    capacity = CapacityInventory()
    seed_capacity(capacity, path=inventory_file)

    fits, reason = capacity.can_fit(
        "r820", ResourceVector(cpu_weight=1, memory_mib=256, process_slots=1)
    )
    assert fits is False
    assert "stale" in reason
