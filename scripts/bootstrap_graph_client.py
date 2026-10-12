#!/usr/bin/env python3
"""Expose the pinned non-native client only after upstream contract verification."""

from __future__ import annotations

import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, distribution
import json
import subprocess
import sys
import sysconfig
from pathlib import Path


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def pinned_source(root: Path) -> tuple[Path, str]:
    entries = [
        line.split()
        for line in (root / "scripts/siblings.lock").read_text().splitlines()
        if line.startswith("epistemic-graph ")
    ]
    if len(entries) != 1 or len(entries[0]) != 4:
        raise ValueError("expected one pinned epistemic-graph source")
    return (root / entries[0][3]).resolve(), entries[0][2]


def verify_source(source: Path, revision: str) -> str:
    """Bind the receipt, generated artifacts and client source to one Git pin."""
    if git(source, "rev-parse", "HEAD") != revision:
        raise ValueError("graph client checkout differs from scripts/siblings.lock")
    dirty = git(
        source, "status", "--porcelain", "--untracked-files=all", "--",
        "epistemic_graph", "contract",
    )
    if dirty:
        raise ValueError("graph client/contract source is modified or untracked")
    receipt_path = "epistemic_graph/contract/receipt.json"
    receipt = json.loads(git(source, "show", f"{revision}:{receipt_path}"))
    for relative, expected in receipt["artifact_digests"].items():
        artifact = (source / relative).resolve()
        if not artifact.is_relative_to(source):
            raise ValueError("contract artifact escapes the pinned source")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"upstream contract artifact mismatch: {relative}")
    verify_upstream_receipt(source, receipt["contract_digest"])
    return receipt["contract_digest"]


def verify_upstream_receipt(source: Path, expected_digest: str) -> None:
    """Use the shipped RF-RULING-003 verifier without importing the client first."""
    path = source / "epistemic_graph/contract/__init__.py"
    spec = importlib.util.spec_from_file_location("_bootstrap_graph_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError("upstream contract verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_receipt(expected_digest)


def configure(root: Path, purelib: Path, *, native: bool) -> None:
    overlay = purelib / "repository_manager_graph_client.pth"
    # A failed check must never leave a previously accepted source path active.
    overlay.unlink(missing_ok=True)
    if native:
        return
    try:
        distribution("epistemic-graph")
    except PackageNotFoundError:
        pass
    else:
        raise ValueError("non-native bootstrap refuses to mix installed engine and source")
    source, revision = pinned_source(root)
    digest = verify_source(source, revision)
    overlay.write_text(str(source) + "\n")
    print(f"bootstrap: graph client {revision}; verified contract {digest}")


if __name__ == "__main__":
    configure(
        Path(__file__).resolve().parents[1],
        Path(sysconfig.get_path("purelib")),
        native=sys.argv[1:] == ["--native"],
    )
