"""Non-native bootstrap must reject unverified or mixed graph client sources."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

from tests.conftest import isolated_git_subprocess_env

BOOTSTRAP = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/bootstrap_graph_client.py")
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    contract = root / "epistemic_graph/contract"
    contract.mkdir(parents=True)
    client = root / "epistemic_graph/client.py"
    client.write_text("CLIENT = 'verified'\n")
    receipt = {
        "contract_digest": "receipt-for-fixture",
        "artifact_digests": {
            "epistemic_graph/client.py": hashlib.sha256(client.read_bytes()).hexdigest()
        },
    }
    (contract / "receipt.json").write_text(json.dumps(receipt))
    (contract / "__init__.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def verify_receipt(expected_digest):\n"
        "    actual = json.loads(Path(__file__).with_name('receipt.json').read_text())\n"
        "    if actual['contract_digest'] != expected_digest:\n"
        "        raise ValueError('upstream receipt mismatch')\n"
    )
    env = isolated_git_subprocess_env()
    commands = [
        ["init", "-q", "-b", "fixture"],
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["add", "epistemic_graph"],
        ["commit", "-q", "-m", "fixture"],
    ]
    for command in commands:
        subprocess.run(["git", *command], cwd=root, env=env, check=True)
    revision = BOOTSTRAP["git"](root, "rev-parse", "HEAD")
    return root, revision


def test_exact_source_and_upstream_receipt_pass(source):
    root, revision = source
    assert BOOTSTRAP["verify_source"](root, revision) == "receipt-for-fixture"


def test_different_pin_is_rejected(source):
    root, _ = source
    with pytest.raises(ValueError, match="differs"):
        BOOTSTRAP["verify_source"](root, "0" * 40)


@pytest.mark.parametrize("kind", ["modified", "untracked", "deleted"])
def test_changed_client_is_rejected(source, kind):
    root, revision = source
    client = root / "epistemic_graph/client.py"
    if kind == "deleted":
        client.unlink()
    elif kind == "untracked":
        (root / "epistemic_graph/injected.py").write_text("INJECTED = True\n")
    else:
        client.write_text("CLIENT = 'modified'\n")
    with pytest.raises(ValueError, match="modified or untracked"):
        BOOTSTRAP["verify_source"](root, revision)


def test_failed_verification_removes_previous_overlay(source, tmp_path):
    root, _ = source
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/siblings.lock").write_text(
        f"epistemic-graph unused {'0' * 40} {root.name}\n"
    )
    overlay = tmp_path / "repository_manager_graph_client.pth"
    overlay.write_text("/previous/source\n")
    with pytest.raises(ValueError):
        BOOTSTRAP["configure"](tmp_path, tmp_path, native=False)
    assert not overlay.exists()


def test_native_mode_removes_source_overlay(tmp_path):
    overlay = tmp_path / "repository_manager_graph_client.pth"
    overlay.write_text("/previous/source\n")
    BOOTSTRAP["configure"](tmp_path, tmp_path, native=True)
    assert not overlay.exists()
