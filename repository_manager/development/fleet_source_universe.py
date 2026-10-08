"""Independent, fail-closed Git-tree proof of the fleet census source universe.

The manifest cannot attest to its own completeness. Each repository's committed
tree is enumerated independently and every source-capable file must be named by
the manifest. Unsupported source languages and metadata formats refuse proof.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fleet_manifest import FleetManifestClosure, FleetRepository

MAX_TREE_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_TREE_ENTRIES = 100_000
MAX_REPOSITORIES = 2048
_COMPOSE_NAMES = {
    "compose.yml",
    "compose.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_UNSUPPORTED_NAMES = {
    "Dockerfile",
    "go.mod",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
}
_UNSUPPORTED_SUFFIXES = {
    ".go",
    ".pyi",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".vue",
    ".svelte",
}
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class RepositoryTreeIdentity:
    repository_id: str
    path: str
    commit: str
    source_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceUniverseReceipt:
    schema: str
    repositories: tuple[RepositoryTreeIdentity, ...]
    source_paths: tuple[str, ...]
    digest: str


def _git(repository: Path, *arguments: str) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise ValueError("fleet Git tree inventory is unavailable")
    try:
        result = subprocess.run(
            [
                str(Path(executable).resolve(strict=True)),
                "-C",
                str(repository),
                *arguments,
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("fleet Git tree inventory is unavailable") from exc
    if result.returncode or len(result.stdout) > MAX_TREE_OUTPUT_BYTES:
        raise ValueError("fleet Git tree inventory failed or exceeded bound")
    return result.stdout


def _source_kind(path: str) -> str | None:
    basename = PurePosixPath(path).name
    if basename == "pyproject.toml":
        return "python_project"
    if basename == "package.json":
        return "frontend_package"
    if basename == "Cargo.toml":
        return "rust_manifest"
    if basename in _COMPOSE_NAMES:
        return "compose"
    if path.endswith(".py"):
        return "python_source"
    if path.endswith(".rs"):
        return "rust_source"
    if PurePosixPath(path).suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return "frontend_source"
    if (
        basename in _UNSUPPORTED_NAMES
        or PurePosixPath(path).suffix in _UNSUPPORTED_SUFFIXES
    ):
        raise ValueError(f"fleet source extractor does not support {path}")
    return None


def _tree_record_path(record: bytes) -> str:
    try:
        meta, name = record.split(b"\t", 1)
        mode, object_type, _object_id = meta.split(b" ", 2)
        path = name.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("fleet Git tree record is invalid") from exc
    if mode not in {b"100644", b"100755"} or object_type != b"blob":
        raise ValueError("fleet Git tree contains unsupported entry")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
        raise ValueError("fleet Git tree has unsafe path")
    return path


def _tree_paths(repository: Path) -> tuple[str, ...]:
    raw = _git(repository, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    records = raw.split(b"\0")
    if records[-1] != b"" or len(records) - 1 > MAX_TREE_ENTRIES:
        raise ValueError("fleet Git tree record count or framing is invalid")
    paths = (_tree_record_path(record) for record in records[:-1])
    return tuple(sorted(path for path in paths if _source_kind(path) is not None))


def _repository_tree_identity(
    root: Path, root_resolved: Path, entry: FleetRepository
) -> RepositoryTreeIdentity:
    repository = root / entry.path
    if repository.is_symlink() or not repository.resolve().is_relative_to(
        root_resolved
    ):
        raise ValueError("fleet repository escapes workspace")
    top_level = _git(repository, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top_level).resolve() != repository.resolve():
        raise ValueError("fleet repository path is not its Git root")
    commit = _git(repository, "rev-parse", "HEAD").decode().strip()
    if not _REVISION.fullmatch(commit):
        raise ValueError("fleet repository revision is invalid")
    if _git(repository, "status", "--porcelain", "-z", "--untracked-files=all"):
        raise ValueError("fleet repository tree is dirty")
    absolute = tuple(f"{entry.path}/{path}" for path in _tree_paths(repository))
    return RepositoryTreeIdentity(entry.repository_id, entry.path, commit, absolute)


def capture_source_universe(
    root: Path, closure: FleetManifestClosure
) -> SourceUniverseReceipt:
    """Prove declared source equality against clean, committed repository trees.

    This checks every tracked source-capable path, including tests and tools;
    unexplained files are refused instead of silently classified as production.
    A release checkout must be clean so working bytes match the inventoried tree.
    """

    if len(closure.repositories) > MAX_REPOSITORIES:
        raise ValueError("fleet repository universe exceeds bound")
    rows: list[RepositoryTreeIdentity] = []
    discovered: list[str] = []
    root_resolved = root.resolve()
    for entry in closure.repositories:
        identity = _repository_tree_identity(root, root_resolved, entry)
        rows.append(identity)
        discovered.extend(identity.source_paths)
    declared = tuple(sorted(source.path for source in closure.sources))
    observed = tuple(sorted(discovered))
    if observed != declared:
        raise ValueError("fleet declared source set differs from Git tree universe")
    payload = {
        "schema": "fleet_source_universe/v1",
        "repositories": [
            (row.repository_id, row.path, row.commit, row.source_paths) for row in rows
        ],
        "source_paths": observed,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceUniverseReceipt(
        "fleet_source_universe/v1", tuple(rows), observed, digest
    )


def verify_source_universe(
    root: Path, closure: FleetManifestClosure, receipt: SourceUniverseReceipt
) -> bool:
    """Re-enumerate Git trees rather than trusting a supplied receipt or digest."""

    if type(receipt) is not SourceUniverseReceipt:
        return False
    try:
        return capture_source_universe(root, closure) == receipt
    except (OSError, ValueError, TypeError, AttributeError):
        return False
