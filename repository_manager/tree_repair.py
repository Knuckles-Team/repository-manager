"""Detect and repair index collapse and ``core.bare`` drift in a worktree.

The repairs are deliberately mechanical and narrow:

* ``probable-index-wipe`` -> ``git read-tree HEAD`` (the working directory is
  never reset, checked out, or otherwise overwritten); and
* ``core-bare-drift`` -> ``git config core.bare false``.

Each diagnosis compares the index to a per-worktree baseline and each repair
checks a content digest before and after the operation.  A baseline is kept in
the worktree's git administrative directory, not in the working tree, so a
doctor run cannot create an untracked file that later becomes a commit.

CONCEPT:RM-TREE-REPAIR (C-12)
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import subprocess  # nosec B404 - fixed argv git repair commands
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from repository_manager import stash_guard

__all__ = ["diagnose", "record_baseline", "repair"]

_BASELINE_FILE = "repository-manager-tree-baseline.json"
_INDEX_COLLAPSE_COUNT = 12
_INDEX_COLLAPSE_RATIO = 0.10


def _run(
    argv: list[str], path: Path, *, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=str(path),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(argv, 1, "", f"{type(exc).__name__}: {exc}")


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip()


def _git_dir(path: Path) -> Path | None:
    result = _run(["git", "rev-parse", "--git-dir"], path)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (path / git_dir).resolve()
    return git_dir


@contextlib.contextmanager
def _repair_lease(path: Path) -> Iterator[None]:
    """Serialize repair with all same-worktree repository-manager mutations."""
    with stash_guard.hold_tree_mutation_lease(path, note="tree repair"):
        yield


def _head_paths(path: Path) -> list[str]:
    result = _run(["git", "ls-tree", "-r", "--name-only", "-z", "HEAD"], path)
    if result.returncode != 0:
        return []
    return [entry for entry in result.stdout.split("\0") if entry]


def _index_paths(path: Path) -> tuple[list[str], str | None]:
    result = _run(["git", "ls-files", "-z"], path)
    if result.returncode != 0:
        return [], _detail(result) or "git ls-files failed"
    return [entry for entry in result.stdout.split("\0") if entry], None


def _content_digest(path: Path, names: list[str]) -> str:
    """Hash every expected tracked path and its bytes, including missing files."""
    digest = hashlib.sha256()
    for name in sorted(names):
        relative = name.encode("utf-8", "surrogateescape")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        target = path / name
        try:
            if target.is_symlink():
                data = os.readlink(target).encode("utf-8", "surrogateescape")
                digest.update(b"L" + len(data).to_bytes(8, "big") + data)
            elif target.is_file():
                data = target.read_bytes()
                digest.update(b"F" + len(data).to_bytes(8, "big") + data)
            else:
                digest.update(b"MISSING")
        except OSError as exc:
            data = f"ERROR:{type(exc).__name__}:{exc}".encode()
            digest.update(b"ERROR" + len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def _baseline_path(path: Path) -> Path | None:
    git_dir = _git_dir(path)
    return git_dir / _BASELINE_FILE if git_dir else None


def _load_baseline(path: Path) -> dict[str, Any] | None:
    target = _baseline_path(path)
    if target is None or not target.is_file():
        return None
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _save_baseline(path: Path, names: list[str], digest: str) -> dict[str, Any]:
    target = _baseline_path(path)
    if target is None:
        return {
            "tracked_count": len(names),
            "content_digest": digest,
            "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
            "persisted": False,
            "persistence_error": "git administrative directory unavailable",
        }
    payload = {
        "tracked_count": len(names),
        "content_digest": digest,
        "head_sha": (_run(["git", "rev-parse", "HEAD"], path).stdout.strip()),
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    try:
        target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        # Diagnosis remains useful when the administrative directory is
        # read-only, but callers must be able to report that the baseline was
        # not durable (safe_commit treats this as a post-commit warning).
        return {
            **payload,
            "persisted": False,
            "persistence_error": f"{type(exc).__name__}: {exc}",
        }
    return {**payload, "persisted": True}


def record_baseline(path: Path | str) -> dict[str, Any]:
    """Record the current HEAD tracked set as this worktree's good baseline."""
    tree = Path(path).expanduser().resolve()
    names = _head_paths(tree)
    if not names and _run(["git", "rev-parse", "HEAD"], tree).returncode != 0:
        return {
            "ok": False,
            "finding": "unavailable",
            "error": "cannot resolve HEAD to record a baseline",
        }
    digest = _content_digest(tree, names)
    payload = _save_baseline(tree, names, digest)
    return {"ok": True, "finding": "clean", "path": str(tree), **payload}


def _core_bare_drift(path: Path) -> dict[str, Any] | None:
    """Return evidence when a checkout with a ``.git`` entry is marked bare."""
    git_entry = path / ".git"
    if not (git_entry.is_dir() or git_entry.is_file()):
        return None
    result = _run(["git", "config", "--bool", "--get", "core.bare"], path)
    value = result.stdout.strip().lower() if result.returncode == 0 else ""
    if value != "true":
        return None
    return {
        "core_bare": value,
        "expected_core_bare": False,
        "config": str(git_entry / "config") if git_entry.is_dir() else str(git_entry),
    }


def diagnose(path: Path | str) -> dict[str, Any]:
    """Diagnose one tree as ``probable-index-wipe``, ``core-bare-drift``, or clean."""
    tree = Path(path).expanduser().resolve()
    evidence: dict[str, Any] = {"path": str(tree)}
    findings: list[str] = []
    bare = _core_bare_drift(tree)
    if bare is not None:
        findings.append("core-bare-drift")
        evidence["core_bare"] = bare

    baseline = _load_baseline(tree)
    head_names = _head_paths(tree)
    if baseline is None and head_names:
        baseline = _save_baseline(tree, head_names, _content_digest(tree, head_names))
    indexed, index_error = _index_paths(tree)
    expected_count = int((baseline or {}).get("tracked_count", len(head_names)))
    actual_count = len(indexed)
    evidence.update(
        {
            "baseline_count": expected_count,
            "indexed_count": actual_count,
            "index_error": index_error,
            "baseline": baseline,
        }
    )
    collapsed = (
        expected_count >= _INDEX_COLLAPSE_COUNT and actual_count < _INDEX_COLLAPSE_COUNT
    ) or (
        expected_count >= 100 and actual_count < expected_count * _INDEX_COLLAPSE_RATIO
    )
    if collapsed:
        findings.append("probable-index-wipe")

    finding = findings[0] if findings else "clean"
    if head_names:
        evidence["tracked_content_digest"] = _content_digest(tree, head_names)
    return {
        "ok": not findings,
        "path": str(tree),
        "finding": finding,
        "findings": findings or ["clean"],
        "evidence": evidence,
        "baseline": baseline,
    }


def _repair_locked(
    path: Path | str, *, finding: str | dict[str, Any]
) -> dict[str, Any]:
    """Repair one freshly confirmed structural finding and prove preservation.

    ``finding`` is a request, not authority.  Diagnosis is repeated while the
    worktree mutation lease is held, so a stale or forged caller-supplied
    finding cannot turn a healthy index into ``HEAD`` (or flip unrelated
    configuration).  The same lease spans the complete before/validate/repair/
    after sequence; managed tree mutators therefore cannot interleave between
    the checks and the repair.
    """
    tree = Path(path).expanduser().resolve()
    requested = finding.get("finding") if isinstance(finding, dict) else finding
    if requested not in {"probable-index-wipe", "core-bare-drift", "clean"}:
        return {"ok": False, "finding": requested, "error": "unknown repair finding"}

    with _repair_lease(tree):
        before_names = _head_paths(tree)
        before_digest = _content_digest(tree, before_names)
        trusted_baseline = _load_baseline(tree)
        current = diagnose(tree)
        current_findings = set(current.get("findings", []))
        if requested not in current_findings:
            return {
                "ok": False,
                "status": "refused",
                "finding": requested,
                "actual_finding": current.get("finding", "unavailable"),
                "path": str(tree),
                "error": (
                    "requested repair finding is stale or mismatched; "
                    "no mutation performed"
                ),
                "checksum_before": before_digest,
                "checksum_after": before_digest,
                "content_preserved": True,
                "post_diagnosis": current,
            }
        if requested == "probable-index-wipe":
            trusted_digest = (
                trusted_baseline.get("content_digest")
                if trusted_baseline is not None
                else None
            )
            if not isinstance(trusted_digest, str) or before_digest != trusted_digest:
                return {
                    "ok": False,
                    "status": "refused",
                    "finding": requested,
                    "actual_finding": current.get("finding", "unavailable"),
                    "path": str(tree),
                    "error": (
                        "tracked content does not match the last trusted baseline; "
                        "no index repair performed"
                    ),
                    "checksum_before": before_digest,
                    "checksum_after": before_digest,
                    "content_preserved": True,
                    "post_diagnosis": current,
                }
        if requested == "clean":
            return {
                "ok": True,
                "status": "noop",
                "finding": "clean",
                "path": str(tree),
                "checksum_before": before_digest,
                "checksum_after": before_digest,
                "content_preserved": True,
                "post_diagnosis": current,
            }

        command = (
            ["git", "read-tree", "HEAD"]
            if requested == "probable-index-wipe"
            else ["git", "config", "core.bare", "false"]
        )
        result = _run(command, tree)
        if result.returncode != 0:
            return {
                "ok": False,
                "status": "error",
                "finding": requested,
                "path": str(tree),
                "error": _detail(result) or "repair command failed",
                "checksum_before": before_digest,
            }

        after_names = _head_paths(tree)
        after_digest = _content_digest(tree, after_names)
        preserved = before_names == after_names and before_digest == after_digest
        if not preserved:
            return {
                "ok": False,
                "status": "error",
                "finding": requested,
                "path": str(tree),
                "error": "repair changed tracked working-tree content",
                "checksum_before": before_digest,
                "checksum_after": after_digest,
                "content_preserved": False,
            }

        baseline = _save_baseline(tree, after_names, after_digest)
        post_diagnosis = diagnose(tree)
        if requested in set(post_diagnosis.get("findings", [])):
            return {
                "ok": False,
                "status": "error",
                "finding": requested,
                "path": str(tree),
                "error": "repair completed but the requested finding remains",
                "checksum_before": before_digest,
                "checksum_after": after_digest,
                "content_preserved": True,
                "baseline": baseline,
                "post_diagnosis": post_diagnosis,
            }
        return {
            "ok": True,
            "status": "repaired",
            "finding": requested,
            "path": str(tree),
            "checksum_before": before_digest,
            "checksum_after": after_digest,
            "content_preserved": True,
            "baseline": baseline,
            "post_diagnosis": post_diagnosis,
        }


def repair(path: Path | str, *, finding: str | dict[str, Any]) -> dict[str, Any]:
    """Re-diagnose and repair a finding, returning a structured lease refusal."""
    tree = Path(path).expanduser().resolve()
    requested = finding.get("finding") if isinstance(finding, dict) else finding
    try:
        return _repair_locked(tree, finding=finding)
    except stash_guard.BlockedByLease as exc:
        return {
            "ok": False,
            "status": "refused",
            "reason": "tree-mutation-busy",
            "finding": requested,
            "path": str(tree),
            "error": str(exc),
        }
    except (OSError, RuntimeError) as exc:
        return {
            "ok": False,
            "status": "error",
            "reason": "tree-mutation-lease-unavailable",
            "finding": requested,
            "path": str(tree),
            "error": str(exc),
        }
