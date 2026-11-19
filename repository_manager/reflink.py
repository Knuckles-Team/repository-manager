"""Copy-on-write (reflink) worktree population.

CONCEPT:RM-WORKTREE-REFLINK — on a filesystem that supports reflinks a new
linked worktree can be populated from an existing worktree of the same
repository at near-zero cost, then reconciled against the target commit so only
the files that differ are rewritten. This module holds the pieces that do not
depend on :class:`~repository_manager.worktree.WorktreeManager`: the capability
probe, source selection, the reflink copy and the artifact seeding. Every
outcome is reported explicitly; callers fall back to a plain checkout and say so.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

try:
    import fcntl
except ImportError:  # Windows has no ioctl; the probe then answers "no".
    fcntl = None  # type: ignore[assignment]

DEFAULT_SEED_ARTIFACTS = ("target", ".venv", "node_modules")
SEED_ARTIFACTS_ENV = "REPOSITORY_MANAGER_SEED_ARTIFACTS"
REFLINK_MODES = ("auto", "never")

# ``FICLONE`` from <linux/fs.h>: _IOW(0x94, 9, int).
_FICLONE = 0x40049409

_PROBE_CACHE: dict[int, ReflinkSupport] = {}
_PROBE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReflinkSupport:
    """Whether the filesystem of a directory can reflink, and why."""

    supported: bool
    reason: str


def seed_artifact_names() -> tuple[str, ...]:
    """The configured untracked artifact directories to seed."""
    raw = os.environ.get(SEED_ARTIFACTS_ENV, "")
    names = tuple(n.strip().strip("/") for n in raw.split(",") if n.strip())
    return names or DEFAULT_SEED_ARTIFACTS


def _clone_linux(src: str, dst: str) -> None:
    if fcntl is None:
        raise OSError("FICLONE is not available on this platform")
    with open(src, "rb") as s, open(dst, "wb") as d:
        fcntl.ioctl(d.fileno(), _FICLONE, s.fileno())


def _clone_macos(src: str, dst: str) -> None:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), dst)


def reflink_file(src: str, dst: str) -> None:
    """Reflink ``src`` to a new file ``dst`` (raises ``OSError`` if unsupported).

    A partially created ``dst`` is removed on failure. File permissions are
    copied; this is the one copy primitive the rest of the module uses.
    """
    if sys.platform.startswith("linux"):
        clone = _clone_linux
    elif sys.platform == "darwin":
        clone = _clone_macos
    else:
        raise OSError(f"reflink is not available on {sys.platform}")
    try:
        clone(src, dst)
        shutil.copymode(src, dst)
    except (OSError, AttributeError):
        with contextlib.suppress(OSError):
            os.unlink(dst)
        raise


def _probe_directory(directory: str, clone: Callable[[str, str], None]) -> bool:
    src = os.path.join(directory, f".rm-reflink-probe-{os.getpid()}-a")
    dst = os.path.join(directory, f".rm-reflink-probe-{os.getpid()}-b")
    try:
        with open(src, "wb") as f:
            f.write(b"reflink-probe\n" * 512)
        clone(src, dst)
        return True
    except (OSError, AttributeError, TypeError):
        return False
    finally:
        for p in (src, dst):
            with contextlib.suppress(OSError):
                os.unlink(p)


def probe_reflink(
    directory: str, clone: Callable[[str, str], None] | None = None
) -> ReflinkSupport:
    """Probe whether ``directory``'s filesystem supports reflinks.

    Two temporary files are created in ``directory`` and cloned (``FICLONE`` on
    Linux, ``clonefile`` on macOS). The answer is cached per filesystem device;
    any error means "no", never an exception.
    """
    try:
        device = os.stat(directory).st_dev
    except OSError as exc:
        return ReflinkSupport(False, f"cannot stat {directory}: {exc.strerror}")
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(device)
    if cached is not None:
        return cached
    ok = _probe_directory(directory, clone or reflink_file)
    result = ReflinkSupport(
        ok,
        "filesystem supports reflinks"
        if ok
        else "filesystem does not support reflinks",
    )
    with _PROBE_LOCK:
        _PROBE_CACHE[device] = result
    return result


def clear_probe_cache() -> None:
    """Forget cached probe answers (tests and long-lived servers)."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


# ── git helpers ──────────────────────────────────────────────────────────
def _git_executable() -> str:
    return shutil.which("git") or "git"


def git_out(args: list[str], cwd: str) -> str | None:
    """Run ``git`` and return stdout, or ``None`` on any failure."""
    try:
        proc = subprocess.run(
            [_git_executable(), *args],
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        proc.stdout.decode("utf-8", "surrogateescape") if proc.returncode == 0 else None
    )


def _nul_list(out: str | None) -> list[str]:
    return [p for p in (out or "").split("\0") if p]


def tracked_files(tree: str) -> list[str] | None:
    """Paths of the gitlink-free tracked files of ``tree`` (``None`` on error)."""
    out = git_out(["ls-files", "-z", "--stage"], tree)
    if out is None:
        return None
    paths: list[str] = []
    for rec in out.split("\0"):
        meta, _, path = rec.partition("\t")
        if path and not meta.startswith("160000"):
            paths.append(path)
    return paths


# ── source selection ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class SourceCandidate:
    """An existing worktree that could seed a new one."""

    path: str
    head: str


def _distance(candidate: SourceCandidate, target: str, tree: str) -> int | None:
    if candidate.head == target:
        return 0
    out = git_out(["diff", "--name-only", "-z", candidate.head, target], tree)
    return None if out is None else len(_nul_list(out))


def _same_device(path: str, device: int) -> bool:
    try:
        return os.stat(path).st_dev == device
    except OSError:
        return False


def select_source(
    candidates: Iterable[SourceCandidate],
    target: str,
    *,
    device: int,
    tree: str,
    distance: Callable[[SourceCandidate, str, str], int | None] = _distance,
) -> SourceCandidate | None:
    """Pick the same-device worktree whose HEAD is closest to ``target``.

    Closeness is the number of files that differ between the candidate's HEAD
    and the target commit (those are the files that must be rewritten); ties
    break on path so the choice is deterministic.
    """
    best: tuple[int, str, SourceCandidate] | None = None
    for cand in candidates:
        if not os.path.isdir(cand.path) or not _same_device(cand.path, device):
            continue
        dist = distance(cand, target, tree)
        if dist is None:
            continue
        key = (dist, cand.path)
        if best is None or key < best[:2]:
            best = (dist, cand.path, cand)
    return best[2] if best else None


# ── tree copy ────────────────────────────────────────────────────────────
def _copy_entry(src: str, dst: str, copy: Callable[[str, str], None]) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        copy(src, dst)


def copy_tracked(
    source: str,
    dest: str,
    paths: Iterable[str],
    copy: Callable[[str, str], None] | None = None,
) -> int:
    """Reflink-copy ``paths`` (relative, tracked) from ``source`` into ``dest``.

    Files missing in the source are skipped; the caller reconciles them with
    ``git checkout-index``. Returns the number of entries copied.
    """
    clone = copy or reflink_file
    n = 0
    for rel in paths:
        src = os.path.join(source, rel)
        if not os.path.lexists(src) or os.path.isdir(src) and not os.path.islink(src):
            continue
        _copy_entry(src, os.path.join(dest, rel), clone)
        n += 1
    return n


def _remove_with_empty_parents(tree: str, rel: str) -> None:
    full = os.path.join(tree, rel)
    with contextlib.suppress(OSError):
        os.unlink(full)
    parent = os.path.dirname(full)
    while parent != tree and parent.startswith(tree):
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def reconcile(tree: str, source_paths: Iterable[str]) -> str | None:
    """Make ``tree`` byte-identical to its HEAD after a tracked-file copy.

    ``git read-tree`` loads the target index; files whose content differs from
    the target (or are missing) are rewritten with ``checkout-index``, copied
    files the target does not track are removed. Returns an error string, or
    ``None`` when ``git status`` is clean.
    """
    if git_out(["read-tree", "HEAD"], tree) is None:
        return "git read-tree failed"
    git_out(["update-index", "-q", "--really-refresh"], tree)
    stale = _nul_list(git_out(["diff-files", "--name-only", "-z"], tree))
    if stale:
        proc = subprocess.run(
            [_git_executable(), "checkout-index", "-f", "-z", "--stdin"],
            cwd=tree,
            input="\0".join(stale).encode() + b"\0",
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return "git checkout-index failed"
    wanted = set(tracked_files(tree) or ())
    for rel in set(source_paths) - wanted:
        _remove_with_empty_parents(tree, rel)
    git_out(["update-index", "-q", "--refresh"], tree)
    status = git_out(["status", "--porcelain"], tree)
    return None if status == "" else "worktree not clean after reflink checkout"


def reset_to_plain(tree: str) -> bool:
    """Discard a partial reflink checkout and populate ``tree`` normally."""
    for name in os.listdir(tree):
        if name == ".git":
            continue
        full = os.path.join(tree, name)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                os.unlink(full)
    return git_out(["reset", "--hard", "-q", "HEAD"], tree) is not None


# ── artifact seeding ─────────────────────────────────────────────────────
def _is_tracked(tree: str, name: str) -> bool:
    return bool(git_out(["ls-files", "--", name], tree))


def _seed_one(
    source: str, dest: str, name: str, copy: Callable[[str, str], None]
) -> dict[str, str]:
    src, dst = os.path.join(source, name), os.path.join(dest, name)
    if not os.path.isdir(src) or os.path.islink(src):
        return {"status": "skipped", "reason": "absent in source worktree"}
    if os.path.lexists(dst) or _is_tracked(source, name):
        return {"status": "skipped", "reason": "tracked or already present"}
    try:
        for root, dirs, files in os.walk(src):
            out = os.path.join(dst, os.path.relpath(root, src))
            os.makedirs(out, exist_ok=True)
            for f in files:
                _copy_entry(os.path.join(root, f), os.path.join(out, f), copy)
            for d in dirs:
                if os.path.islink(os.path.join(root, d)):
                    _copy_entry(os.path.join(root, d), os.path.join(out, d), copy)
    except OSError as exc:
        shutil.rmtree(dst, ignore_errors=True)
        return {"status": "skipped", "reason": f"reflink copy failed: {exc}"}
    return {"status": "seeded", "reason": "reflinked from source worktree"}


def seed_artifacts(
    source: str | None,
    dest: str,
    support: ReflinkSupport,
    names: Iterable[str] | None = None,
    copy: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    """Reflink-seed untracked artifact directories; never a full copy.

    Returns an explicit report: ``status`` (``seeded``/``partial``/``skipped``),
    a ``reason`` and per-directory results.
    """
    if not support.supported:
        return {"status": "skipped", "reason": support.reason, "artifacts": {}}
    if source is None:
        return {
            "status": "skipped",
            "reason": "no source worktree on the same filesystem",
            "artifacts": {},
        }
    results = {
        n: _seed_one(source, dest, n, copy or reflink_file)
        for n in (names or seed_artifact_names())
    }
    seeded = [r for r in results.values() if r["status"] == "seeded"]
    status = (
        "skipped"
        if not seeded
        else "seeded"
        if len(seeded) == len(results)
        else "partial"
    )
    return {
        "status": status,
        "reason": f"source worktree {os.path.basename(source)}",
        "artifacts": results,
    }


# ── planning ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReflinkPlan:
    """What ``add`` will do for the checkout and the artifact seed."""

    support: ReflinkSupport | None
    source: SourceCandidate | None
    reason: str
    attempt: bool
    seed_source: SourceCandidate | None = None

    @property
    def seed_path(self) -> str | None:
        src = self.seed_source or self.source
        return src.path if src else None


def parse_candidates(porcelain: str) -> list[SourceCandidate]:
    """Parse ``git worktree list --porcelain`` into usable source candidates."""
    out: list[SourceCandidate] = []
    cur: dict[str, str] = {}
    for line in [*porcelain.splitlines(), ""]:
        if line:
            key, _, val = line.partition(" ")
            cur[key] = val
            continue
        if (
            cur.get("worktree")
            and cur.get("HEAD")
            and "bare" not in cur
            and "prunable" not in cur
        ):
            out.append(SourceCandidate(os.path.normpath(cur["worktree"]), cur["HEAD"]))
        cur = {}
    return out


def _target_commit(canonical: str, branch: str, base: str) -> str | None:
    for ref in (f"refs/heads/{branch}", base):
        out = git_out(
            ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], canonical
        )
        if out and out.strip():
            return out.strip()
    return None


def plan_reflink(
    canonical: str, wt_parent: str, branch: str, base: str, *, mode: str, seed: bool
) -> ReflinkPlan:
    """Decide the checkout path before anything is created.

    ``wt_parent`` must exist (it holds the capability probe's temp files).
    """
    if mode == "never" and not seed:
        return ReflinkPlan(None, None, "reflink=never", False)
    support = probe_reflink(wt_parent)
    if not support.supported:
        return ReflinkPlan(support, None, support.reason, False)
    target = _target_commit(canonical, branch, base)
    listing = git_out(["worktree", "list", "--porcelain"], canonical)
    if target is None or listing is None:
        return ReflinkPlan(
            support, None, "could not resolve target commit or worktrees", False
        )
    candidates = parse_candidates(listing)
    device = os.stat(wt_parent).st_dev
    source = select_source(candidates, target, device=device, tree=canonical)
    if source is None:
        return ReflinkPlan(
            support, None, "no source worktree on the same filesystem", False
        )
    attempt = mode != "never"
    reason = (
        "reflink=never"
        if not attempt
        else f"source worktree {os.path.basename(source.path)}"
    )
    seed_source = None
    if seed:
        warm = [c for c in candidates if _has_artifacts(c.path)]
        seed_source = select_source(warm, target, device=device, tree=canonical)
    return ReflinkPlan(support, source, reason, attempt, seed_source)


def _has_artifacts(tree: str) -> bool:
    return any(os.path.isdir(os.path.join(tree, n)) for n in seed_artifact_names())
