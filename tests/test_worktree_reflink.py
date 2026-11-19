"""Copy-on-write worktree creation (CONCEPT:RM-WORKTREE-REFLINK)."""

import argparse
import errno
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from repository_manager import reflink as rl
from repository_manager import worktree as wt_mod
from repository_manager.worktree import WorktreeManager
from tests.test_worktree import FakeGit


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _plain_copy(src, dst):
    shutil.copyfile(src, dst)
    shutil.copymode(src, dst)


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    rl.clear_probe_cache()
    yield
    rl.clear_probe_cache()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo with three commits: main = c2, branch ``tgt`` = c3 (child of c2)."""
    monkeypatch.setenv("PRE_COMMIT_HOME", str(tmp_path / "pre-commit-home"))
    ws = tmp_path / "workspace"
    path = ws / "myrepo"
    (path / "sub").mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@t.io")
    _git(path, "config", "user.name", "t")
    for name, text in (("a.txt", "a1\n"), ("b.txt", "b1\n"), ("sub/c.txt", "c1\n")):
        (path / name).write_text(text)
    (path / "run.sh").write_text("#!/bin/sh\n")
    os.chmod(path / "run.sh", 0o755)
    _git(path, "add", "--", "a.txt", "b.txt", "sub/c.txt", "run.sh")
    _git(path, "commit", "-q", "-m", "c1")
    c1 = _git(path, "rev-parse", "HEAD")
    (path / "a.txt").write_text("a2\n")
    _git(path, "rm", "-q", "b.txt")
    (path / "d.txt").write_text("d2\n")
    _git(path, "add", "--", "a.txt", "d.txt")
    _git(path, "commit", "-q", "-m", "c2")
    _git(path, "checkout", "-q", "-b", "tgt")
    (path / "b.txt").write_text("b3\n")
    (path / "sub" / "e.txt").write_text("e3\n")
    _git(path, "add", "--", "b.txt", "sub/e.txt")
    _git(path, "commit", "-q", "-m", "c3")
    _git(path, "checkout", "-q", "main")
    monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", str(tmp_path / "worktrees"))
    git = FakeGit(str(ws), {"git@x/myrepo.git": str(path)})
    return SimpleNamespace(
        wm=WorktreeManager(git), path=str(path), c1=c1, git=git, tmp=tmp_path
    )


def _snapshot(tree):
    out = {}
    for root, dirs, files in os.walk(tree):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in (f for f in files if f != ".git"):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, tree)
            with open(p, "rb") as fh:
                out[rel] = (fh.read(), os.stat(p).st_mode & 0o111)
    return out


def _unsupported(monkeypatch):
    def refuse(src, dst):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(rl, "reflink_file", refuse)


# ── RM-GOVERNANCE-R034: capability probe ─────────────────────────────────
@pytest.mark.spec("RM-GOVERNANCE-R034")
def test_probe_reports_supported_when_the_clone_succeeds(tmp_path):
    seen = []

    def clone(src, dst):
        seen.append((src, dst))
        shutil.copyfile(src, dst)

    result = rl.probe_reflink(str(tmp_path), clone=clone)
    assert result.supported is True
    assert len(seen) == 1
    assert os.listdir(tmp_path) == []  # temp files cleaned up


@pytest.mark.spec("RM-GOVERNANCE-R034")
def test_probe_answers_no_without_errors_when_the_clone_is_refused(tmp_path):
    def clone(src, dst):
        raise OSError(errno.EOPNOTSUPP, "no")

    result = rl.probe_reflink(str(tmp_path), clone=clone)
    assert result.supported is False
    assert "does not support" in result.reason
    assert os.listdir(tmp_path) == []


@pytest.mark.spec("RM-GOVERNANCE-R034")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="FICLONE is Linux")
@pytest.mark.parametrize("fails", [False, True])
def test_probe_drives_the_ficlone_ioctl(tmp_path, monkeypatch, fails):
    import fcntl

    calls = []

    def fake_ioctl(fd, request, arg):
        calls.append(request)
        if fails:
            raise OSError(errno.EOPNOTSUPP, "no")
        return 0

    monkeypatch.setattr(fcntl, "ioctl", fake_ioctl)
    result = rl.probe_reflink(str(tmp_path))
    assert result.supported is (not fails)
    assert calls == [0x40049409]


@pytest.mark.spec("RM-GOVERNANCE-R034")
def test_probe_result_is_cached_per_device(tmp_path):
    calls = []

    def clone(src, dst):
        calls.append(1)
        shutil.copyfile(src, dst)

    sub = tmp_path / "other"
    sub.mkdir()
    assert rl.probe_reflink(str(tmp_path), clone=clone).supported
    assert rl.probe_reflink(str(sub), clone=clone).supported  # same device
    assert len(calls) == 1


@pytest.mark.spec("RM-GOVERNANCE-R034")
def test_probe_of_a_missing_directory_is_a_no_not_an_error(tmp_path):
    result = rl.probe_reflink(str(tmp_path / "missing"))
    assert result.supported is False


# ── RM-GOVERNANCE-R035: source selection and reflink checkout ───────────
@pytest.mark.spec("RM-GOVERNANCE-R035")
def test_select_source_prefers_the_closest_head_on_the_same_device(tmp_path):
    near, far, other = (tmp_path / n for n in ("near", "far", "other"))
    for d in (near, far, other):
        d.mkdir()
    dev = os.stat(tmp_path).st_dev
    dist = {"n": 2, "f": 9, "o": 0}
    cands = [
        rl.SourceCandidate(str(far), "f"),
        rl.SourceCandidate(str(near), "n"),
        rl.SourceCandidate(str(other), "o"),
        rl.SourceCandidate(str(tmp_path / "gone"), "g"),
    ]
    picked = rl.select_source(
        cands,
        "t",
        device=dev,
        tree=str(tmp_path),
        distance=lambda c, t, tree: dist.get(c.head),
    )
    assert picked is not None and picked.path == str(other)
    cands.pop(2)
    picked = rl.select_source(
        cands,
        "t",
        device=dev,
        tree=str(tmp_path),
        distance=lambda c, t, r: dist.get(c.head),
    )
    assert picked is not None and picked.path == str(near)
    assert (
        rl.select_source(
            cands, "t", device=dev + 1, tree=str(tmp_path), distance=lambda c, t, r: 0
        )
        is None
    )


@pytest.mark.spec("RM-GOVERNANCE-R035")
def test_select_source_uses_the_changed_file_count_from_git(repo):
    c3 = _git(repo.path, "rev-parse", "tgt")
    c2 = _git(repo.path, "rev-parse", "main")
    cands = [
        rl.SourceCandidate(repo.path + "/sub", repo.c1),  # a path on the device
        rl.SourceCandidate(repo.path, c2),
    ]
    picked = rl.select_source(
        cands, c3, device=os.stat(repo.path).st_dev, tree=repo.path
    )
    assert picked is not None and picked.head == c2  # 2 files differ, versus 4 for c1


def _plain_reference(repo, branch="ref"):
    res = repo.wm.add("myrepo", branch, base="tgt", reflink="never")
    assert res["ok"], res
    return res["path"]


@pytest.mark.spec("RM-GOVERNANCE-R035")
def test_reflink_checkout_is_identical_to_a_plain_checkout(repo, monkeypatch):
    # dirty canonical source: a modified tracked file and an untracked one
    with open(os.path.join(repo.path, "a.txt"), "w") as f:
        f.write("dirty\n")
    with open(os.path.join(repo.path, "scratch.log"), "w") as f:
        f.write("untracked\n")
    reference = _plain_reference(repo)
    copies = []

    def copy(src, dst):
        copies.append(dst)
        _plain_copy(src, dst)

    monkeypatch.setattr(rl, "reflink_file", copy)
    res = repo.wm.add("myrepo", "cow", base="tgt")

    assert res["ok"] and res["created"], res
    assert res["checkout"]["path"] == "reflink"
    assert res["checkout"]["fallback"] is False
    assert copies, "the copy primitive was used"
    cow = res["path"]
    assert _git(cow, "status", "--porcelain") == ""
    assert _git(cow, "rev-parse", "HEAD^{tree}") == _git(
        reference, "rev-parse", "HEAD^{tree}"
    )
    assert _git(cow, "write-tree") == _git(reference, "write-tree")
    assert _snapshot(cow) == _snapshot(reference)
    assert not os.path.exists(os.path.join(cow, "scratch.log"))


@pytest.mark.spec("RM-GOVERNANCE-R035")
def test_reflink_checkout_removes_files_the_target_does_not_track(repo, monkeypatch):
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    res = repo.wm.add("myrepo", "old-base", base=repo.c1)
    assert res["checkout"]["path"] == "reflink"
    assert not os.path.exists(os.path.join(res["path"], "d.txt"))
    assert os.path.exists(os.path.join(res["path"], "b.txt"))
    assert _git(res["path"], "status", "--porcelain") == ""


@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_a_failed_reflink_checkout_falls_back_to_plain_and_says_so(repo, monkeypatch):
    reference = _plain_reference(repo)
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    monkeypatch.setattr(rl, "reconcile", lambda tree, paths: "forced failure")
    res = repo.wm.add("myrepo", "cow", base="tgt")
    assert res["ok"]
    assert res["checkout"]["path"] == "plain"
    assert res["checkout"]["fallback"] is True
    assert "forced failure" in res["checkout"]["reason"]
    assert _git(res["path"], "status", "--porcelain") == ""
    assert _snapshot(res["path"]) == _snapshot(reference)


# ── RM-GOVERNANCE-R037: fallback behaviour and reporting ─────────────────
@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_unsupported_filesystem_takes_the_plain_path_identical_to_git(
    repo, monkeypatch
):
    _unsupported(monkeypatch)
    res = repo.wm.add("myrepo", "plain-auto", base="tgt")
    assert res["ok"] and res["created"]
    assert res["checkout"] == {
        "path": "plain",
        "reason": "filesystem does not support reflinks",
        "fallback": False,
    }
    git_wt = str(repo.tmp / "gitwt")
    _git(repo.path, "worktree", "add", "-q", "-b", "gitplain", git_wt, "tgt")
    assert _git(res["path"], "status", "--porcelain") == ""
    assert _git(res["path"], "write-tree") == _git(git_wt, "write-tree")
    assert _snapshot(res["path"]) == _snapshot(git_wt)


@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_reflink_never_is_reported_and_never_probes(repo, monkeypatch):
    monkeypatch.setattr(rl, "probe_reflink", lambda *a, **k: pytest.fail("probed"))
    res = repo.wm.add("myrepo", "never-x", base="tgt", reflink="never")
    assert res["checkout"]["path"] == "plain"
    assert res["checkout"]["reason"] == "reflink=never"
    assert "seed" not in res


@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_no_source_worktree_on_the_device_falls_back_with_a_reason(repo, monkeypatch):
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    monkeypatch.setattr(rl, "_same_device", lambda path, device: False)
    res = repo.wm.add("myrepo", "no-src", base="tgt")
    assert res["checkout"]["path"] == "plain"
    assert res["checkout"]["reason"] == "no source worktree on the same filesystem"
    assert _git(res["path"], "status", "--porcelain") == ""


@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_invalid_reflink_mode_is_refused(repo):
    res = repo.wm.add("myrepo", "bad", reflink="sometimes")
    assert res["ok"] is False and "reflink" in res["error"]


# ── RM-GOVERNANCE-R036: seeded build artifacts ───────────────────────────
def _make_artifacts(repo):
    for rel in ("target/debug/app", "node_modules/pkg/index.js", ".venv/bin/python"):
        p = os.path.join(repo.path, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(rel)
    os.symlink("pkg", os.path.join(repo.path, "node_modules", "link"))


@pytest.mark.spec("RM-GOVERNANCE-R036")
def test_seed_artifacts_reflinks_configured_directories(repo, monkeypatch):
    _make_artifacts(repo)
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    res = repo.wm.add("myrepo", "warm", base="tgt", seed_artifacts=True)
    assert res["seed"]["status"] == "seeded"
    assert set(res["seed"]["artifacts"]) == {"target", "node_modules"}
    for rel in ("target/debug/app", "node_modules/pkg/index.js"):
        assert open(os.path.join(res["path"], rel)).read() == rel
    assert not os.path.exists(os.path.join(res["path"], ".venv"))
    assert os.readlink(os.path.join(res["path"], "node_modules", "link")) == "pkg"
    assert _git(res["path"], "status", "--porcelain") == "?? node_modules/\n?? target/"


@pytest.mark.spec("RM-GOVERNANCE-R036")
@pytest.mark.parametrize("name", [".venv", "venv", "custom-env"])
def test_a_virtual_environment_is_refused_as_path_bound(repo, monkeypatch, name):
    _make_artifacts(repo)
    os.makedirs(os.path.join(repo.path, "custom-env"))
    with open(os.path.join(repo.path, "custom-env", "pyvenv.cfg"), "w") as f:
        f.write("home = /usr/bin\n")
    os.makedirs(os.path.join(repo.path, "venv", "bin"))
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    monkeypatch.setenv(rl.SEED_ARTIFACTS_ENV, f"target,{name}")
    res = repo.wm.add(
        "myrepo", "venv-" + name.strip("."), base="tgt", seed_artifacts=True
    )
    refused = res["seed"]["artifacts"][name]
    assert refused["status"] == "refused"
    assert "path-bound" in refused["reason"]
    assert res["seed"]["status"] == "partial"
    assert not os.path.exists(os.path.join(res["path"], name))
    assert res["seed"]["artifacts"]["target"]["status"] == "seeded"


@pytest.mark.spec("RM-GOVERNANCE-R036")
def test_seed_is_off_by_default_and_honours_the_configured_list(repo, monkeypatch):
    _make_artifacts(repo)
    monkeypatch.setattr(rl, "reflink_file", _plain_copy)
    res = repo.wm.add("myrepo", "cold", base="tgt")
    assert "seed" not in res and not os.path.exists(os.path.join(res["path"], "target"))
    monkeypatch.setenv(rl.SEED_ARTIFACTS_ENV, "target,missing")
    res = repo.wm.add("myrepo", "some", base="tgt", seed_artifacts=True)
    assert res["seed"]["status"] == "partial"
    assert res["seed"]["artifacts"]["missing"]["status"] == "skipped"
    assert not os.path.exists(os.path.join(res["path"], "node_modules"))


@pytest.mark.spec("RM-GOVERNANCE-R036")
def test_seed_without_reflink_support_is_skipped_and_reported_never_copied(
    repo, monkeypatch
):
    _make_artifacts(repo)
    _unsupported(monkeypatch)
    res = repo.wm.add("myrepo", "nocow", base="tgt", seed_artifacts=True)
    assert res["ok"]
    assert res["seed"]["status"] == "skipped"
    assert "does not support" in res["seed"]["reason"]
    assert not os.path.exists(os.path.join(res["path"], "target"))


@pytest.mark.spec("RM-GOVERNANCE-R036")
def test_a_failing_artifact_copy_leaves_no_partial_directory(repo, monkeypatch):
    _make_artifacts(repo)
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if "target" in src and src.endswith("app"):
            raise OSError(errno.EXDEV, "cross-device")
        _plain_copy(src, dst)

    monkeypatch.setattr(rl, "reflink_file", flaky)
    res = repo.wm.add("myrepo", "flaky", base="tgt", seed_artifacts=True)
    assert res["seed"]["artifacts"]["target"]["status"] == "skipped"
    assert not os.path.exists(os.path.join(res["path"], "target"))
    assert res["seed"]["artifacts"]["node_modules"]["status"] == "seeded"


# ── RM-GOVERNANCE-R037: exposure ─────────────────────────────────────────
@pytest.mark.spec("RM-GOVERNANCE-R037")
def test_lane_start_and_cli_expose_the_two_parameters(repo, monkeypatch):
    from repository_manager import lane_doctor
    from repository_manager.cli_commands.lane import run_lane_cli

    result = lane_doctor.start(
        "myrepo", "lane-x", base="tgt", git=repo.git, reflink="never"
    )
    assert result["checkout"]["reason"] == "reflink=never"

    seen = {}
    monkeypatch.setattr(
        lane_doctor, "dispatch", lambda action, **kw: seen.update(kw) or {"ok": True}
    )
    args = argparse.Namespace(
        lane="start",
        lane_path=None,
        lane_repo="myrepo",
        lane_branch="b",
        lane_base="main",
        workspace=None,
        lane_force=False,
        lane_shell=False,
        lane_reflink="never",
        lane_seed_artifacts=True,
    )
    assert run_lane_cli(args) == 0
    assert seen["reflink"] == "never" and seen["seed_artifacts"] is True


# ── opt-in: real reflinks, no mocks ──────────────────────────────────────
REAL_DIR_ENV = "RM_REFLINK_TEST_DIR"


def _extents(path):
    """``(physical_start, shared)`` per extent from ``filefrag -v``."""
    out = subprocess.run(
        ["filefrag", "-v", path], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        cols = line.split(":")
        if len(cols) > 4 and cols[0].strip().isdigit():
            rows.append((cols[2].strip().split("..")[0], "shared" in line))
    return rows


@pytest.mark.spec("RM-GOVERNANCE-R035")
@pytest.mark.skipif(
    not os.environ.get(REAL_DIR_ENV) or shutil.which("filefrag") is None,
    reason=f"set {REAL_DIR_ENV} to a directory on a reflink filesystem",
)
def test_real_reflinks_are_byte_identical_and_share_extents(tmp_path, monkeypatch):
    import tempfile

    base_dir = tempfile.mkdtemp(dir=os.environ[REAL_DIR_ENV], prefix="rm-real-")
    try:
        ws = os.path.join(base_dir, "workspace")
        path = os.path.join(ws, "myrepo")
        os.makedirs(path)
        _git(path, "init", "-q", "-b", "main")
        _git(path, "config", "user.email", "t@t.io")
        _git(path, "config", "user.name", "t")
        with open(os.path.join(path, "big.bin"), "wb") as f:
            f.write(os.urandom(64 * 1024 * 1024))
        with open(os.path.join(path, "a.txt"), "w") as f:
            f.write("a1\n")
        _git(path, "add", "--", "big.bin", "a.txt")
        _git(path, "commit", "-q", "-m", "c1")
        with open(os.path.join(path, "a.txt"), "w") as f:
            f.write("a2\n")
        _git(path, "commit", "-q", "-am", "c2")
        os.makedirs(os.path.join(path, "target"))
        with open(os.path.join(path, "target", "obj.bin"), "wb") as f:
            f.write(os.urandom(32 * 1024 * 1024))
        monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", os.path.join(base_dir, "wts"))
        wm = WorktreeManager(FakeGit(ws, {"x": path}))

        assert rl.probe_reflink(os.path.join(base_dir)).supported is True
        reference = wm.add("myrepo", "ref", base="main", reflink="never")["path"]
        res = wm.add("myrepo", "cow", base="main", seed_artifacts=True)

        assert res["checkout"]["path"] == "reflink", res
        assert res["checkout"]["fallback"] is False
        assert _git(res["path"], "status", "--porcelain") == "?? target/"
        assert _git(res["path"], "write-tree") == _git(reference, "write-tree")
        assert {
            k: v for k, v in _snapshot(res["path"]).items() if k != "target/obj.bin"
        } == _snapshot(reference)
        assert res["seed"]["artifacts"]["target"]["status"] == "seeded"
        for rel, src in (
            ("big.bin", os.path.join(path, "big.bin")),
            ("target/obj.bin", os.path.join(path, "target", "obj.bin")),
        ):
            dst = os.path.join(res["path"], rel)
            subprocess.run(["sync", "-f", dst], check=False)
            with open(src, "rb") as a, open(dst, "rb") as b:
                assert a.read() == b.read()
            sx, dx = _extents(src), _extents(dst)
            assert sx and dx
            assert [e[0] for e in sx] == [e[0] for e in dx], (rel, sx, dx)
            assert all(e[1] for e in dx), (rel, "extents not flagged shared", dx)
        # the plain reference does not share extents with the source
        assert [e[0] for e in _extents(os.path.join(reference, "big.bin"))] != [
            e[0] for e in _extents(os.path.join(path, "big.bin"))
        ]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
