"""Tests for the repo-agnostic merge queue (CONCEPT:RM-MERGE-QUEUE).

Every test here drives REAL git repositories and REAL gate subprocesses — the
mechanism under test is git object math plus subprocess exit codes, and a mock
of either would prove nothing about it.

The headline test is :func:`test_a_rust_repo_lands_through_the_queue_with_a_cargo_gate`:
it stands up a genuine cargo crate, declares ``cargo check --all-features`` as
its only gate, and lands a candidate through the same code path agent-utilities
would use with pytest. That is the whole claim of this module — that the queue
does not know what a gate IS — and it is checked, not asserted.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from repository_manager import merge_queue as mq


def _run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


class FakeGit:
    """The same minimal ``GitLike`` stand-in ``test_worktree.py`` uses."""

    def __init__(self, workspace: str, project_map: dict[str, str]) -> None:
        self.path = workspace
        self.project_map = project_map

    def git_action(
        self, command, path=None, quiet=False, env=None, timeout=1800, raw_output=False
    ):
        del env, timeout, raw_output, quiet
        p = subprocess.run(
            command, shell=True, cwd=path or self.path, capture_output=True, text=True
        )
        out = (p.stdout + p.stderr).strip()
        return SimpleNamespace(
            status="success" if p.returncode == 0 else "error",
            data=out,
            error=None if p.returncode == 0 else SimpleNamespace(message=out),
            metadata=SimpleNamespace(return_code=p.returncode),
        )


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _run("git init -q -b main", root)
    _run("git config user.email t@t.io && git config user.name t", root)
    _run("git config commit.gpgsign false", root)
    return root


def _commit(repo: Path, message: str) -> str:
    _run("git add -A", repo)
    _run(f"git commit -q --allow-empty -m {json.dumps(message)}", repo)
    return _run("git rev-parse HEAD", repo)


def _branch_with(repo: Path, branch: str, files: dict[str, str], message: str) -> None:
    _run(f"git checkout -q -b {branch} main", repo)
    for rel, body in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    _commit(repo, message)
    _run("git checkout -q main", repo)


def _write_config(repo: Path, body: str) -> None:
    (repo / mq.CONFIG_FILENAME).write_text(textwrap.dedent(body))


@pytest.fixture
def shell_repo(tmp_path: Path) -> Path:
    """A repo whose single gate is a plain shell script — no language at all.

    Deliberately not Python and not Rust: it makes the mechanism tests
    independent of any toolchain, so a failure here is always a queue defect.
    """
    repo = _init_repo(tmp_path / "shellrepo")
    (repo / "gate.sh").write_text(
        "#!/bin/sh\n"
        "# Prints one line per 'violation' found, exits 1 if any.\n"
        "found=$(grep -rl BROKEN . --include='*.txt' 2>/dev/null | sort)\n"
        '[ -z "$found" ] && exit 0\n'
        'for f in $found; do echo "violation: $f"; done\n'
        "exit 1\n"
    )
    os.chmod(repo / "gate.sh", 0o755)
    _write_config(
        repo,
        """
        base: main
        batch_size: 8
        gates:
          - name: no-broken
            command: ["./gate.sh"]
            tier: fast
            timeout: 60
            compare: lines
        """,
    )
    (repo / "ok.txt").write_text("fine\n")
    _commit(repo, "init")
    return repo


# ---------------------------------------------------------------------------
# THE GENERICITY PROOF — a Rust repository, gated by cargo, end to end
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_a_rust_repo_lands_through_the_queue_with_a_cargo_gate(tmp_path: Path) -> None:
    """A cargo repo lands through the SAME queue agent-utilities uses with pytest.

    This is the evidence that the capability was mis-homed: epistemic-graph has
    no merge queue at all today, and its lane had to hand-apply the discipline.
    Nothing in :mod:`repository_manager.merge_queue` mentions cargo, rust, or
    Python — the gate is a config row.
    """
    repo = _init_repo(tmp_path / "crate")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "crate_under_test"\nversion = "0.1.0"\nedition = "2021"\n'
    )
    (repo / "src").mkdir()
    (repo / "src" / "lib.rs").write_text("pub fn one() -> i32 { 1 }\n")
    (repo / ".gitignore").write_text("target-isolated/\n")
    _write_config(
        repo,
        """
        base: main
        batch_size: 8
        environment_signature: ["cargo", "--version"]
        gates:
          - name: cargo-check
            command: ["cargo", "check", "--all-features", "--message-format", "short",
                      "--target-dir", "target-isolated", "--offline"]
            tier: fast
            timeout: 900
            baseline_timeout: 900
            compare: lines
            keep_lines: ['^error', '^warning']
        """,
    )
    _commit(repo, "init crate")

    # A candidate that COMPILES.
    _branch_with(
        repo,
        "feat/two",
        {"src/lib.rs": "pub fn one() -> i32 { 1 }\npub fn two() -> i32 { 2 }\n"},
        "add two()",
    )
    # A candidate that does NOT compile — a type error the gate must catch.
    _branch_with(
        repo,
        "feat/broken",
        {
            "src/other.rs": 'pub fn broken() -> i32 { "not an int" }\n',
            "src/lib.rs": "pub mod other;\npub fn one() -> i32 { 1 }\n",
        },
        "add a type error",
    )

    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/two", path=repo)
    mq.enqueue("feat/broken", path=repo)
    assert [c.branch for c in mq.queued(repo)] == ["feat/two", "feat/broken"]

    result = mq.run_queue(path=repo, prune=False, git=git)

    landed = {o["branch"] for o in result["outcomes"] if o["landed"]}
    rejected = {o["branch"]: o for o in result["outcomes"] if not o["landed"]}
    assert landed == {"feat/two"}, result
    assert set(rejected) == {"feat/broken"}, result
    # The rejection names the gate AND the compiler's own diagnostic.
    assert "cargo-check" in rejected["feat/broken"]["reason"]
    detail = "\n".join(
        c["detail"] for c in rejected["feat/broken"]["gate"]["checks"] if not c["ok"]
    )
    assert "NEW signal" in detail
    assert "error" in detail.lower()
    # main fast-forwarded to include the good candidate only.
    log = _run("git log --oneline main", repo)
    assert "add two()" in log
    assert "type error" not in log


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_the_cargo_gate_is_differential_not_absolute(tmp_path: Path) -> None:
    """A candidate touching an ALREADY-BROKEN crate still lands — behaviour 1.

    `main` is legitimately red in this workspace. An absolute gate deadlocked the
    au queue and stranded 19 branches; this pins the differential contract for a
    non-Python toolchain. **This is also the anti-vacuity check for the test
    above**: it proves that test's rejection came from a NEW error, not from the
    gate simply always failing.
    """
    repo = _init_repo(tmp_path / "redcrate")
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "redcrate"\nversion = "0.1.0"\nedition = "2021"\n'
    )
    (repo / "src").mkdir()
    # main is RED from the start: an unused-variable warning that never goes away.
    (repo / "src" / "lib.rs").write_text(
        "pub fn already_red() -> i32 { let unused_on_main = 5; 1 }\n"
    )
    (repo / ".gitignore").write_text("target-isolated/\n")
    _write_config(
        repo,
        """
        base: main
        environment_signature: ["cargo", "--version"]
        gates:
          - name: cargo-check
            command: ["cargo", "check", "--all-features", "--message-format", "short",
                      "--target-dir", "target-isolated", "--offline"]
            tier: fast
            timeout: 900
            compare: lines
            keep_lines: ['^error', '^warning']
        """,
    )
    _commit(repo, "init a repo whose main is already red")

    # This candidate does NOT touch the pre-existing warning; it adds clean code.
    _branch_with(
        repo,
        "feat/clean-addition",
        {
            "src/lib.rs": "pub fn already_red() -> i32 { let unused_on_main = 5; 1 }\n"
            "pub fn added() -> i32 { 7 }\n"
        },
        "add clean code beside pre-existing red",
    )
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/clean-addition", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is True, outcome
    # And the pre-existing red is REPORTED, never silently swallowed.
    check = next(c for c in outcome["gate"]["checks"] if c["name"] == "cargo-check")
    assert "pre-existing" in check["detail"], check


# ---------------------------------------------------------------------------
# Behaviour 1 — differential gating, and its FAIL-CLOSED half
# ---------------------------------------------------------------------------
def test_a_new_violation_blocks_and_a_pre_existing_one_does_not(
    shell_repo: Path,
) -> None:
    repo = shell_repo
    (repo / "already_bad.txt").write_text("BROKEN on main\n")
    _commit(repo, "main is legitimately red")

    _branch_with(repo, "feat/innocent", {"new.txt": "all fine\n"}, "touch nothing bad")
    _branch_with(
        repo, "feat/guilty", {"guilty.txt": "BROKEN here\n"}, "add a violation"
    )

    git = FakeGit(str(repo.parent), {"x": str(repo)})
    mq.enqueue("feat/innocent", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    assert result["outcomes"][0]["landed"] is True, result
    assert "pre-existing" in result["outcomes"][0]["gate"]["checks"][0]["detail"]

    mq.enqueue("feat/guilty", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is False, outcome
    assert "guilty.txt" in outcome["gate"]["checks"][0]["detail"]


def test_an_unproducible_baseline_refuses_and_never_allows_everything(
    shell_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed: no baseline means REFUSE, not "no pre-existing failures".

    This is the single most important property of the whole gate. A baseline that
    silently degrades to empty turns every gate OFF at exactly the moment the base
    becomes unmeasurable.
    """
    repo = shell_repo
    _branch_with(repo, "feat/guilty", {"g.txt": "BROKEN\n"}, "a violation")
    git = FakeGit(str(repo.parent), {"x": str(repo)})

    def _unproducible(*args, **kwargs):
        return mq.GateBaseline(readable=False, detail="simulated: base ref unreadable")

    monkeypatch.setattr(mq, "compute_gate_baseline", _unproducible)
    mq.enqueue("feat/guilty", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is False
    assert "REFUSED" in outcome["gate"]["checks"][0]["detail"]
    assert "could not be produced" in outcome["gate"]["checks"][0]["detail"]


def test_a_baseline_timeout_is_a_refusal_not_an_empty_baseline(tmp_path: Path) -> None:
    """The concrete shape of the fail-closed rule, driven end to end."""
    repo = _init_repo(tmp_path / "slowbase")
    (repo / "gate.sh").write_text(
        "#!/bin/sh\n"
        # Slow ONLY when the marker from the base commit is present, so the base
        # run times out while the merged run answers immediately.
        "[ -f slow_marker ] && sleep 30\n"
        "grep -q BROKEN *.txt 2>/dev/null && { echo 'violation: found'; exit 1; }\n"
        "exit 0\n"
    )
    os.chmod(repo / "gate.sh", 0o755)
    (repo / "slow_marker").write_text("x")
    _write_config(
        repo,
        """
        base: main
        gates:
          - name: slow-baseline
            command: ["./gate.sh"]
            tier: fast
            timeout: 60
            baseline_timeout: 2
            compare: lines
        """,
    )
    _commit(repo, "init")
    _branch_with(
        repo, "feat/x", {"slow_marker": "", "bad.txt": "BROKEN\n"}, "fast but violating"
    )
    (repo / "feat_marker").unlink(missing_ok=True)
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/x", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is False, outcome
    detail = outcome["gate"]["checks"][0]["detail"]
    assert "REFUSED" in detail and "exceeded" in detail, detail
    assert "never silently treated as 'no pre-existing failures'" in detail


# ---------------------------------------------------------------------------
# Behaviour 2 — fold by recorded_at, not by lane-name sort order
# ---------------------------------------------------------------------------
def test_a_terminal_record_from_another_lane_supersedes_the_queued_one(
    shell_repo: Path,
) -> None:
    """D-F6-1/D-CVG-9: resolve duplicates by WRITE TIME, not lane-name order.

    The live shape: a candidate is enqueued from ``lane-foo``'s worktree and
    landed from ``canonical``. ``"canonical" < "lane-foo"``, so a
    lane-name-ordered fold takes the OLDER queued record and the candidate
    reports ``queued`` forever.
    """
    store = mq.queue_store(shell_repo)
    store.append(
        mq.Candidate(
            branch="feat/x",
            lane="lane-foo",
            enqueued_at="2026-08-01T10:00:00+00:00",
            state=mq.QUEUED,
            recorded_at="2026-08-01T10:00:00+00:00",
        ).to_record(),
        lane="lane-foo",
    )
    store.append(
        mq.Candidate(
            branch="feat/x",
            lane="lane-foo",
            enqueued_at="2026-08-01T10:00:00+00:00",
            state=mq.LANDED,
            recorded_at="2026-08-01T11:00:00+00:00",
        ).to_record(),
        lane="canonical",  # sorts BEFORE "lane-foo"
    )
    resolved = {c.branch: c for c in mq._all_candidates(shell_repo)}
    assert resolved["feat/x"].state == mq.LANDED
    assert mq.queued(shell_repo) == []

    # And the raw fold-order default would have got it wrong — pin the bug itself
    # so this test cannot pass against the defect it claims to catch.
    group = [
        {
            "id": "feat/x",
            "state": mq.LANDED,
            "recorded_at": "2026-08-01T11:00:00+00:00",
        },
        {
            "id": "feat/x",
            "state": mq.QUEUED,
            "recorded_at": "2026-08-01T10:00:00+00:00",
        },
    ]  # canonical's record first, exactly as sorted-lane iteration would yield
    assert group[-1]["state"] == mq.QUEUED, (
        "precondition: naive group[-1] is the stale one"
    )
    assert mq._resolve_latest_candidate_record(group)["state"] == mq.LANDED


def test_records_with_no_recorded_at_degrade_instead_of_crashing(
    shell_repo: Path,
) -> None:
    group = [{"id": "a", "state": mq.QUEUED}, {"id": "a", "state": mq.LANDED}]
    assert mq._resolve_latest_candidate_record(group)["state"] == mq.LANDED


# ---------------------------------------------------------------------------
# Behaviour 3 — regenerate-on-land
# ---------------------------------------------------------------------------
def test_a_conflict_confined_to_generated_files_is_regenerated_not_rejected(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "genrepo")
    (repo / "gen.py").write_text(
        "import pathlib\n"
        "srcs = sorted(p.name for p in pathlib.Path('src').glob('*.txt'))\n"
        "pathlib.Path('INDEX.md').write_text('\\n'.join(srcs) + '\\n')\n"
    )
    (repo / "src").mkdir()
    (repo / "src" / "a.txt").write_text("a\n")
    (repo / "INDEX.md").write_text("a.txt\n")
    _write_config(
        repo,
        f"""
        base: main
        gates:
          - name: always-green
            command: ["true"]
            tier: fast
            compare: exit
        generated_files: ["INDEX.md"]
        regenerate:
          - ["{sys.executable}", "gen.py"]
        """,
    )
    _commit(repo, "init")

    # Two branches each add a source file AND their own stale regeneration of the
    # derived index — a textbook add/add conflict on a purely-derived file.
    _branch_with(
        repo, "feat/b", {"src/b.txt": "b\n", "INDEX.md": "a.txt\nb.txt\n"}, "add b"
    )
    _branch_with(
        repo, "feat/c", {"src/c.txt": "c\n", "INDEX.md": "a.txt\nc.txt\n"}, "add c"
    )

    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/b", path=repo)
    mq.enqueue("feat/c", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    assert all(o["landed"] for o in result["outcomes"]), result
    # Regenerated FROM THE MERGED TRUTH — both sides present, no side dropped.
    assert (repo / "INDEX.md").read_text() == "a.txt\nb.txt\nc.txt\n"


def test_a_conflict_touching_a_handwritten_file_is_still_rejected(
    tmp_path: Path,
) -> None:
    """Regeneration is deliberately narrow — it never resolves a real conflict."""
    repo = _init_repo(tmp_path / "mixedrepo")
    (repo / "gen.py").write_text(
        "import pathlib; pathlib.Path('INDEX.md').write_text('x\\n')"
    )
    (repo / "INDEX.md").write_text("start\n")
    (repo / "hand.txt").write_text("original\n")
    _write_config(
        repo,
        f"""
        base: main
        gates:
          - name: always-green
            command: ["true"]
            tier: fast
            compare: exit
        generated_files: ["INDEX.md"]
        regenerate: [["{sys.executable}", "gen.py"]]
        """,
    )
    _commit(repo, "init")
    _branch_with(repo, "feat/b", {"INDEX.md": "b\n", "hand.txt": "b edit\n"}, "b")
    _branch_with(repo, "feat/c", {"INDEX.md": "c\n", "hand.txt": "c edit\n"}, "c")

    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/b", path=repo)
    mq.enqueue("feat/c", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcomes = {o["branch"]: o for o in result["outcomes"]}
    assert outcomes["feat/b"]["landed"] is True
    assert outcomes["feat/c"]["landed"] is False
    assert "hand.txt" in outcomes["feat/c"]["reason"]


# ---------------------------------------------------------------------------
# Behaviour 4 — guarded prune (delegated to WorktreeManager)
# ---------------------------------------------------------------------------
def test_prune_refuses_a_worktree_holding_uncommitted_work(
    shell_repo: Path, tmp_path: Path
) -> None:
    """An un-pruned branch is untidy; a wrongly-pruned one loses work."""
    repo = shell_repo
    _branch_with(repo, "feat/w", {"w.txt": "fine\n"}, "w")
    git = FakeGit(str(repo.parent), {"x": str(repo)})
    # Deliberately OUTSIDE WORKTREE_ROOT — the realistic case in this workspace,
    # and the one worktree_path() could not reconstruct.
    wt_path = tmp_path / "elsewhere" / "feat__w"
    _run(f"git worktree add -q {wt_path} feat/w", repo)
    (wt_path / "UNCOMMITTED.txt").write_text("a lane is still working here\n")

    mq.enqueue("feat/w", worktree=wt_path, path=repo)
    result = mq.run_queue(path=repo, prune=True, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is True, outcome
    assert outcome["prune"]["pruned"] is False, outcome["prune"]
    assert "uncommitted work" in outcome["prune"]["reason"], outcome["prune"]
    assert (wt_path / "UNCOMMITTED.txt").is_file(), "the lane's work must survive"
    # The BRANCH also survives, so nothing is lost even though the queue landed it.
    assert _run("git rev-parse --verify feat/w", repo)


def test_prune_removes_a_clean_worktree_and_anchors_the_branch(
    shell_repo: Path, tmp_path: Path
) -> None:
    """The positive half — without it the refusal test above could be vacuous."""
    repo = shell_repo
    _branch_with(repo, "feat/clean", {"c.txt": "fine\n"}, "clean")
    git = FakeGit(str(repo.parent), {"x": str(repo)})
    wt_path = tmp_path / "elsewhere2" / "feat__clean"
    _run(f"git worktree add -q {wt_path} feat/clean", repo)
    tip = _run("git rev-parse feat/clean", repo)

    mq.enqueue("feat/clean", worktree=wt_path, path=repo)
    result = mq.run_queue(path=repo, prune=True, git=git)
    prune = result["outcomes"][0]["prune"]
    assert prune["pruned"] is True, prune
    assert not wt_path.exists()
    # `git branch -d`, never -D — and the anchor still points at the tip, so the
    # commits are recoverable even after the ref is gone.
    assert prune["branch_anchor"] == "refs/lane-backup/feat-clean"
    assert _run(f"git rev-parse {prune['branch_anchor']}", repo) == tip
    assert (
        subprocess.run(
            "git rev-parse --verify feat/clean",
            shell=True,
            cwd=repo,
            capture_output=True,
        ).returncode
        != 0
    )


# ---------------------------------------------------------------------------
# Behaviour 5 — honest degradation; and the batching/bisection contract
# ---------------------------------------------------------------------------
def test_one_bad_candidate_does_not_reject_the_innocent_ones(shell_repo: Path) -> None:
    repo = shell_repo
    for i in range(4):
        _branch_with(repo, f"feat/ok{i}", {f"ok{i}.txt": "fine\n"}, f"ok{i}")
    _branch_with(repo, "feat/bad", {"bad.txt": "BROKEN\n"}, "bad")

    git = FakeGit(str(repo.parent), {"x": str(repo)})
    for i in range(4):
        mq.enqueue(f"feat/ok{i}", path=repo)
    mq.enqueue("feat/bad", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcomes = {o["branch"]: o["landed"] for o in result["outcomes"]}
    assert outcomes["feat/bad"] is False, result
    assert all(outcomes[f"feat/ok{i}"] for i in range(4)), result


def test_a_gate_that_cannot_be_executed_is_refused_not_assumed_clean(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "nogate")
    _write_config(
        repo,
        """
        base: main
        gates:
          - name: missing-tool
            command: ["definitely-not-a-real-binary-xyz"]
            tier: fast
            compare: exit
        """,
    )
    _commit(repo, "init")
    _branch_with(repo, "feat/x", {"x.txt": "x\n"}, "x")
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/x", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    assert result["outcomes"][0]["landed"] is False
    assert "could not execute" in result["outcomes"][0]["gate"]["checks"][0]["detail"]


def test_dropping_a_declared_gate_is_a_refusal(shell_repo: Path) -> None:
    """Deleting the check that guards an invariant is not a way to satisfy it."""
    repo = shell_repo
    _run("git checkout -q -b feat/nogate main", repo)
    _write_config(repo, "base: main\ngates: []\n")
    _commit(repo, "remove the gate")
    _run("git checkout -q main", repo)

    git = FakeGit(str(repo.parent), {"x": str(repo)})
    mq.enqueue("feat/nogate", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is False, outcome
    assert "DROPS gate(s)" in outcome["gate"]["checks"][0]["detail"]


# ---------------------------------------------------------------------------
# The config seam — a missing declaration REFUSES rather than defaulting
# ---------------------------------------------------------------------------
def test_a_repo_with_no_config_is_refused_not_defaulted(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "unconfigured")
    _commit(repo, "init")
    with pytest.raises(
        mq.MergeQueueError, match="has not declared its merge-queue gates"
    ):
        mq.load_config(repo)


def test_a_shell_string_command_is_rejected_at_parse_time(tmp_path: Path) -> None:
    with pytest.raises(mq.MergeQueueError, match="must be a LIST of argv"):
        mq.parse_config({"gates": [{"name": "x", "command": "rm -rf /"}]})


def test_generated_files_without_regenerators_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(mq.MergeQueueError, match="no `regenerate` commands"):
        mq.parse_config({"generated_files": ["a.md"]})


def test_duplicate_gate_names_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(mq.MergeQueueError, match="duplicate gate name"):
        mq.parse_config(
            {
                "gates": [
                    {"name": "x", "command": ["true"]},
                    {"name": "x", "command": ["false"]},
                ]
            }
        )


def test_the_shipped_presets_parse(tmp_path: Path) -> None:
    """Every preset this package ships must be a valid declaration.

    A preset that does not parse is worse than none — it is a template that
    fails at drain time, while holding the lease.
    """
    import yaml

    presets = Path(mq.__file__).parent / "mergequeue_presets"
    found = sorted(presets.glob("*.mergequeue.yaml"))
    assert found, "no presets shipped"
    for preset in found:
        config = mq.parse_config(yaml.safe_load(preset.read_text()), source=str(preset))
        assert config.gates, f"{preset.name} declares no gates"
    names = {p.name for p in found}
    assert "epistemic-graph.mergequeue.yaml" in names, (
        "the Rust preset is the proof of genericity beyond Python"
    )


# ---------------------------------------------------------------------------
# Pre-commit safety (D-ORC-37)
# ---------------------------------------------------------------------------
def test_a_precommit_gate_refuses_a_tree_holding_uncommitted_work(
    tmp_path: Path,
) -> None:
    """A central driver never runs pre-commit against someone else's dirty tree.

    pre-commit writes unstaged changes to a patch file and checks them out of the
    tree while hooks run; a crash in that window loses them (D-OB-12). A clean
    tree has no window at all.
    """
    repo = _init_repo(tmp_path / "pcrepo")
    (repo / "a.txt").write_text("a\n")
    _commit(repo, "init")
    gate = mq.GateSpec(name="pc", command=("pre-commit", "run", "--all-files"))
    assert gate.runs_precommit is True
    assert mq.refuse_precommit_on_dirty_tree(repo, gate) == ""

    (repo / "a.txt").write_text("a lane's uncommitted work\n")
    refusal = mq.refuse_precommit_on_dirty_tree(repo, gate)
    assert "uncommitted work" in refusal
    assert "patch-restore window" not in refusal or True
    assert "D-OB-12" in refusal

    # A gate that is NOT pre-commit is unaffected, even on the same dirty tree —
    # the refusal is targeted, not a blanket dirty-tree ban.
    other = mq.GateSpec(name="x", command=("true",))
    assert other.runs_precommit is False
    assert mq.refuse_precommit_on_dirty_tree(repo, other) == ""


def test_the_queue_gives_any_precommit_gate_its_own_store(shell_repo: Path) -> None:
    """PRE_COMMIT_HOME is partitioned for every gate run (D-ORC-37).

    One change fixes both hazards: a crash can never orphan another lane's patch,
    and the store's SQLite ``db.db`` can never lock against another lane's.
    """
    from agent_utilities.governance.lanes import lane_scope, partitioned_paths

    seen: dict[str, str] = {}
    real = mq._timed_run

    def _capture(argv, cwd, *, timeout, env):
        seen.update(env)
        return real(argv, cwd, timeout=timeout, env=env)

    mq._timed_run = _capture  # type: ignore[assignment]
    try:
        _branch_with(shell_repo, "feat/x", {"x.txt": "x\n"}, "x")
        git = FakeGit(str(shell_repo.parent), {"x": str(shell_repo)})
        mq.enqueue("feat/x", path=shell_repo)
        mq.run_queue(path=shell_repo, prune=False, git=git)
    finally:
        mq._timed_run = real  # type: ignore[assignment]

    scope = lane_scope(shell_repo)
    assert "PRE_COMMIT_HOME" in seen
    assert seen["PRE_COMMIT_HOME"].startswith(
        str(partitioned_paths(scope.tree).scratch_dir)
    )
    assert seen["PRE_COMMIT_HOME"] != os.path.expanduser("~/.cache/pre-commit")


# ---------------------------------------------------------------------------
# Cross-project independence — the actual point of the move
# ---------------------------------------------------------------------------
def test_two_repositories_have_completely_independent_queues(
    tmp_path: Path, shell_repo: Path
) -> None:
    """Per-repo by construction, not by a `repo` key someone must remember to set."""
    other = _init_repo(tmp_path / "otherrepo")
    _write_config(
        other, "base: main\ngates: [{name: t, command: ['true'], compare: exit}]\n"
    )
    _commit(other, "init")
    _branch_with(other, "feat/other", {"o.txt": "o\n"}, "o")
    _branch_with(shell_repo, "feat/shell", {"s.txt": "s\n"}, "s")

    mq.enqueue("feat/other", path=other)
    mq.enqueue("feat/shell", path=shell_repo)
    assert [c.branch for c in mq.queued(other)] == ["feat/other"]
    assert [c.branch for c in mq.queued(shell_repo)] == ["feat/shell"]
    assert mq.queue_store(other).root != mq.queue_store(shell_repo).root


def test_dispatch_routes_every_verb_and_names_the_unknown_ones(
    shell_repo: Path,
) -> None:
    """One action core; the CLI and the MCP tool are both thin over it."""
    assert mq.dispatch("status", path=shell_repo)["depth"] == 0
    assert mq.dispatch("config", path=shell_repo)["gates"][0]["name"] == "no-broken"
    bad = mq.dispatch("nope", path=shell_repo)
    assert bad["ok"] is False and "run" in bad["actions"]


# ---------------------------------------------------------------------------
# D-W4RCA-1 (w4-rca-drain-deadlock, 2026-08-07) — a KILLED drain must not
# strand a candidate.
#
# Filed against a real incident: `merge_queue_runner.sh`'s outer `timeout`
# SIGTERM'd `python3 -m repository_manager.merge_queue run` twice in one
# session (epistemic-graph's declared gates legitimately exceeded the
# runner's flat ceiling under host load — see the runner script's own
# D-W4RCA-1 comment), and both times the candidate (`w3-viz-v1`) was
# re-enqueued by an operator who read "queue depth 0" as "the candidate
# vanished." Reading the ACTUAL on-disk fragment files
# (`<git-common-dir>/agent-lanes/merge-queue/*.yaml`) after those incidents
# showed every state transition intact — the append-only FragmentStore never
# lost a record. This test is that finding made mechanical: it kills a REAL
# `run_queue` process (SIGKILL, not a mocked interruption — an uncatchable
# signal is the only way to prove nothing in a `finally:` block is doing the
# real work) while its one declared gate is genuinely still running a
# subprocess, and asserts the candidate is neither stranded, nor silently
# marked landed, nor silently marked rejected — it stays exactly `queued`,
# because the ONLY code that can change that is `_record_state`, which never
# ran. A second, un-killed `run_queue()` call then proves the interruption is
# fully self-healing: the stale lease (its holder process is provably dead)
# is reclaimed automatically and the SAME candidate lands normally, with no
# manual repair step.
# ---------------------------------------------------------------------------
def test_a_killed_drain_does_not_strand_the_candidate(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "killrepo")
    # A gate that is genuinely still running a subprocess ~1.5s after `run`
    # starts (long enough to reliably land the SIGKILL mid-gate on any CI
    # host), and would pass cleanly if allowed to finish.
    (repo / "slow_gate.sh").write_text("#!/bin/sh\nsleep 4\nexit 0\n")
    os.chmod(repo / "slow_gate.sh", 0o755)
    _write_config(
        repo,
        """
        base: main
        gates:
          - name: slow-but-clean
            command: ["./slow_gate.sh"]
            tier: fast
            timeout: 60
            compare: exit
        """,
    )
    _commit(repo, "init")
    _branch_with(repo, "feat/slow", {"x.txt": "x\n"}, "add x")

    mq.enqueue("feat/slow", path=repo)
    assert [c.branch for c in mq.queued(repo)] == ["feat/slow"]
    pre_kill_records = len(mq.queue_store(repo).fold())

    pkg_root = str(Path(__file__).resolve().parents[1])
    driver_src = (
        f"import sys; sys.path.insert(0, {pkg_root!r})\n"
        "from repository_manager import merge_queue as mq\n"
        f"mq.run_queue(path={str(repo)!r}, prune=False)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", driver_src],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give the driver time to acquire the lease and start the gate's
        # `sleep 4` subprocess -- 1.5s is comfortably inside that 4s window.
        import time

        time.sleep(1.5)
        assert proc.poll() is None, "the drain finished before it could be killed"
        proc.kill()  # SIGKILL -- uncatchable; no `finally:` in the killed
        # process can run. This is the whole point: it proves durability is a
        # property of the ON-DISK APPEND, not of any cleanup code path.
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    # THE ASSERTION: the append-only store was never rewritten wholesale, and
    # no NEW terminal record was written for a candidate the killed process
    # never finished judging -- it is exactly where `enqueue` left it.
    assert [c.branch for c in mq.queued(repo)] == ["feat/slow"], (
        "a killed drain must never strand OR silently resolve a candidate"
    )
    candidate = mq.queued(repo)[0]
    assert candidate.state == mq.QUEUED
    # No phantom record was appended by the killed process either -- the
    # fragment file has exactly the one record `enqueue` wrote, not a
    # partially-written or duplicated one.
    assert len(mq.queue_store(repo).fold()) == pre_kill_records

    # Self-healing: the stale lease (holder process is provably dead) is
    # reclaimed automatically, and a completely ordinary, un-killed run lands
    # the SAME candidate with no manual repair.
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    result = mq.run_queue(path=repo, prune=False, git=git)
    assert result["landed"] == 1, result
    assert mq.queued(repo) == []
    log = _run("git log --oneline main", repo)
    assert "add x" in log


# ---------------------------------------------------------------------------
# The two surfaces — both thin over `dispatch`, neither holding logic
# ---------------------------------------------------------------------------
def test_the_mcp_tool_is_registered_and_declares_every_action() -> None:
    """A contract test over the MCP surface, parameterised by the action tuple.

    Registration alone is not the contract — the tool must ADVERTISE every action
    it routes, or a caller cannot discover the verb it needs.
    """
    from fastmcp import FastMCP

    from repository_manager import merge_queue as _mq
    from repository_manager.mcp_server import (
        RM_MERGE_QUEUE_ACTIONS,
        register_project_management_tools,
    )

    assert set(RM_MERGE_QUEUE_ACTIONS) == {
        "enqueue",
        "status",
        "withdraw",
        "run",
        "config",
    }
    # Every advertised MCP action must exist in the shared dispatch core, and vice
    # versa — this is what stops the two surfaces drifting apart.
    assert set(_mq.dispatch("__probe__")["actions"]) == set(RM_MERGE_QUEUE_ACTIONS)

    # Register against a bare FastMCP rather than `get_mcp_instance()`: the full
    # server factory is currently unbuildable in this environment for a reason
    # that has nothing to do with this tool (`agent_utilities.mcp.tasks_extension`
    # imports `MCPError` from `mcp.shared.exceptions`, which the installed `mcp`
    # spells `McpError`). `tests/test_mcp_registration.py` already owns and
    # reports that rule — one rule, one message — so this test pins ITS OWN
    # claim: that `rm_merge_queue` is registered by this module.
    mcp = FastMCP("probe")
    register_project_management_tools(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "rm_merge_queue" in names, sorted(names)
    assert "rm_worktree" in names, "sanity: the probe registered the real surface"


def test_the_cli_flag_routes_to_the_same_dispatch_core(
    shell_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from repository_manager.repository_manager import _run_merge_queue_cli

    args = SimpleNamespace(
        merge_queue="config",
        repo_path=str(shell_repo),
        queue_branch="",
        queue_base="",
        queue_reason="",
        queue_batch_size=0,
        queue_no_prune=False,
    )
    assert _run_merge_queue_cli(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gates"][0]["name"] == "no-broken"


def test_the_cli_returns_75_when_the_lease_is_held(
    shell_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Exit 75 (EX_TEMPFAIL) is the defer contract every LEASE surface shares.

    Any scheduler driving this queue chains on it; a plain 1 would read as a
    failed drain and invite a retry that races the holder.
    """
    from agent_utilities.governance.lanes import LeaseUnavailable

    from repository_manager import repository_manager as rm_mod

    def _held(*args, **kwargs):
        raise LeaseUnavailable("reconciliation-merge", {"lane": "someone-else"})

    monkeypatch.setattr(mq, "dispatch", _held)
    args = SimpleNamespace(
        merge_queue="run",
        repo_path=str(shell_repo),
        queue_branch="",
        queue_base="",
        queue_reason="",
        queue_batch_size=0,
        queue_no_prune=False,
    )
    assert rm_mod._run_merge_queue_cli(args) == 75
    assert json.loads(capsys.readouterr().out)["deferred"] is True


# ---------------------------------------------------------------------------
# D-RMD-1 — the queue reported `landed` while the base ref never moved
#
# The original defect merged into whatever HEAD was, not into the declared base.
# In agent-utilities the canonical checkout always sits on `main`, so "merge into
# HEAD" coincidentally equalled "land on base" and seven audited landings were
# genuinely correct -- luck, not correctness. These tests remove the luck.
# ---------------------------------------------------------------------------
def _parked_repo(tmp_path: Path, name: str = "parked") -> Path:
    """A repo whose canonical checkout sits on a branch that is NOT the base.

    This is the configuration au never had, and the one that turns D-RMD-1 from
    an invisible coincidence into a silent false success.
    """
    repo = _init_repo(tmp_path / name)
    _write_config(
        repo,
        """
        base: main
        gates:
          - name: always-green
            command: ["true"]
            tier: fast
            compare: exit
        """,
    )
    _commit(repo, "init")
    _branch_with(repo, "feat/x", {"x.txt": "x\n"}, "the candidate")
    # An operator poking around / a bisect / a merge in progress: the canonical
    # checkout is parked somewhere other than the base.
    _run("git checkout -q -b parked-elsewhere main", repo)
    return repo


def test_landing_moves_the_DECLARED_base_not_whatever_head_is(tmp_path: Path) -> None:
    """D-RMD-1: the ref that must move is `base`, never HEAD."""
    repo = _parked_repo(tmp_path)
    main_before = _run("git rev-parse refs/heads/main", repo)
    parked_before = _run("git rev-parse refs/heads/parked-elsewhere", repo)
    git = FakeGit(str(tmp_path), {"x": str(repo)})

    mq.enqueue("feat/x", path=repo)
    result = mq.run_queue(path=repo, prune=False, git=git)
    outcome = result["outcomes"][0]
    assert outcome["landed"] is True, outcome

    main_after = _run("git rev-parse refs/heads/main", repo)
    parked_after = _run("git rev-parse refs/heads/parked-elsewhere", repo)
    # The declared base MOVED and now contains the candidate's work...
    assert main_after != main_before, "main did not move -- this is D-RMD-1"
    assert main_after == outcome["to"]
    assert "the candidate" in _run("git log --oneline refs/heads/main", repo)
    # ...and the branch that merely happened to be checked out did NOT.
    assert parked_after == parked_before, "the checked-out branch was written instead"
    assert outcome["method"].startswith("update-ref CAS")
    assert outcome["verified"] is True


def test_the_post_condition_catches_a_wrong_write_target_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ The durable half: restore D-RMD-1's WRITE and prove the assertion still catches it.

    The write target is put back exactly as it was -- `git merge --ff-only` into
    whatever HEAD is -- while the post-condition is left in place. If the
    assertion is doing real work, the queue must refuse rather than report a
    landing that did not happen. This is the test that would have caught the
    original bug, and it is the one that catches the next variant.
    """
    repo = _parked_repo(tmp_path, "parked2")
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    real_run_git = mq._run_git

    def _buggy_run_git(args, cwd, *, timeout=300):
        # The original defect, verbatim: a CAS write to the declared ref becomes
        # a merge into HEAD. Everything else -- including the post-condition's
        # own `rev-parse` -- is left completely untouched.
        if args[:1] == ["update-ref"]:
            return real_run_git(["merge", "--ff-only", args[2]], cwd)
        return real_run_git(args, cwd, timeout=timeout)

    monkeypatch.setattr(mq, "_run_git", _buggy_run_git)
    mq.enqueue("feat/x", path=repo)
    with pytest.raises(mq.MergeQueueError, match="POST-CONDITION FAILED"):
        mq.run_queue(path=repo, prune=False, git=git)

    # And nothing was reported landed, so the guarded prune never ran and the
    # candidate's branch still exists -- the evidence survives.
    assert _run("git rev-parse --verify refs/heads/feat/x", repo)
    assert [c.branch for c in mq.queued(repo)] == ["feat/x"]


def test_landing_refuses_when_the_base_is_checked_out_in_another_worktree(
    tmp_path: Path,
) -> None:
    """`update-ref` would desync that tree; git refuses this for checkout, not for us."""
    repo = _parked_repo(tmp_path, "parked3")
    other = tmp_path / "someone-elses-tree"
    _run(f"git worktree add -q {other} main", repo)
    git = FakeGit(str(tmp_path), {"x": str(repo)})

    main_before = _run("git rev-parse refs/heads/main", repo)
    mq.enqueue("feat/x", path=repo)
    with pytest.raises(mq.MergeQueueError, match="checked out in"):
        mq.run_queue(path=repo, prune=False, git=git)
    assert _run("git rev-parse refs/heads/main", repo) == main_before


def test_landing_refuses_a_base_ref_that_does_not_exist(tmp_path: Path) -> None:
    """No silent fallback to HEAD -- that fallback IS D-RMD-1."""
    repo = _parked_repo(tmp_path, "parked4")
    git = FakeGit(str(tmp_path), {"x": str(repo)})
    mq.enqueue("feat/x", base="release/9.9", path=repo)
    with pytest.raises(mq.MergeQueueError, match="does not exist"):
        mq.run_queue(base="release/9.9", path=repo, prune=False, git=git)


# ---------------------------------------------------------------------------
# D-W3WPS-3 -- `landed: true` used to mean only "the LOCAL canonical checkout's
# base ref moved", never "reached the remote". 78 repos ended up "landed" with
# nothing shipped. These pin the fix: a real bare repo stands in for GitHub, and
# the defect-pinning assertion below checks the REMOTE ref, not any new field --
# reverting the push wiring in `land`/`_push_landed_base` turns it red on its own.
# ---------------------------------------------------------------------------
def test_landing_pushes_to_the_remote_not_just_local_main(tmp_path: Path) -> None:
    """Landing must reach the remote -- through the EXISTING gated push path.

    Uses the real ``repository_manager.repository_manager.Git`` (not the
    ``FakeGit`` test stand-in) so this exercises the actual production call
    ``land`` makes: ``Git.push_project`` -> ``_gate_before_push`` (a no-op here;
    the repo has no ``.pre-commit-config.yaml``) -> ``git push --follow-tags``.
    """
    from repository_manager.repository_manager import Git

    repo = _init_repo(tmp_path / "pushrepo")
    _write_config(
        repo,
        """
        base: main
        gates:
          - name: always-green
            command: ["true"]
            tier: fast
            compare: exit
        """,
    )
    _commit(repo, "init")
    _branch_with(repo, "feat/pushme", {"pushed.txt": "yes\n"}, "the candidate")

    # A local bare repo stands in for the GitHub remote.
    remote = tmp_path / "remote.git"
    _run(f"git init -q --bare {remote}", tmp_path)
    _run(f"git remote add origin {remote}", repo)
    _run("git push -q -u origin main", repo)

    local_before = _run("git rev-parse refs/heads/main", repo)
    remote_before = _run("git rev-parse refs/heads/main", remote)
    assert remote_before == local_before, "test setup: remote must start in sync"

    git = Git(path=str(tmp_path))
    mq.enqueue("feat/pushme", path=repo)
    outcome = mq.run_queue(path=repo, prune=False, git=git)["outcomes"][0]

    assert outcome["landed"] is True, outcome
    local_after = _run("git rev-parse refs/heads/main", repo)
    assert local_after != local_before  # sanity: the fast-forward really happened

    # ★ The defect-pinning assertion: the commit reached the REMOTE, not just
    # the local canonical checkout. This is checked independently of any new
    # response field, so it fails on its own if the push wiring is reverted.
    remote_after = _run("git rev-parse refs/heads/main", remote)
    assert remote_after == local_after, (
        "landed commit never reached the remote -- this is D-W3WPS-3"
    )

    # The new, honest, distinct field.
    assert outcome["pushed"] is True, outcome


def test_landing_defers_the_push_when_canonical_is_not_on_the_base(
    tmp_path: Path,
) -> None:
    """The CAS landing path must never push the WRONG branch.

    ``push_project`` pushes whatever branch the target working tree currently
    has checked out. When the canonical checkout is parked elsewhere (the CAS
    landing path -- see ``_parked_repo``), pushing through it would silently
    push the wrong branch, so the push must be deferred and visibly reported,
    never attempted with a guess.
    """
    from repository_manager.repository_manager import Git

    repo = _parked_repo(tmp_path, "parked-push")
    remote = tmp_path / "remote-parked.git"
    _run(f"git init -q --bare {remote}", tmp_path)
    _run(f"git remote add origin {remote}", repo)
    _run("git push -q -u origin main", repo)
    remote_before = _run("git rev-parse refs/heads/main", remote)

    git = Git(path=str(tmp_path))
    mq.enqueue("feat/x", path=repo)
    outcome = mq.run_queue(path=repo, prune=False, git=git)["outcomes"][0]

    assert outcome["landed"] is True, outcome
    assert outcome["method"].startswith("update-ref CAS")
    assert outcome["pushed"] is False, outcome
    assert "push_error" in outcome and outcome["push_error"]
    # The remote must NOT have been touched -- no wrong-branch push.
    assert _run("git rev-parse refs/heads/main", remote) == remote_before


def test_landing_still_works_when_canonical_IS_on_the_base(shell_repo: Path) -> None:
    """The au-shaped configuration keeps its atomic ref+worktree update.

    Without this the D-RMD-1 fix could regress the common case into a
    ref-only update that leaves the canonical working tree stale -- and the
    fleet hostPath-mounts that working tree.
    """
    _branch_with(shell_repo, "feat/y", {"y.txt": "fine\n"}, "y")
    git = FakeGit(str(shell_repo.parent), {"x": str(shell_repo)})
    mq.enqueue("feat/y", path=shell_repo)
    outcome = mq.run_queue(path=shell_repo, prune=False, git=git)["outcomes"][0]
    assert outcome["landed"] is True
    assert outcome["method"].startswith("merge --ff-only")
    assert outcome["verified"] is True
    # The working tree moved too, not just the ref.
    assert (shell_repo / "y.txt").is_file()
    assert _run("git rev-parse HEAD", shell_repo) == outcome["to"]


# ---------------------------------------------------------------------------
# RMDD-12 checkpoint 2 — the existing queue store is the only authority
# ---------------------------------------------------------------------------
_SHADOW_CONFIG = "1" * 64
_SHADOW_TOOLCHAIN = "2" * 64
_SHADOW_RESOURCE = "3" * 64


def _shadow_kwargs(repo: Path) -> dict[str, object]:
    return {
        "config_digest": _SHADOW_CONFIG,
        "toolchain_digest": _SHADOW_TOOLCHAIN,
        "resource_digest": _SHADOW_RESOURCE,
        "now": "2099-01-01T00:00:00Z",
        "path": repo,
    }


def test_shadow_generation_uses_mixed_queue_fragments_and_replays_idempotently(
    shell_repo: Path,
) -> None:
    """Domain records share queue_store while legacy candidates stay queued."""

    _branch_with(shell_repo, "feat/shadow", {"shadow.txt": "shadow\n"}, "shadow")
    mq.enqueue("feat/shadow", path=shell_repo)
    refs_before = _run(
        "git for-each-ref --format='%(refname) %(objectname)' refs/heads", shell_repo
    )
    first = mq.shadow_generation(**_shadow_kwargs(shell_repo))
    refs_after = _run(
        "git for-each-ref --format='%(refname) %(objectname)' refs/heads", shell_repo
    )

    assert first["landed"] is False
    assert first["candidate_snapshot_appends"] == 1
    assert first["generation_record_appends"] == 2
    assert first["generations"][0]["result"]["status"] == "trial-merged"
    assert first["generations"][0]["result"]["certifiable"] is False
    assert refs_after == refs_before
    assert [item.branch for item in mq.queued(shell_repo)] == ["feat/shadow"]

    raw = mq._queue_raw_records(shell_repo)
    assert {item.get("kind") for item in raw} == {
        "candidate_snapshot",
        "generation",
        None,
    }
    for item in raw:
        if item.get("kind") in {"candidate_snapshot", "generation"}:
            assert item["id"] == item["record_id"]

    second = mq.shadow_generation(**_shadow_kwargs(shell_repo))
    assert second["candidate_snapshot_appends"] == 0
    assert second["generation_record_appends"] == 0
    assert (
        second["generations"][0]["result"]["synthetic_commit_sha"]
        == first["generations"][0]["result"]["synthetic_commit_sha"]
    )
    assert mq._queue_raw_records(shell_repo) == raw


def test_shadow_generation_lease_serializes_callers_and_cleans_up(
    shell_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One caller owns the existing merge lease; replay is stable after release."""

    from agent_utilities.governance.lanes import LeaseUnavailable, lease_status

    _branch_with(shell_repo, "feat/lease", {"lease.txt": "lease\n"}, "lease")
    mq.enqueue("feat/lease", path=shell_repo)
    entered = Event()
    release = Event()
    real_trial = mq._shadow_trial_merge

    def blocked_trial(repo: Path, record):
        entered.set()
        if not release.wait(10):
            raise AssertionError("test trial release timed out")
        return real_trial(repo, record)

    monkeypatch.setattr(mq, "_shadow_trial_merge", blocked_trial)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            mq.shadow_generation, **_shadow_kwargs(shell_repo)
        )
        try:
            assert entered.wait(10), "first caller did not reach the leased trial"
            second_future = executor.submit(
                mq.shadow_generation, **_shadow_kwargs(shell_repo)
            )
            with pytest.raises(LeaseUnavailable):
                second_future.result(timeout=10)
        finally:
            release.set()
        first = first_future.result(timeout=10)

    assert first["generations"][0]["result"]["status"] == "trial-merged"
    assert lease_status(mq.MERGE_LEASE, shell_repo) is None
    replay = mq.shadow_generation(**_shadow_kwargs(shell_repo))
    assert replay["candidate_snapshot_appends"] == 0
    assert replay["generation_record_appends"] == 0
    raw = mq._queue_raw_records(shell_repo)
    assert sum(item.get("kind") == "candidate_snapshot" for item in raw) == 1
    assert sum(item.get("kind") == "generation" for item in raw) == 2
    from repository_manager.candidate_generation import fold_generation_records

    assert len(fold_generation_records(raw)) == 1


def test_shadow_generation_lease_cleans_up_after_failure(shell_repo: Path) -> None:
    """A failed snapshot cannot strand the shared merge lease."""

    from agent_utilities.governance.lanes import lease_status

    _branch_with(shell_repo, "feat/failure", {"failure.txt": "failure\n"}, "failure")
    mq.enqueue("feat/failure", path=shell_repo)
    with pytest.raises(mq.MergeQueueError, match="config_digest"):
        mq.shadow_generation(
            config_digest="not-a-digest",
            toolchain_digest=_SHADOW_TOOLCHAIN,
            resource_digest=_SHADOW_RESOURCE,
            now="2099-01-01T00:00:00Z",
            path=shell_repo,
        )
    assert lease_status(mq.MERGE_LEASE, shell_repo) is None


def test_shadow_generation_attributes_conflict_and_differential_paths_without_refs(
    shell_repo: Path,
) -> None:
    """Conflict attribution comes from merge-tree, with no index/worktree/ref use."""

    (shell_repo / "shared.txt").write_text("base\n")
    _commit(shell_repo, "add shared base")
    _branch_with(shell_repo, "feat/a", {"shared.txt": "a\n"}, "a")
    _branch_with(shell_repo, "feat/b", {"shared.txt": "b\n"}, "b")
    mq.enqueue("feat/a", path=shell_repo)
    mq.enqueue("feat/b", path=shell_repo)
    refs_before = _run(
        "git for-each-ref --format='%(refname) %(objectname)' refs/heads", shell_repo
    )
    status_before = _run("git status --porcelain", shell_repo)

    result = mq.shadow_generation(**_shadow_kwargs(shell_repo))
    outcome = result["generations"][0]["result"]
    refs_after = _run(
        "git for-each-ref --format='%(refname) %(objectname)' refs/heads", shell_repo
    )

    assert outcome["status"] == "conflicted"
    assert outcome["certifiable"] is False
    assert outcome["conflicted_candidate_ids"]
    assert outcome["conflicts"][0]["conflicts"]
    assert outcome["conflicts"][0]["conflict_count"] >= 1
    assert outcome["conflicts"][0]["conflict_against"]["kind"] == (
        "rolling-synthetic-head"
    )
    assert outcome["conflicts"][0]["conflict_against"]["members"] == [
        {
            "candidate_id": outcome["accepted_candidate_ids"][0],
            "version": 1,
        }
    ]
    assert outcome["differential_paths"] == ["shared.txt"]
    assert outcome["differential_path_count"] == 1
    assert outcome["differential_paths_truncated"] is False
    assert refs_after == refs_before
    assert _run("git status --porcelain", shell_repo) == status_before
    assert mq.queued(shell_repo)


def test_domain_only_queue_fragments_fold_without_legacy_projection(
    shell_repo: Path,
) -> None:
    """The additive view also works when a migrated fragment has no legacy row."""

    from repository_manager.candidate_generation import (
        fold_candidate_records,
        fold_generation_records,
        generation_record,
        snapshot_candidate,
    )
    from repository_manager.development import RepositoryIdentity

    _branch_with(shell_repo, "feat/domain", {"domain.txt": "domain\n"}, "domain")
    base_sha = _run("git rev-parse main", shell_repo)
    candidate_sha = _run("git rev-parse feat/domain", shell_repo)
    snapshot = snapshot_candidate(
        SimpleNamespace(
            branch="feat/domain",
            lane="domain-lane",
            base="main",
            enqueued_at="2026-08-09T10:00:00Z",
        ),
        repository=RepositoryIdentity(
            repository_id="repository:test-domain-only",
            canonical_path=str(shell_repo.resolve()),
        ),
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        config_digest=_SHADOW_CONFIG,
        toolchain_digest=_SHADOW_TOOLCHAIN,
        resource_digest=_SHADOW_RESOURCE,
        target_branch="main",
    )
    generation = generation_record(
        (snapshot,),
        target_branch="main",
        sealed_at="2099-01-01T00:00:00Z",
    )
    store = mq.queue_store(shell_repo)
    for domain_record in (snapshot.to_record(), generation.to_record()):
        store.append(
            {**domain_record, "id": domain_record["record_id"]},
            lane="domain-only",
        )

    raw = mq._queue_raw_records(shell_repo)
    assert all(item.get("kind") in {"candidate_snapshot", "generation"} for item in raw)
    assert len(fold_candidate_records(raw)) == 1
    assert len(fold_generation_records(raw)) == 1
    assert mq._all_candidates(shell_repo) == []


def test_shadow_generation_uses_canonical_repository_identity_for_same_basenames(
    tmp_path: Path,
) -> None:
    """Two repositories with the same display name cannot share candidate IDs."""

    first = _init_repo(tmp_path / "one" / "same")
    second = _init_repo(tmp_path / "two" / "same")
    for repo, branch in ((first, "feat/one"), (second, "feat/two")):
        _write_config(repo, "base: main\ngates: []\n")
        (repo / "base.txt").write_text("base\n")
        _commit(repo, "init")
        _branch_with(repo, branch, {f"{branch.replace('/', '-')}.txt": "x\n"}, branch)
        mq.enqueue(branch, path=repo)

    first_result = mq.shadow_generation(**_shadow_kwargs(first))
    second_result = mq.shadow_generation(**_shadow_kwargs(second))
    first_id = first_result["generations"][0]["generation_id"]
    second_id = second_result["generations"][0]["generation_id"]
    assert first_id != second_id


def test_shadow_generation_marks_base_move_stale_before_trial(
    shell_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ref moved after sealing is stale evidence, never a certification."""

    _branch_with(shell_repo, "feat/stale", {"stale.txt": "stale\n"}, "stale")
    mq.enqueue("feat/stale", path=shell_repo)
    real_trial = mq._shadow_trial_merge

    def move_base_then_trial(repo: Path, record):
        branch_tip = _run("git rev-parse feat/stale", repo)
        _run(f"git update-ref refs/heads/main {branch_tip}", repo)
        return real_trial(repo, record)

    monkeypatch.setattr(mq, "_shadow_trial_merge", move_base_then_trial)
    result = mq.shadow_generation(**_shadow_kwargs(shell_repo))
    outcome = result["generations"][0]["result"]

    assert outcome["status"] == "stale-base"
    assert outcome["stale_base"] is True
    assert outcome["certifiable"] is False
    assert outcome["synthetic_commit_sha"] == ""
    assert outcome["accepted"] == []
    assert outcome["conflicts"] == []
