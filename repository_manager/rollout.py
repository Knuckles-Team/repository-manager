"""Deterministic, resumable dependency rollout across a fleet (RM-RELEASE-R009).

One command raises a dependency floor in every selected repository through a
pull request, never by pushing to ``main``:

1. **select** -- workspace manifest group (a selector name or a subdirectory
   prefix) intersected with the forge's non-archived repositories and with the
   repositories whose ``pyproject.toml`` on the default branch declares the
   dependency;
2. **plan** -- per repository, read the current floor, the version and any
   rollout pull request; record a row in an append-only JSONL ledger keyed by
   ``(dependency, floor, repo)``;
3. **apply** -- in a fresh ``git worktree add`` off ``origin/<default>``: set
   ``>=floor,<next-minor``, relock and verify the lock resolved the floor,
   patch-bump through the bumpversion config (plus Dockerfile pins), add a
   CHANGELOG entry, regenerate spec status, run pre-commit, commit with the
   repository's configured identity, push, open the PR, arm
   ``gh pr merge --auto --rebase``;
4. **status** -- merged / failing (with failing check names) / pending;
5. **release** -- once the PR merged and main's floor meets the target, tag
   ``v<version>`` on main's HEAD unless already tagged.

Every step is idempotent: a re-run resumes from the ledger and the forge, so it
never opens a duplicate pull request and never bumps a version twice. All
process calls go through an injectable :class:`Runner` so tests use fakes and
no network. Only ``pathlib``/``subprocess`` are used (Windows-portable).
"""

from __future__ import annotations

import concurrent.futures as cf
import configparser
import datetime as _dt
import json
import os
import re
import subprocess
import threading
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

TERMINAL_OK = {"pr_open", "merged", "released", "skipped"}
FAILING_CONCLUSIONS = {
    "FAILURE",
    "TIMED_OUT",
    "ERROR",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
}
TRANSIENT = re.compile(
    r"(NOTES\.md|HANDOFF|BASELINE|\.log$|\.venv/|__pycache__|\.bak$|\.orig$|\.rej$)"
)
DEFAULT_GROUP = "agent-connectors"
DEFAULT_JOBS = 6


# --------------------------------------------------------------------- process seam
@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


class Runner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> Completed: ...


def subprocess_runner(
    argv: Sequence[str], *, cwd: Path | None = None, timeout: int | None = None
) -> Completed:
    """Real runner: no shell, captured text output."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, no shell
            list(argv),
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Completed(127, "", f"{type(exc).__name__}: {exc}")
    return Completed(proc.returncode, proc.stdout, proc.stderr)


# ------------------------------------------------------------------- version helpers
def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3]) or (0,)


def next_minor(floor: str) -> str:
    parts = list(parse_version(floor)) + [0, 0]
    return f"{parts[0]}.{parts[1] + 1}.0"


def patch_bump(version: str) -> str:
    parts = version.split(".")
    parts[-1] = str(int(re.match(r"\d+", parts[-1]).group(0)) + 1)  # type: ignore[union-attr]
    return ".".join(parts)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dep_regex(dependency: str) -> re.Pattern[str]:
    name = r"[-_.]".join(re.escape(p) for p in re.split(r"[-_.]+", dependency))
    return re.compile(
        rf"""(?P<q>["'])(?P<name>{name})(?P<extras>\[[^\]]*\])?(?P<spec>\s*[<>=!~][^"';]*)?(?P<marker>;[^"']*)?(?P=q)""",
        re.IGNORECASE,
    )


def _requirements(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject.get("project", {}) or {}
    reqs = list(project.get("dependencies", []) or [])
    for group in (project.get("optional-dependencies", {}) or {}).values():
        reqs.extend(group or [])
    for group in (pyproject.get("dependency-groups", {}) or {}).values():
        reqs.extend(r for r in group or [] if isinstance(r, str))
    return reqs


def current_floor(pyproject_text: str, dependency: str) -> tuple[bool, str | None]:
    """Return (declares dependency, lowest ``>=`` floor declared)."""
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return False, None
    floors: list[str] = []
    found = False
    for req in _requirements(data):
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if not m or normalize(m.group(1)) != normalize(dependency):
            continue
        found = True
        f = re.search(r">=\s*([0-9][0-9.]*)", req)
        if f:
            floors.append(f.group(1))
    if not floors:
        return found, None
    return found, min(floors, key=parse_version)


def project_version(pyproject_text: str) -> str | None:
    try:
        return (tomllib.loads(pyproject_text).get("project") or {}).get("version")
    except tomllib.TOMLDecodeError:
        return None


def project_name(pyproject_text: str) -> str | None:
    try:
        return (tomllib.loads(pyproject_text).get("project") or {}).get("name")
    except tomllib.TOMLDecodeError:
        return None


def rewrite_floor(
    text: str, dependency: str, floor: str, extra: str | None = None
) -> str:
    """Set every quoted requirement of *dependency* to ``>=floor,<next-minor``."""
    spec = f">={floor},<{next_minor(floor)}"

    def sub(m: re.Match[str]) -> str:
        extras = m.group("extras") or ""
        if extra:
            have = [e.strip() for e in extras.strip("[]").split(",") if e.strip()]
            if extra not in have:
                have.append(extra)
            extras = f"[{','.join(have)}]"
        return f"{m.group('q')}{m.group('name')}{extras}{spec}{m.group('marker') or ''}{m.group('q')}"

    return _dep_regex(dependency).sub(sub, text)


def locked_version(lock_text: str, dependency: str) -> str | None:
    try:
        data = tomllib.loads(lock_text)
    except tomllib.TOMLDecodeError:
        return None
    for pkg in data.get("package", []) or []:
        if normalize(str(pkg.get("name", ""))) == normalize(dependency):
            return str(pkg.get("version"))
    return None


def bump_files(repo: Path, current: str, new: str, package: str | None) -> list[str]:
    """Patch-bump per the repository's bumpversion config plus Dockerfile pins.

    Returns warnings. ``pyproject.toml`` is always bumped.
    """
    warnings: list[str] = []
    edits: list[tuple[str, str, str]] = [
        ("pyproject.toml", f'version = "{current}"', f'version = "{new}"')
    ]
    for cfg_name in (".bumpversion.cfg", "setup.cfg"):
        cfg_path = repo / cfg_name
        if not cfg_path.is_file():
            continue
        cp = configparser.RawConfigParser()
        cp.read(cfg_path, encoding="utf-8")
        if not cp.has_section("bumpversion"):
            continue
        for sec in cp.sections():
            m = re.match(r"bumpversion:file(?:\([^)]*\))?:(.*)$", sec)
            if not m:
                continue
            search = cp.get(sec, "search", fallback="{current_version}")
            replace = cp.get(sec, "replace", fallback="{new_version}")
            edits.append(
                (
                    m.group(1).strip(),
                    search.replace("{current_version}", current),
                    replace.replace("{new_version}", new),
                )
            )
        edits.append(
            (cfg_name, f"current_version = {current}", f"current_version = {new}")
        )
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    bmv = (pyproject.get("tool") or {}).get("bumpversion") or {}
    for entry in bmv.get("files", []) or []:
        fname = entry.get("filename")
        if fname:
            edits.append(
                (
                    fname,
                    str(entry.get("search", "{current_version}")).replace(
                        "{current_version}", current
                    ),
                    str(entry.get("replace", "{new_version}")).replace(
                        "{new_version}", new
                    ),
                )
            )
    if bmv:
        edits.append(
            (
                "pyproject.toml",
                f'current_version = "{current}"',
                f'current_version = "{new}"',
            )
        )
    seen: set[tuple[str, str]] = set()
    for rel, search, replace in edits:
        if (rel, search) in seen:
            continue
        seen.add((rel, search))
        path = repo / rel
        if not path.is_file():
            warnings.append(f"bumpversion file missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if search not in text:
            if rel != "pyproject.toml" or "current_version" not in search:
                warnings.append(f"bumpversion search not found in {rel}")
            continue
        path.write_text(text.replace(search, replace), encoding="utf-8")
    if package:
        pin = re.compile(
            rf"({re.escape(package)}(?:\[[^\]]*\])?==){re.escape(current)}\b",
            re.IGNORECASE,
        )
        for docker in sorted(
            {*repo.glob("Dockerfile*"), *repo.glob("docker/*Dockerfile*")}
        ):
            text = docker.read_text(encoding="utf-8")
            new_text = pin.sub(rf"\g<1>{new}", text)
            if new_text != text:
                docker.write_text(new_text, encoding="utf-8")
    return warnings


def add_changelog_entry(repo: Path, version: str, line: str, today: str) -> None:
    path = repo / "CHANGELOG.md"
    entry = f"## [{version}] - {today}\n\n### Changed\n- {line}\n\n"
    if not path.is_file():
        path.write_text(f"# Changelog\n\n{entry}", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if f"## [{version}]" in text:
        return
    m = re.search(r"^## \[(?!Unreleased)", text, re.M)
    if m:
        text = text[: m.start()] + entry + text[m.start() :]
    else:
        text = text.rstrip("\n") + "\n\n" + entry
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- ledger
def default_ledger_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(
            os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
        )
    return base / "repository-manager" / "rollout" / "ledger.jsonl"


class Ledger:
    """Append-only JSONL; the latest record per (dependency, floor, repo) wins."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_ledger_path()
        self._lock = threading.Lock()

    def rows(self, dependency: str, floor: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.path.is_file():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("dependency") == dependency and rec.get("floor") == floor:
                out[rec["repo"]] = {**out.get(rec["repo"], {}), **rec}
        return out

    def record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- engine
@dataclass(frozen=True)
class FleetRepo:
    name: str
    owner: str
    identifier: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class RolloutConfig:
    dependency: str
    floor: str
    extra: str | None = None
    group: str = DEFAULT_GROUP
    jobs: int = DEFAULT_JOBS
    only: tuple[str, ...] = ()
    dry_run: bool = False
    spec: str = "none (dependency rollout)"
    trailers: tuple[str, ...] = ()
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    worktree_root: Path = field(
        default_factory=lambda: (
            Path.home() / ".local" / "state" / "repository-worktrees"
        )
    )

    @property
    def branch(self) -> str:
        return f"chore/rollout-{normalize(self.dependency)}-{self.floor}"

    @property
    def spec_range(self) -> str:
        return f">={self.floor},<{next_minor(self.floor)}"


def manifest_repos(manifest: dict[str, Any], group: str) -> list[FleetRepo]:
    """Repositories of a selector (or a subdirectory prefix) with GitHub owners."""
    from repository_manager.workspace_manifest import (
        _repository_entries,
        select_repositories,
    )

    urls: dict[str, str] = {}

    def walk(node: Any, parent: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        for repo in node.get("repositories", []) or []:
            url = str(repo.get("url", "")).strip().rstrip("/")
            urls["/".join((*parent, url.rsplit("/", 1)[-1].removesuffix(".git")))] = url
        for key, child in (node.get("subdirectories", {}) or {}).items():
            walk(child, (*parent, key))

    walk(manifest, ())
    if group in (manifest.get("selectors") or {}):
        identifiers, _, _ = select_repositories(manifest, selectors=[group])
    else:
        prefix = group.strip("/") + "/"
        identifiers = tuple(
            e.identifier
            for e in _repository_entries(manifest)
            if e.identifier.startswith(prefix)
        )
    out = []
    for ident in identifiers:
        m = re.match(
            r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?$", urls.get(ident, "")
        )
        if m:
            out.append(FleetRepo(name=m.group(2), owner=m.group(1), identifier=ident))
    return sorted(out, key=lambda r: r.name)


class Rollout:
    def __init__(
        self,
        config: RolloutConfig,
        *,
        runner: Runner = subprocess_runner,
        ledger: Ledger | None = None,
        today: Callable[[], str] | None = None,
    ) -> None:
        self.cfg = config
        self.run = runner
        self.ledger = ledger or Ledger()
        self.today = today or (lambda: _dt.date.today().isoformat())
        self._precommit_lock = threading.Lock()

    # ---------------------------------------------------------------- forge reads
    def _gh_json(self, argv: Sequence[str]) -> Any:
        res = self.run(["gh", *argv])
        if res.returncode:
            raise RuntimeError(
                res.output.strip()[-400:] or f"gh {' '.join(argv)} failed"
            )
        return json.loads(res.stdout or "null")

    def _main_file(self, repo: FleetRepo, path: str, ref: str) -> str | None:
        res = self.run(
            [
                "gh",
                "api",
                f"repos/{repo.slug}/contents/{path}?ref={ref}",
                "-H",
                "Accept: application/vnd.github.raw",
            ]
        )
        return res.stdout if res.returncode == 0 else None

    def _default_branch(self, repo: FleetRepo) -> str:
        res = self.run(["gh", "api", f"repos/{repo.slug}", "-q", ".default_branch"])
        return (res.stdout.strip() if res.returncode == 0 else "") or "main"

    def _rollout_pr(self, repo: FleetRepo) -> dict[str, Any] | None:
        prs = (
            self._gh_json(
                [
                    "pr",
                    "list",
                    "-R",
                    repo.slug,
                    "--head",
                    self.cfg.branch,
                    "--state",
                    "all",
                    "--json",
                    "number,url,state",
                ]
            )
            or []
        )
        for state in ("OPEN", "MERGED"):
            for pr in prs:
                if pr.get("state") == state:
                    return pr
        return None

    def live_repos(self, owners: Iterable[str]) -> set[str]:
        live: set[str] = set()
        for owner in sorted(set(owners)):
            res = self.run(
                [
                    "gh",
                    "repo",
                    "list",
                    owner,
                    "--no-archived",
                    "-L",
                    "2000",
                    "--json",
                    "name",
                    "-q",
                    ".[].name",
                ]
            )
            if res.returncode:
                raise RuntimeError(
                    f"cannot list repositories of {owner}: {res.output[-300:]}"
                )
            live.update(f"{owner}/{n}" for n in res.stdout.split())
        return live

    # -------------------------------------------------------------- 1+2 select/plan
    def select(self, manifest: dict[str, Any]) -> list[FleetRepo]:
        repos = manifest_repos(manifest, self.cfg.group)
        if self.cfg.only:
            repos = [r for r in repos if r.name in self.cfg.only]
        live = self.live_repos(r.owner for r in repos)
        return [r for r in repos if r.slug in live]

    def _row(self, repo: FleetRepo, **fields: Any) -> dict[str, Any]:
        return {
            "dependency": self.cfg.dependency,
            "floor": self.cfg.floor,
            "repo": repo.name,
            "slug": repo.slug,
            **fields,
        }

    def plan_one(
        self, repo: FleetRepo, prior: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        base = self._default_branch(repo)
        text = self._main_file(repo, "pyproject.toml", base)
        if text is None:
            return None
        declares, floor = current_floor(text, self.cfg.dependency)
        if not declares:
            return None
        row = self._row(
            repo,
            base=base,
            current_floor=floor,
            version=project_version(text),
            at=self.today(),
        )
        if prior and prior.get("state") in {"merged", "released"}:
            return {**prior, **{k: row[k] for k in ("current_floor", "version")}}
        pr = self._rollout_pr(repo)
        if pr:
            state = "merged" if pr["state"] == "MERGED" else "pr_open"
            return {**row, "state": state, "pr": pr["url"]}
        if floor and parse_version(floor) >= parse_version(self.cfg.floor):
            return {**row, "state": "skipped", "reason": f"floor already >={floor}"}
        return {**row, "state": "planned"}

    def plan(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        prior = self.ledger.rows(self.cfg.dependency, self.cfg.floor)
        repos = self.select(manifest)
        rows: list[dict[str, Any]] = []

        def one(repo: FleetRepo) -> dict[str, Any] | None:
            try:
                return self.plan_one(repo, prior.get(repo.name))
            except Exception as exc:  # noqa: BLE001 - becomes a failed row
                return self._row(repo, state="failed", error=f"plan: {exc}"[:500])

        with cf.ThreadPoolExecutor(max_workers=max(1, self.cfg.jobs)) as ex:
            for row in ex.map(one, repos):
                if row is None:
                    continue
                old = prior.get(row["repo"])
                if old and old.get("state") == "failed" and row["state"] == "planned":
                    row["previous_error"] = old.get("error")
                rows.append(row)
                if not self.cfg.dry_run and old != row:
                    self.ledger.record(row)
        return sorted(rows, key=lambda r: r["repo"])

    # ---------------------------------------------------------------------- 3 apply
    def apply(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self.plan(manifest)
        if self.cfg.dry_run:
            return rows
        todo = [r for r in rows if r["state"] in {"planned", "failed"}]
        done = {r["repo"]: r for r in rows if r not in todo}
        with cf.ThreadPoolExecutor(max_workers=max(1, self.cfg.jobs)) as ex:
            for row in ex.map(self._apply_safe, todo):
                self.ledger.record(row)
                done[row["repo"]] = row
        return [done[k] for k in sorted(done)]

    def _apply_safe(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.apply_one(row)
        except Exception as exc:  # noqa: BLE001 - one repo never stops the fleet
            return {
                **row,
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:800],
            }

    def _fail(self, row: dict[str, Any], error: str) -> dict[str, Any]:
        return {**row, "state": "failed", "error": error[-800:]}

    def _checkout(self, name: str) -> Path:
        root = self.cfg.workspace_root
        for cand in (
            root / "agent-packages" / "agents" / name,
            root / "agent-packages" / name,
            root / name,
        ):
            if (cand / ".git").exists():
                return cand
        clone = self.cfg.worktree_root.parent / "repository-rollout" / "clones" / name
        return clone

    def apply_one(self, row: dict[str, Any]) -> dict[str, Any]:
        cfg, run = self.cfg, self.run
        repo = FleetRepo(row["repo"], row["slug"].split("/")[0], row["repo"])
        # Never a duplicate PR: the forge is re-read right before mutating.
        pr = self._rollout_pr(repo)
        if pr:
            return {
                **row,
                "state": "merged" if pr["state"] == "MERGED" else "pr_open",
                "pr": pr["url"],
            }
        src = self._checkout(repo.name)
        if not (src / ".git").exists():
            res = run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    f"https://github.com/{repo.slug}.git",
                    str(src),
                ]
            )
            if res.returncode:
                return self._fail(row, "clone: " + res.output)
        base = row.get("base") or "main"
        if run(["git", "fetch", "-q", "origin", base], cwd=src, timeout=600).returncode:
            return self._fail(row, "fetch failed")
        remote_branch = run(
            ["git", "ls-remote", "--heads", "origin", cfg.branch], cwd=src
        ).stdout.strip()
        if remote_branch:
            # A previous run pushed but never opened the PR: open it, never re-bump.
            return self._open_pr(row, repo, src, base)
        wt = (
            cfg.worktree_root
            / repo.name
            / f"rollout-{normalize(cfg.dependency)}-{cfg.floor}"
        )
        if wt.exists():
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=src)
        run(["git", "branch", "-D", cfg.branch], cwd=src)
        wt.parent.mkdir(parents=True, exist_ok=True)
        res = run(
            ["git", "worktree", "add", str(wt), "-b", cfg.branch, f"origin/{base}"],
            cwd=src,
        )
        if res.returncode:
            return self._fail(row, "worktree add: " + res.output)
        try:
            return self._apply_in(row, repo, src, wt, base)
        finally:
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=src)

    def _apply_in(
        self, row: dict[str, Any], repo: FleetRepo, src: Path, wt: Path, base: str
    ) -> dict[str, Any]:
        cfg, run = self.cfg, self.run
        pp = wt / "pyproject.toml"
        text = pp.read_text(encoding="utf-8")
        new_text = rewrite_floor(text, cfg.dependency, cfg.floor, cfg.extra)
        if new_text == text:
            return self._fail(row, "pyproject: no requirement rewritten")
        pp.write_text(new_text, encoding="utf-8")
        res = run(
            ["uv", "lock", "--upgrade-package", cfg.dependency, "--refresh"],
            cwd=wt,
            timeout=1200,
        )
        if res.returncode:
            return self._fail(row, "uv lock failed: " + res.output)
        lock = wt / "uv.lock"
        got = (
            locked_version(lock.read_text(encoding="utf-8"), cfg.dependency)
            if lock.is_file()
            else None
        )
        if not got or parse_version(got) < parse_version(cfg.floor):
            return self._fail(
                row,
                f"uv.lock resolved {cfg.dependency} {got or 'nothing'}, need >={cfg.floor}",
            )
        current = project_version(new_text)
        if not current:
            return self._fail(row, "no [project].version in pyproject.toml")
        new_version = patch_bump(current)
        warnings = bump_files(wt, current, new_version, project_name(new_text))
        add_changelog_entry(
            wt,
            new_version,
            f"Require `{cfg.dependency}{cfg.spec_range}`; `uv.lock` relocked.",
            self.today(),
        )
        # Regenerate spec status so the PR carries the current status schema.
        if (wt / "specs").is_dir():
            res = run(["pipelines-hook", "spec-status", "--write"], cwd=wt, timeout=900)
            if res.returncode:
                return self._fail(
                    row, "pipelines-hook spec-status --write failed: " + res.output
                )
        if (wt / ".pre-commit-config.yaml").is_file():
            with self._precommit_lock:  # pre-commit's cache is not concurrency-safe
                ok, out = False, ""
                for _ in range(3):  # autofixing hooks settle on a re-run
                    res = run(
                        ["pre-commit", "run", "--all-files"], cwd=wt, timeout=1800
                    )
                    ok, out = res.returncode == 0, res.output
                    if ok:
                        break
            if not ok:
                return self._fail(row, "pre-commit failed: " + out)
        changed = self._changed_files(wt)
        bad = [f for f in changed if TRANSIENT.search(f)]
        if bad:
            return self._fail(row, "transient artifacts in diff: " + ", ".join(bad))
        if not changed:
            return self._fail(row, "nothing changed")
        name = run(["git", "config", "user.name"], cwd=src).stdout.strip()
        email = run(["git", "config", "user.email"], cwd=src).stdout.strip()
        if not name or not email or name.lower() == "claude":
            return self._fail(row, f"refusing commit identity {name!r}")
        run(["git", "add", "--", *changed], cwd=wt)
        subject = f"chore: require {cfg.dependency}>={cfg.floor} ({new_version})"
        msg = (
            f"{subject}\n\nRaise the {cfg.dependency} floor to {cfg.spec_range}, relock uv.lock "
            f"and patch-bump {current} -> {new_version}.\n\nSpec: {cfg.spec}"
        )
        if cfg.trailers:
            msg += "\n" + "\n".join(cfg.trailers)
        res = run(
            [
                "git",
                "-c",
                f"user.name={name}",
                "-c",
                f"user.email={email}",
                "commit",
                "-q",
                "-m",
                msg,
            ],
            cwd=wt,
        )
        if res.returncode:
            return self._fail(row, "commit failed: " + res.output)
        res = run(
            ["git", "push", "-q", "-u", "origin", cfg.branch], cwd=wt, timeout=900
        )
        if res.returncode:
            return self._fail(row, "push failed: " + res.output)
        row = {**row, "new_version": new_version, "warnings": warnings}
        return self._open_pr(row, repo, wt, base)

    def _changed_files(self, wt: Path) -> list[str]:
        out = self.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=wt
        ).stdout
        files = []
        for line in out.splitlines():
            path = line[3:].strip().split(" -> ")[-1].strip('"')
            if path:
                files.append(path)
        return sorted(set(files))

    def _open_pr(
        self, row: dict[str, Any], repo: FleetRepo, cwd: Path, base: str
    ) -> dict[str, Any]:
        cfg = self.cfg
        title = f"chore: require {cfg.dependency}>={cfg.floor}" + (
            f" ({row['new_version']})" if row.get("new_version") else ""
        )
        body = (
            f"Raise `{cfg.dependency}` to `{cfg.spec_range}`, relock `uv.lock`, patch-bump the "
            f"version and regenerate spec status.\n\nOpened by `repository-manager rollout`; the tag "
            f"is created by `repository-manager rollout release` after merge.\n\nSpec: {cfg.spec}"
        )
        res = self.run(
            [
                "gh",
                "pr",
                "create",
                "-R",
                repo.slug,
                "--base",
                base,
                "--head",
                cfg.branch,
                "--title",
                title,
                "--body",
                body,
            ],
            cwd=cwd,
        )
        if res.returncode:
            return self._fail(row, "gh pr create failed: " + res.output)
        url = res.stdout.strip().splitlines()[-1]
        merge = self.run(["gh", "pr", "merge", url, "--auto", "--rebase"], cwd=cwd)
        out = {
            **row,
            "state": "pr_open",
            "pr": url,
            "auto_merge": merge.returncode == 0,
        }
        out.pop("error", None)
        if merge.returncode:
            out["auto_merge_error"] = merge.output[-300:]
        return out

    # --------------------------------------------------------------------- 4 status
    def status(self) -> list[dict[str, Any]]:
        out = []
        for name, row in sorted(
            self.ledger.rows(self.cfg.dependency, self.cfg.floor).items()
        ):
            report = {"repo": name, "state": row.get("state"), "pr": row.get("pr")}
            if row.get("pr") and row.get("state") in {"pr_open", "merged"}:
                info = (
                    self._gh_json(
                        ["pr", "view", row["pr"], "--json", "state,statusCheckRollup"]
                    )
                    or {}
                )
                failing = sorted(
                    c.get("name") or c.get("context") or "?"
                    for c in info.get("statusCheckRollup") or []
                    if (c.get("conclusion") or c.get("state") or "").upper()
                    in FAILING_CONCLUSIONS
                )
                if info.get("state") == "MERGED":
                    report["status"] = "merged"
                elif info.get("state") == "CLOSED":
                    report["status"] = "closed"
                elif failing:
                    report["status"], report["failing_checks"] = "failing", failing
                else:
                    report["status"] = "pending"
            elif row.get("state") == "failed":
                report["status"], report["error"] = "failed", row.get("error")
            else:
                report["status"] = row.get("state")
            out.append(report)
        return out

    # -------------------------------------------------------------------- 5 release
    def release(self) -> list[dict[str, Any]]:
        out = []
        for name, row in sorted(
            self.ledger.rows(self.cfg.dependency, self.cfg.floor).items()
        ):
            if row.get("state") == "released" or not row.get("pr"):
                continue
            repo = FleetRepo(name, row["slug"].split("/")[0], name)
            info = self._gh_json(["pr", "view", row["pr"], "--json", "state"]) or {}
            if info.get("state") != "MERGED":
                if row.get("state") != "pr_open":
                    self.ledger.record({**row, "state": "pr_open"})
                continue
            base = row.get("base") or "main"
            text = self._main_file(repo, "pyproject.toml", base) or ""
            _, floor = current_floor(text, self.cfg.dependency)
            version = project_version(text)
            if (
                not floor
                or parse_version(floor) < parse_version(self.cfg.floor)
                or not version
            ):
                self.ledger.record(
                    {**row, "state": "merged", "release_blocked": f"main floor {floor}"}
                )
                out.append(
                    {"repo": name, "status": "blocked", "reason": f"main floor {floor}"}
                )
                continue
            tag = f"v{version}"
            exists = (
                self.run(
                    ["gh", "api", f"repos/{repo.slug}/git/refs/tags/{tag}"]
                ).returncode
                == 0
            )
            if exists:
                result = "already-tagged"
            else:
                sha = self.run(
                    ["gh", "api", f"repos/{repo.slug}/commits/{base}", "-q", ".sha"]
                ).stdout.strip()
                res = self.run(
                    [
                        "gh",
                        "api",
                        f"repos/{repo.slug}/git/refs",
                        "-f",
                        f"ref=refs/tags/{tag}",
                        "-f",
                        f"sha={sha}",
                    ]
                )
                if res.returncode or not sha:
                    out.append(
                        {
                            "repo": name,
                            "status": "tag-failed",
                            "error": res.output[-300:],
                        }
                    )
                    continue
                result = "tagged"
            self.ledger.record({**row, "state": "released", "tag": tag})
            out.append({"repo": name, "status": result, "tag": tag})
        return out


def load_manifest(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in rows:
        key = r.get("state") or r.get("status") or "?"
        counts[key] = counts.get(key, 0) + 1
    return {"counts": dict(sorted(counts.items())), "rows": rows}


ROLLOUT_ACTIONS = ("plan", "apply", "status", "release")


def run_rollout_action(
    action: str,
    *,
    dependency: str,
    floor: str,
    manifest_path: Path | None = None,
    runner: Runner = subprocess_runner,
    ledger: Ledger | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Shared CLI/MCP core: one admission path for every rollout action."""
    if action not in ROLLOUT_ACTIONS:
        raise ValueError(
            f"unknown rollout action {action!r}; expected one of {ROLLOUT_ACTIONS}"
        )
    if not re.fullmatch(r"\d+\.\d+\.\d+", floor):
        raise ValueError(f"floor must be MAJOR.MINOR.PATCH, got {floor!r}")
    cfg = RolloutConfig(dependency=dependency, floor=floor, **options)
    engine = Rollout(cfg, runner=runner, ledger=ledger)
    if action in {"plan", "apply"}:
        if manifest_path is None:
            from repository_manager.workspace_manifest import (
                default_runtime_manifest_path,
            )

            manifest_path = default_runtime_manifest_path()
        manifest = load_manifest(manifest_path)
        rows = engine.plan(manifest) if action == "plan" else engine.apply(manifest)
    elif action == "status":
        rows = engine.status()
    else:
        rows = engine.release()
    return {
        "action": action,
        "dependency": dependency,
        "floor": floor,
        "range": cfg.spec_range,
        "branch": cfg.branch,
        "dry_run": cfg.dry_run,
        "ledger": str(engine.ledger.path),
        **summarize(rows),
    }
