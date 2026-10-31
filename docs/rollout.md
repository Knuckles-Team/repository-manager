# Dependency rollout

`repository-manager rollout` raises one dependency floor across a fleet of
repositories, deterministically and resumably, through pull requests
(RM-RELEASE-R009). It replaces the retired phased bump/push, which pushed
straight to `main`, still named pre-rename packages and never raised a
dependency floor or relocked.

```bash
# Preview: select + plan, writes nothing
repository-manager rollout --dependency scion-sdk --floor 0.6.4 --extra mcp --dry-run

# Plan + apply: one auto-merging PR per repository
repository-manager rollout --dependency scion-sdk --floor 0.6.4 --extra mcp --jobs 6

# Track: merged / failing (with check names) / pending
repository-manager rollout status --dependency scion-sdk --floor 0.6.4

# Tag v<version> on main once the PR merged and main's floor is met (schedule-safe)
repository-manager rollout release --dependency scion-sdk --floor 0.6.4
```

MCP: `rm_rollout(action="plan"|"apply"|"status"|"release", dependency=..., floor=...)`
(`dry_run` defaults to true on MCP). Toggle with `ROLLOUTTOOL`.

## Phases

1. **Select** — the `--group` (a `workspace.yml` selector, default
   `agent-connectors`, or a subdirectory prefix such as `arbora/connectors`)
   ∩ the forge's non-archived repositories ∩ repositories whose default-branch
   `pyproject.toml` declares the dependency. `--only a,b` narrows further.
2. **Plan** — per repository, read from the forge: the current floor, the
   version and any PR from branch `chore/rollout-<dependency>-<floor>`. Rows go
   to an append-only JSONL ledger (`$XDG_STATE_HOME/repository-manager/rollout/ledger.jsonl`,
   `%LOCALAPPDATA%` on Windows) keyed by (dependency, floor, repo). A floor
   already met is `skipped`.
3. **Apply** — per repository, in a fresh `git worktree add` off
   `origin/<default>` (never the canonical checkout, never `git stash`), with
   bounded parallelism: set `>=X,<next-minor`; `uv lock --upgrade-package D
   --refresh` and fail the row unless the lock resolves D ≥ X; patch-bump via the
   bumpversion config (and `Dockerfile` pins); add a CHANGELOG entry; run
   `pipelines-hook spec-status --write`; run pre-commit; stage an explicit file
   list; commit with the repository's configured identity (`claude` is
   refused); push; open the PR; `gh pr merge --auto --rebase`.
4. **Track** — `rollout status`.
5. **Release** — `rollout release` tags `v<pyproject version>` on main's HEAD
   once merged and main's floor is ≥ X; existing tags are skipped. Tag pushes
   publish PyPI and Docker.

## Resume

Re-running the same command resumes from the ledger and the forge: an open or
merged rollout PR is adopted, a branch that was pushed but never got a PR is
opened without re-bumping, and `failed` rows are retried. It never opens a
duplicate PR and never bumps twice.
