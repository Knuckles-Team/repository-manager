---
name: repository-manager-workspace-release
skill_type: skill
description: >-
  Roll a dependency floor across the fleet deterministically and manage the
  `workspace.yml` manifest — via `rm_rollout` (or `repository-manager rollout`) and
  `rm_workspace_*`. A rollout selects the workspace.yml group's non-archived repos
  that depend on the package, plans per repo from main into a resumable ledger,
  applies each change in a fresh worktree through a pull request with auto-merge,
  reports merged/failing/pending, and tags `v<version>` only after the PR merged
  and main's floor is met. Nothing reaches main except through a PR. Use for
  "roll scion-sdk X to all connectors", "which rollout PRs are failing", "tag the
  repos whose rollout merged", or scaffolding/saving `workspace.yml`. Do NOT use
  for per-project validation (repository-manager-workspace-validation) or bulk
  clone/pull/push (repository-manager-bulk-git-operations).
license: MIT
tags: [repository-manager, release, rollout, dependency, workspace, mcp]
metadata:
  author: Knuckles-Team
  version: '2.0.0'
---
# Repository Manager — Workspace Release (dependency rollout)

A fleet release is a **dependency rollout**: raise one dependency's floor in every
repository that uses it, land it through a pull request per repository, then tag.
Each phase is deterministic and the ledger makes every re-run resume where the last
one stopped — never a duplicate PR, never a double bump.

## When to use
- Preview which repos a floor raise would touch, and their current floor/version (`plan`).
- Open the rollout PRs (`apply`), with bounded parallelism.
- See which PRs merged, which fail (with check names), which are pending (`status`).
- Tag repos whose rollout merged and whose main meets the floor (`release`; safe on a schedule).
- Scaffold, template, save, or list the `workspace.yml` manifest.

## When NOT to use
- Per-project validation semantics (pre-commit + pytest pass/fail) →
  `repository-manager-workspace-validation`.
- Bulk clone/pull/push → `repository-manager-bulk-git-operations`.
- Candidate/generation certification for one branch landing →
  `repository-manager-candidate-certification`.

## Tools & actions
| Tool group | Per-operation tools relevant here |
|----------------|---------|
| `rm_rollout` | actions `plan`, `apply`, `status`, `release` |
| `rm_workspace_*` | `rm_workspace_fleet_evidence_check`, `rm_workspace_list`, `rm_workspace_list_branches`, `rm_workspace_setup`, `rm_workspace_template`, `rm_workspace_save` |

CLI:
```
repository-manager rollout --dependency scion-sdk --floor 0.6.4 [--extra mcp] [--group agents] [--dry-run]
repository-manager rollout status  --dependency scion-sdk --floor 0.6.4
repository-manager rollout release --dependency scion-sdk --floor 0.6.4
```

## The phases
1. **Select** — workspace.yml group (default `agents`) ∩ non-archived GitHub repos ∩
   repos whose `pyproject.toml` depends on the package.
2. **Plan** — per repo, read main: current floor, version, any open rollout PR. Written
   to a JSONL ledger under the XDG state directory keyed by (dependency, floor, repo).
3. **Apply** — fresh `git worktree add` off `origin/main` (never stash, never the
   canonical checkout): set `>=X,<next-minor`, `uv lock --upgrade-package`, verify the
   lock resolved X (else the row fails), patch-bump via the repo's bumpversion config,
   CHANGELOG entry, pre-commit hooks, commit with the repo's own identity, push, open a
   PR, `gh pr merge --auto --rebase`.
4. **Status** — merged / failing (failing check names) / pending, from ledger + GitHub.
5. **Release** — after merge and main's floor ≥ X, tag `v<pyproject version>` on main's
   HEAD if untagged. Tag pushes publish PyPI and Docker.

## Recipes
Preview (mutates nothing):
```
rm_rollout(action="plan", dependency="scion-sdk", floor="0.6.4")
```
Open the PRs, then watch them:
```
rm_rollout(action="apply", dependency="scion-sdk", floor="0.6.4")
rm_rollout(action="status", dependency="scion-sdk", floor="0.6.4")
```
Tag what merged:
```
rm_rollout(action="release", dependency="scion-sdk", floor="0.6.4")
```

## Gotchas
- Run `plan`/`--dry-run` first; `apply` opens real PRs that auto-merge when CI is green.
- Re-running `apply` is safe: the ledger and the open-PR check skip finished rows.
- `release` never tags before the PR merged and main carries the new floor.
- `save` overwrites `workspace.yml` — pass an explicit `yml_path` when you don't
  want the default.

## Related
- `repository-manager-workspace-validation` — per-project validation.
- `repository-manager-bulk-git-operations` — bulk clone/pull/push.
- Mechanism: `repository_manager/rollout.py`, `docs/rollout.md`.
