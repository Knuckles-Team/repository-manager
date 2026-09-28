# Contributing to repository-manager

## Setup

```bash
git clone https://github.com/Knuckles-Team/repository-manager.git
cd repository-manager
scripts/bootstrap.sh
```

`scripts/bootstrap.sh` is idempotent. It installs uv 0.9 or newer, the Python
in `.python-version`, clones the sibling sources `uv.lock` needs at the commits
in `scripts/siblings.lock`, syncs `.venv` from the lockfile with the `test` and
`agent` extras, and installs the pre-commit and pre-push git hooks. Pass
`--native` to also build the epistemic-graph kernel for `pytest -m native`.
Claude Code cloud sessions run it automatically through
`.claude/hooks/session-start.sh`.

## Checks

```bash
uvx pre-commit run --all-files
uv run --frozen --no-sync pytest tests -m "not slow and not integration and not native"
```

CI runs `scripts/bootstrap.sh` and then the same `.pre-commit-config.yaml`, so
local and CI results match. A gate whose tool or sibling checkout is missing
prints `SKIPPED (<gate>): <reason>` locally and fails with `CANNOT RUN` in CI.
Gates check behaviour or contracts; do not add hand-maintained counts, pins or
golden copies of source text to them, and do not make a gate depend on an
external service.

## Branches and pull requests

1. Branch from `main` (commits to `main` are refused by a hook).
2. Keep commits to one logical change each; run the checks above.
3. `git push -u origin <branch>` and open a pull request against `main`.
4. Keep `uv.lock` and `requirements.txt` in sync with `pyproject.toml`, and
   move a sibling pin only in `scripts/siblings.lock`.
