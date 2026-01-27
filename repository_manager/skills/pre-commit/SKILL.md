---
name: code-quality
description: Run code quality checks and pre-commits
---

### Overview
Ensure code quality by running pre-commit checks on repositories.

### Tools
- `run_pre_commit`: Run `pre-commit run --all-files` on a project.
  - Params: `run` (bool, default True), `autoupdate` (bool, default False), `workspace` (optional), `project` (optional relative path).

### Usage
- **Validation**: Run this before considering a task complete to ensure no linting or formatting errors exist.
- **Maintenance**: Use `autoupdate=True` to update pre-commit hooks.

### Examples
- Check current repo: `run_pre_commit(project="my-active-project")`
- Update hooks: `run_pre_commit(autoupdate=True, project="my-active-project")`
