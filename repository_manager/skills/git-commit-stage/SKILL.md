---
name: git-commit-stage
description: Manages staging/commits. Use for add/commit/amend. Triggers - save changes, contributions.
---

### Overview
Staging and committing via `git_action`. Ideal for contributions.

### Key Tools
- `git_action`: For all (e.g., "git add -A", "git commit -m 'msg'"). Params: command (required), repository_directory?.

### Usage Instructions
1. Status: "git status".
2. Add: "git add <file>" or "-A".
3. Commit: "git commit -m 'Message'".
4. Amend: "git commit --amend -m 'New msg'".
5. Contribution flow: status -> add -> commit -> push (from remote skill).

### Examples
- Add all: `git_action` with command="git add -A".
- Commit: `git_action` with command="git commit -m 'Fix bug'".

### Error Handling
- Nothing to commit: Check status.
- Amend no changes: Add first.
