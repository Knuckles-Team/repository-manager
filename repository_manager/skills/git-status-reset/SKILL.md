---
name: git-status-reset
description: Checks status/resets. Use for working dir state, undo changes. Triggers - cleanups, mistakes.
---

### Overview
Status and reset via `git_action`.

### Key Tools
- `git_action`: "git status", "git reset --hard", "git clean -f".

### Usage Instructions
1. Status: "git status -s".
2. Reset: "git reset <commit>" (soft/hard/mixed).
3. Clean: "git clean -fd" (untracked).
4. Troubleshoot: Status -> reset if needed.

### Examples
- Status: `git_action` with command="git status".
- Hard reset: `git_action` with command="git reset --hard HEAD".

### Error Handling
- Irreversible: Caution with --hard.
