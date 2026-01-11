---
name: git-branch-merge
description: Manages branches/merges. Use for switching/creating/deleting branches, merging. Triggers - feature branches, PRs.
---

### Overview
Branch ops via `git_action`. Supports troubleshooting merges.

### Key Tools
- `git_action`: For all (e.g., "git branch -a", "git checkout -b new"). Params: command (required), repository_directory?.

### Usage Instructions
1. List: "git branch -a".
2. Create/switch: "git checkout -b <branch>".
3. Merge: "git merge <branch>".
4. Delete: "git branch -d <branch>".
5. Troubleshoot: "git status" -> "git diff" -> resolve -> merge.

### Examples
- List: `git_action` with command="git branch --list".
- Merge: `git_action` with command="git merge main".

### Error Handling
- Conflicts: Use "git status" to identify, edit files, "git add", "git merge --continue".
  Reference `troubleshoot-merge.md` for conflicts.
