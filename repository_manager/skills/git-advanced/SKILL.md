---
name: git-advanced
description: Advanced Git ops. Use for rebase/cherry-pick/bisect. Triggers - complex history, bugs.
---

### Overview
Advanced via `git_action`. For troubleshooting.

### Key Tools
- `git_action`: "git rebase main", "git cherry-pick <commit>", "git bisect start".

### Usage Instructions
1. Rebase: "git rebase <branch>".
2. Cherry-pick: "git cherry-pick <commit>".
3. Bisect: start/bad/good/reset.
4. Workflow: Bisect for bugs.

### Examples
- Rebase: `git_action` with command="git rebase -i HEAD~3".

### Error Handling
- Conflicts: Resolve as in merge.
  Reference `debug-bisect.md` for bug hunting.
