---
name: git-stash
description: Manages stashes. Use for temp saving changes. Triggers - branch switches.
---

### Overview
Stash via `git_action`.

### Key Tools
- `git_action`: "git stash push -m 'msg'", "git stash pop".

### Usage Instructions
1. Stash: "git stash".
2. List: "git stash list".
3. Apply: "git stash apply stash@{0}".
4. Drop: "git stash drop".

### Examples
- Stash: `git_action` with command="git stash push".

### Error Handling
- No changes: Nothing stashed.
