---
name: git-log-history
description: Queries Git history/logs. Use for commits/diffs/blame. Triggers - audits, reviews.
---

### Overview
History inspection via `git_action`.

### Key Tools
- `git_action`: For log/diff/blame (e.g., "git log --oneline", "git diff HEAD~1").

### Usage Instructions
1. Log: "git log --pretty=format:'%h %s'".
2. Diff: "git diff <commit1>..<commit2>".
3. Blame: "git blame <file>".
4. Show: "git show <commit>".
5. Subset: Limit with --since, --author.

### Examples
- Recent logs: `git_action` with command="git log -n 10".
- File blame: `git_action` with command="git blame README.md".

### Error Handling
- No history: New repo.
