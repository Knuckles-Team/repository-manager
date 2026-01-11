---
name: git-remote-pull-push
description: Manages remotes/pull/push. Use for sync with upstream. Triggers - updates, deployments.
---

### Overview
Remote ops via MCP tools/`git_action`.

### Key Tools
- `pull_project`: Pull single. Params: git_project (dir required), repository_directory?.
- `pull_projects`: Pull all in dir. Params: repository_directory?.
- `git_action`: Custom (e.g., "git pull", "git push origin main").

### Usage Instructions
1. Pull: Prefer dedicated; "git pull" for options.
2. Push: "git push <remote> <branch>".
3. Remote: "git remote add origin <url>".
4. Chain: Pull -> merge (branch skill) -> push.

### Examples
- Pull single: `pull_project` with git_project="repo-dir".
- Push: `git_action` with command="git push".

### Error Handling
- Behind: Pull first.
- Auth: Ensure creds/SSH.
