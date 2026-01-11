---
name: repository-manager-clone
description: Initializes and clones Git repos. Use for setup/new projects. Triggers - start repo, clone from remote.
---

### Overview
Handles repo creation/cloning via MCP tools. Use `git_action` for custom init/clone.

### Key Tools
- `clone_project`: Clone single. Params: git_project (URL required), repository_directory?, threads?, set_to_default_branch?.
- `clone_projects`: Clone multiple. Params: projects? (list), projects_file?, repository_directory?.
- `git_action`: Custom (e.g., "git init", "git clone <url>"). Params: command (required), repository_directory?.

### Usage Instructions
1. For init: `git_action` with "git init".
2. Clone: Prefer dedicated tools; fallback to `git_action` for options (e.g., "git clone --depth 1 <url>").
3. Chain: Clone -> pull (from pull skill).

### Examples
- Init: `git_action` with command="git init", repository_directory="/path".
- Clone single: `clone_project` with git_project="https://github.com/user/repo".
- Clone multiple: `clone_projects` with projects=["url1", "url2"].

### Error Handling
- Exists: Check dir first.
- Auth: Ensure URL has creds if private.
