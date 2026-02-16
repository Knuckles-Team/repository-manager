---
name: Git Operations
description: Comprehensive skill for managing Git repositories, including cloning, pulling, branching, and versioning.
---

# Git Operations

This skill provides a complete set of tools for managing Git repositories. It covers cloning, pulling, committing (via git_action), and version management.

## Tools

### Core Git Actions
- **`git_action`**: Execute any arbitrary Git command. Use this tool for any git operation not covered by specialized tools.
    - `command` (str): The git command (e.g., "git status", "git commit -m 'msg'").
    - `path` (str, optional): Path to execute the command in. Defaults to workspace.
    - `threads` (int, optional): Number of threads for parallel processing.
    - `set_to_default_branch` (bool, optional): Whether to checkout default branch.

### Project Management
- **`create_project`**: Create a new project directory and initialize it as a git repository.
    - `path` (str): Path for the new project.
- **`clone_project`**: Clone a single repository.
    - `url` (str): URL of the repo.
    - `path` (str, optional): Path to clone into.
- **`clone_projects`**: Clone multiple repositories in parallel.
    - `projects` (List[str], optional): List of URLs.
    - `projects_file` (str, optional): File containing list of URLs.
    - `path` (str, optional): Path to clone into.
- **`pull_project`**: Pull updates for a single repository.
    - `path` (str): Path of the project to pull.
- **`pull_projects`**: Pull updates for multiple repositories in parallel.
    - `path` (str, optional): Workspace path containing projects.
- **`list_projects`**: List all projects in the workspace.
    - `projects_file` (str, optional): File containing list of URLs.
    - `path` (str, optional): Workspace path.

### Maintenance
- **`run_pre_commit`**: Run pre-commit hooks.
    - `run` (bool): Run hooks (default True).
    - `autoupdate` (bool): Update hooks (default False).
    - `path` (str, optional): Path to run in.
- **`bump_version`**: Bump project version using bump2version.
    - `part` (str): Part to bump (major, minor, patch).
    - `path` (str, optional): Path to project.

## Usage Examples

### Cloning a Repository
```python
await clone_project(url="https://github.com/user/repo.git", path="/workspace/repo")
```

### Checking Status
```python
await git_action(command="git status", path="/workspace/repo")
```

### Committing Changes
```python
await git_action(command="git add .", path="/workspace/repo")
await git_action(command="git commit -m 'feat: new feature'", path="/workspace/repo")
```

### Common Git Actions

#### Configuration
```python
# Set user name globally
await git_action(command="git config --global user.name 'John Doe'")
# Set project as safe directory
await git_action(command="git config --global --add safe.directory '/workspace/repositories-list'")
```

#### Log & History
```python
# View recent logs
await git_action(command="git log -n 10", path="/workspace/repo")
# Check diff
await git_action(command="git diff HEAD~1", path="/workspace/repo")
# Blame file
await git_action(command="git blame README.md", path="/workspace/repo")
```

#### Remote Operations
```python
# Push changes
await git_action(command="git push origin main", path="/workspace/repo")
# Add remote
await git_action(command="git remote add origin https://github.com/user/repo.git", path="/workspace/repo")
```

#### Stashing
```python
# Stash changes
await git_action(command="git stash push -m 'temp work'", path="/workspace/repo")
# Pop stash
await git_action(command="git stash pop", path="/workspace/repo")
# List stashes
await git_action(command="git stash list", path="/workspace/repo")
```

#### Status & Reset
```python
# Check status
await git_action(command="git status", path="/workspace/repo")
# Hard reset (use with caution)
await git_action(command="git reset --hard HEAD", path="/workspace/repo")
# Clean untracked files
await git_action(command="git clean -fd", path="/workspace/repo")
```

#### Tagging
```python
# Create tag
await git_action(command="git tag -a v1.0 -m 'Version 1'", path="/workspace/repo")
# Push tags
await git_action(command="git push origin --tags", path="/workspace/repo")
```

### Creating a New Project
```python
await create_project(path="/workspace/new-project")
```
