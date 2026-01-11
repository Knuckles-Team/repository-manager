---
name: git-tag-release
description: Manages tags. Use for versioning/releases. Triggers - milestones.
---

### Overview
Tagging via `git_action`.

### Key Tools
- `git_action`: "git tag v1.0", "git push --tags".

### Usage Instructions
1. Create: "git tag -a v1.0 -m 'Version 1'".
2. List: "git tag".
3. Delete: "git tag -d v1.0".
4. Push: "git push origin v1.0".

### Examples
- Tag: `git_action` with command="git tag v1.0".

### Error Handling
- Exists: Delete first.
