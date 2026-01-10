---
name: git-config
description: Manages Git config. Use for user/email, remotes. Triggers - setup, personalization.
---

### Overview
Config ops via `git_action`.

### Key Tools
- `git_action`: "git config --list", "git config user.name 'User'".

### Usage Instructions
1. Global: --global flag.
2. Set: "git config <key> <value>".
3. Get: "git config <key>".

### Examples
- Set user: `git_action` with command="git config --global user.name 'John Doe'".

### Error Handling
- Invalid key: Check docs.