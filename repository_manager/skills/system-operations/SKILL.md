---
name: System Operations
description: Specialized skill for executing system-level commands and scripts.
---

# System Operations

This skill provides the ability to execute arbitrary shell commands on the underlying system. This is a powerful capability that allows for extensive automation and integration not covered by other specific tools.

## Tools

### Command Execution
- **`run_command`**: Execute a shell command.
    - `command` (str): The command to run (e.g., "ls -la", "python3 script.py").
    - `ctx` (Context, optional): Context for progress reporting (implicitly handled by MCP).

## Usage Examples

### Listing Files
`run_command` command tool with `command` = `"ls -la /workspace"`

### Running a Python Script

`run_command` command tool with `command` = `"python3 scripts/setup.py"`

### Checking System Status
`run_command` command tool with `command` = `"uptime"`
