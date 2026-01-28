---
name: shell
description: Execute bash commands for system interaction, code search, refactoring, library inspection, and small validations
---

### Overview
This skill allows execution of **any bash command** on the local system via a single tool: `run_command`.
It is especially powerful for **coding-related workflows**:

- Searching codebases (grep, rg, git grep)
- Refactoring & bulk editing (sed, perl -i, awk)
- Finding files & directories (find, fd, locate)
- Inspecting installed libraries & function source (python -c, cat, bat, inspect)
- Running small test/validation snippets
- Piping & combining utilities for complex queries

All operations must go through `run_command(command=...)`.
Be safety-conscious: prefer read-only operations first, use backups (`-i.bak`), dry-runs, and narrow globs when editing.

### Tools
- `run_command`
  **Description**: Execute an arbitrary bash command (single line or multi-line script via `bash -c '...'`).
  **Arguments**:
  - `command` (string, required): The full shell command to run

### Recommended Patterns (all executed via run_command)


#### 1. Code Search (find definitions, usages, TODOs, strings…)
```bash
# Basic grep (recursive, case-insensitive, show line numbers & filename)
grep -rni "def calculate_price" .

# Better: recursive + include only python files + exclude common dirs
grep -rni --include="*.py" --exclude-dir={.git,venv,node_modules,__pycache__} "TODO\|FIXME" .

# Show 5 lines of context around matches
grep -rniC 5 "some_function" src/

# Use ripgrep if installed (usually faster & nicer output)
rg -i --glob "*.py" "class DataProcessor"

# Find function definition + signature
grep -rni "^\s*def\s*process_data" .

# Find all calls to a function (approximate)
grep -rni "\bprocess_data(" .
```

#### 2. Text Processing (sed, awk)
```bash
# SED: Replace string in file (create backup .bak)
sed -i.bak 's/old_string/new_string/g' filename.py

# SED: Recursive replace in all python files (USE WITH CAUTION)
# Mac users: sed -i '' ...
grep -rl "OldClass" src/ | xargs sed -i.bak 's/OldClass/NewClass/g'

# AWK: Print the 2nd column of a CSV (comma separated)
awk -F, '{print $2}' data.csv

# AWK: Sum lines where the first column is "Error"
grep "Error" log.txt | awk '{count++} END {print count}'

# AWK: Print only lines longer than 80 chars
awk 'length($0) > 80' src/main.py
```
