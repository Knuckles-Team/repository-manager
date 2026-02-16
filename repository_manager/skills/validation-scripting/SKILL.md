---
name: Validation Scripting
description: Create small Python scripts to validate library usage, troubleshoot issues, or verify functionality.
---

# Validation Scripting

This skill involves writing and executing small, targeted Python scripts to validate assumptions about libraries, test specific functionality, or troubleshoot complex issues by isolating them.

## Purpose

- **Validate Library Usage**: Verify how a specific function or class behaves before integrating it into the main codebase.
- **Troubleshoot Issues**: Isolate a bug by reproducing it in a minimal script.
- **Verify Fixes**: quick confirmation that a change works as expected.
- **Inspect Objects**: Print attributes and methods of objects at runtime to understand their structure.

## Process

1.  **Draft Script**: Create a file (e.g., `validate_feature.py`) using `text_editor` or `create_project` (if a full repo structure is needed).
2.  **Write Code**: Implement the minimal code necessary to test the feature or reproduce the issue.
3.  **Execute**: Run the script using `run_command`.
4.  **Analyze Output**: Review stdout/stderr to confirm behavior.
5.  **Refine**: Modify the script and re-run as needed.
6.  **Cleanup**: Delete the script once validation is complete.

## Toolkit

- **`text_editor`**: To create and edit the script file.
- **`run_command`**: To execute the script (`python3 script.py`).
- **`read_file`**: To check the script content if needed.
- **`delete_directory` / `run_command`**: To remove the script after use.

## Example Scenario

### Validating a Library Function

```python
# Create the validation script
script_content = """
import os
from repository_manager.utils import to_boolean

print(f"True -> {to_boolean('True')}")
print(f"false -> {to_boolean('false')}")
print(f"None -> {to_boolean(None)}")
"""

await text_editor(command="create", path="validate_bool.py", file_text=script_content)

# Run the script
result = await run_command(command="python3 validate_bool.py")
print(result["output"])

# Cleanup
await run_command(command="rm validate_bool.py")
```

### Troubleshooting an Import Error

```python
# Create a script to check sys.path and imports
script_content = """
import sys
print(sys.path)
try:
    import some_module
    print("Module imported successfully")
except ImportError as e:
    print(f"Import failed: {e}")
"""

await text_editor(command="create", path="debug_import.py", file_text=script_content)
await run_command(command="python3 debug_import.py")
await run_command(command="rm debug_import.py")
```
