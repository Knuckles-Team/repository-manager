---
name: File Operations
description: Advanced skill for file navigation, reading, editing, and search within the codebase.
---

# File Operations

This skill provides powerful tools for exploring and modifying the codebase. It includes capabilities for searching, reading, editing content, and managing directories.

## Tools

### Search and Navigation
- **`search_codebase`**: Search for code patterns using regex (ripgrep).
    - `query` (str): Regex pattern.
    - `path` (str, optional): Directory to search (absolute or relative).
    - `glob_pattern` (str, optional): Filter files (e.g., "*.py").
    - `case_sensitive` (bool): Case sensitivity flag.
- **`find_files`**: Find files by name pattern.
    - `name_pattern` (str): Glob pattern for filename (e.g., "*.md").
    - `path` (str, optional): Directory to search.
- **`get_project_readme`**: content of the README.md file.
    - `path` (str, optional): Path to project or directory.

### File Content
- **`read_file`**: Read file content.
    - `path` (str): Path to file.
    - `start_line` (int, optional): Start line number (1-indexed).
    - `end_line` (int, optional): End line number (1-indexed).

### Editing
- **`text_editor`**: Versatile file system editor tool.
    - `command` (str): Command to execute (view, create, str_replace, insert, undo_edit).
    - `path` (str): Path to file.
    - `file_text` (str, optional): Content for create command.
    - `view_range` (List[int], optional): Range of lines for view command.
    - `old_str` (str, optional): String to replace.
    - `new_str` (str, optional): Replacement string.
    - `insert_line` (int, optional): Line to insert at.
- **`replace_in_file`**: Replace a block of text.
    - `path` (str): Path to file.
    - `target_content` (str): Exact text to replace (must be unique).
    - `replacement_content` (str): New text.
- **`create_directory`**: Create a new directory.
    - `path` (str): Path where directory should be created.
- **`delete_directory`**: Recursively delete a directory.
    - `path` (str): Path of directory to delete.
- **`rename_directory`**: Rename or move a directory or file.
    - `old_path` (str): Current path.
    - `new_path` (str): New path.

## Usage Examples

### Searching for Code
```python
# specific query in a specific path
await search_codebase(query="class Git", path="/workspace/repo")
# case-insensitive search
await search_codebase(query="error", case_sensitive=False)
```

### Reading a File Segment
```python
await read_file(path="/workspace/repo/README.md", start_line=1, end_line=10)
```

### Replacing Text
```python
await replace_in_file(
    path="/workspace/repo/config.py",
    target_content="DEBUG = True",
    replacement_content="DEBUG = False"
)
```

### Using Text Editor
```python
# Create a new file
await text_editor(command="create", path="/workspace/repo/new_script.py", file_text="print('Hello')")

# Insert text at line 5
await text_editor(command="insert", path="/workspace/repo/script.py", insert_line=5, new_str="    # New comment\n")
```
