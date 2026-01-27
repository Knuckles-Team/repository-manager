---
name: documentation
description: Read and manage project documentation
---

### Overview
Retrieve and manage the documentation (README.md) of projects.

### Tools
- `get_project_readme`: Get the content and path of the README.md file.
  - Params: `project` (optional directory name), `workspace` (optional).

### Usage
- **Context**: Read the README to understand the project structure and goals.
- **Updates**: Use `text_editor` (from text_editor skill) to update the README after reading it.

### Examples
- Read workspace root README: `get_project_readme()`
- Read specific project README: `get_project_readme(project="my-project")`
