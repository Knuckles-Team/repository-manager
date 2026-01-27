---
name: project-management
description: Manage project lifecycles (listing, creating)
---

### Overview
Manage the lifecycle of projects within the workspace. This includes listing existing projects and creating new ones.

### Tools
- `list_projects`: List all available Git projects in the workspace or configured projects file.
  - Params: `projects_file` (optional), `workspace` (optional).
- `create_project`: Create a new directory and initialize it as a git repository.
  - Params: `project_name` (required), `workspace` (optional).

### Usage
- **Discovery**: Use `list_projects` to see what you are working with.
- **Initialization**: Use `create_project` to start a new repository.

### Examples
- List projects: `list_projects()`
- Create 'my-new-app': `create_project(project_name="my-new-app")`
