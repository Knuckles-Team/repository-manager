# NODE_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers, Universal Skills, and Skill Graphs.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag / ID | Source MCP / Skill |
|------|-------------|---------------|-------|----------|--------------------|
| Repository-Manager Devops Engineer Specialist | Expert specialist for devops_engineer domain tasks. | You are a Repository-Manager Devops Engineer specialist. Help users manage and interact with Devops Engineer functionality using the available tools. | git_action, get_workspace_projects, clone_projects, pull_projects | devops_engineer | repository-manager |
| Repository-Manager Workspace Management Specialist | Expert specialist for workspace_management domain tasks. | You are a Repository-Manager Workspace Management specialist. Help users manage and interact with Workspace Management functionality using the available tools. | setup_workspace, install_projects, build_projects, validate_projects, generate_workspace_template, save_workspace_config, maintain_workspace | workspace_management | repository-manager |
| Repository-Manager Project Management Specialist | Expert specialist for project_management domain tasks. | You are a Repository-Manager Project Management specialist. Help users manage and interact with Project Management functionality using the available tools. | get_project_status, update_task_status | project_management | repository-manager |
| Repository-Manager Graph Intelligence Specialist | Expert specialist for graph_intelligence domain tasks. | You are a Repository-Manager Graph Intelligence specialist. Help users manage and interact with Graph Intelligence functionality using the available tools. | graph_build, graph_query, graph_path, graph_status, graph_reset, graph_impact | graph_intelligence | repository-manager |
| Repository-Manager Visualization Specialist | Expert specialist for visualization domain tasks. | You are a Repository-Manager Visualization specialist. Help users manage and interact with Visualization functionality using the available tools. | get_workspace_tree, get_workspace_mermaid, generate_agents_documentation | visualization | repository-manager |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| git_action | Executes an arbitrary Git command. | devops_engineer, project_manager, workspace_management | repository-manager |
| get_workspace_projects | Lists all project URLs defined in the workspace configuration. | devops_engineer, git_operations, project_management, workspace_management | repository-manager |
| clone_projects | Clones repositories. Defaults to all in workspace.yml if none provided. | devops_engineer, git_operations, project_manager | repository-manager |
| pull_projects | Pulls updates for all projects in the workspace. | devops_engineer, git_operations, project_manager | repository-manager |
| setup_workspace | Sets up the entire workspace, clones repos, and organizes subdirectories. | workspace_management | repository-manager |
| install_projects | Bulk installs Python projects defined in the workspace. | workspace_management | repository-manager |
| build_projects | Bulk builds Python projects defined in the workspace. | workspace_management | repository-manager |
| validate_projects | Bulk validates agent/MCP servers in the workspace. | workspace_management | repository-manager |
| generate_workspace_template | Generates a new workspace.yml template. | workspace_management | repository-manager |
| save_workspace_config | Saves a WorkspaceConfig to YAML. Useful for programmatically updating the workspace. | workspace_management | repository-manager |
| maintain_workspace | Runs the maintenance lifecycle across all projects in the workspace. | workspace_management | repository-manager |
| get_project_status | Reads the current project state from tasks.json and progress.json. | project_management | repository-manager |
| update_task_status | Updates the status and result of a specific task in tasks.json. | project_management | repository-manager |
| graph_build | Builds or synchronizes the Hybrid Workspace Graph (NetworkX + Ladybug). | graph_intelligence | repository-manager |
| graph_query | Queries the Hybrid Graph using vector similarity or Cypher structure. | graph_intelligence | repository-manager |
| graph_path | Finds the shortest path between two symbols across the workspace graph. | graph_intelligence | repository-manager |
| graph_status | Returns the current status of the workspace graph (nodes, edges, communities). | graph_intelligence | repository-manager |
| graph_reset | Purges the graph database and Forces a clean rebuild on next build. | graph_intelligence | repository-manager |
| graph_impact | Calculates multi-repo impact for a symbol using the GraphEngine. | graph_intelligence | repository-manager |
| get_workspace_tree | Generates an ASCII tree of the workspace structure. | visualization | repository-manager |
| get_workspace_mermaid | Generates a Mermaid diagram of the workspace structure. | visualization | repository-manager |
| generate_agents_documentation | Generates an AGENTS.md catalog of discovered projects. | visualization | repository-manager |
