from repository_manager.repository_manager import Git

workspace_yml = "/home/genius/Workspace/agent-packages/agents/repository-manager/repository_manager/workspace.yml"
git = Git(path="/home/genius/Workspace")
git.load_projects_from_yaml(workspace_yml)

print(f"Workspace path: {git.path}")
results = git.validate_projects(type="all")
summary = results.to_markdown()

with open("report_final.md", "w") as f:
    f.write(summary)

print(summary)
