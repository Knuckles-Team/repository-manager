from repository_manager.repository_manager import Git

workspace_yml = "/home/genius/Workspace/agent-packages/agents/repository-manager/repository_manager/workspace.yml"
git = Git(path="/home/genius/Workspace/agent-packages")
git.load_projects_from_yaml(workspace_yml)


git.path = "/home/genius/Workspace"

results = git.validate_projects(type="all")
summary = git.generate_markdown_summary("Validation", results)
print(summary)

with open("report_fixed.md", "w") as f:
    f.write(summary)
