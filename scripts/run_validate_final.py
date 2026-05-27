from pathlib import Path
from repository_manager.repository_manager import Git

workspace_yml = str(
    Path(__file__).parent.parent / "repository_manager" / "workspace.yml"
)
git = Git(path=str(Path(__file__).parent.parent.parent.parent.parent))
git.load_projects_from_yaml(workspace_yml)

print(f"Workspace path: {git.path}")
results = git.validate_projects(type="all")
summary = results.to_markdown()

with open("report_final.md", "w") as f:
    f.write(summary)

print(summary)
