import os

from repository_manager.repository_manager import Git

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

workspace_yml = os.path.join(REPO_ROOT, "repository_manager", "workspace.yml")
git = Git(path=WORKSPACE_ROOT)
git.load_projects_from_yaml(workspace_yml)

print(f"Workspace path: {git.path}")
results = git.validate_projects(type="all")
summary = results.to_markdown()

with open("report_final.md", "w") as f:
    f.write(summary)

print(summary)
