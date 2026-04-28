<<<<<<< HEAD
import os

from repository_manager.repository_manager import Git

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

workspace_yml = os.path.join(REPO_ROOT, "repository_manager", "workspace.yml")
git = Git(path=os.path.join(WORKSPACE_ROOT, "agent-packages"))
git.load_projects_from_yaml(workspace_yml)


git.path = WORKSPACE_ROOT

results = git.validate_projects(type="all")
summary = git.generate_markdown_summary("Validation", results)

=======
from repository_manager.repository_manager import Git

workspace_yml = "/home/genius/Workspace/agent-packages/agents/repository-manager/repository_manager/workspace.yml"
git = Git(path="/home/genius/Workspace/agent-packages")
git.load_projects_from_yaml(workspace_yml)


git.path = "/home/genius/Workspace"

results = git.validate_projects(type="all")
summary = results.to_markdown()
>>>>>>> 61af4a3 (Fixed several issues.)
print(summary)

with open("report_fixed.md", "w") as f:
    f.write(summary)
