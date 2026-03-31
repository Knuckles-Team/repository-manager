import os
from repository_manager.repository_manager import Git

workspace_yml = "/home/genius/Workspace/agent-packages/agents/repository-manager/repository_manager/workspace.yml"
git = Git(path="/home/genius/Workspace")
git.load_projects_from_yaml(workspace_yml)

print(f"Workspace path: {git.path}")
for url, path in git.project_map.items():
    pkg = url.split("/")[-1].replace(".git", "")
    exists = os.path.exists(path)
    bump_exists = os.path.exists(os.path.join(path, ".bumpversion.cfg"))
    print(f"Project: {pkg}")
    print(f"  Path: {path}")
    print(f"  Exists: {exists}")
    print(f"  Bump Exists: {bump_exists}")
