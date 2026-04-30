#!/usr/bin/env python3
import asyncio
import os
import sys

# Add current directory to PYTHONPATH
sys.path.append(os.getcwd())

from repository_manager.mcp_server import get_git_instance


async def main():
    print("Executing get_workspace_projects directly...")
    try:
        git = get_git_instance()
        projects = list(git.project_map.keys())
        print(f"\nFound {len(projects)} projects:")
        for p in sorted(projects):
            print(f"  - {p}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
