import asyncio
import logging
import os

from repository_manager.repository_manager import Git

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Validator")


async def main():
    # Initialize Git instance
    workspace_root = os.path.expanduser("~/Workspace")
    git = Git(path=workspace_root, threads=8)

    # Load projects from workspace.yml
    yml_path = "repository_manager/workspace.yml"
    if not os.path.exists(yml_path):
        # Fallback if run from a different CWD
        yml_path = os.path.join(os.path.dirname(__file__), "workspace.yml")

    logger.info(f"Loading workspace from {yml_path}")
    git.load_projects_from_yaml(yml_path)

    # Run validation
    logger.info("Starting bulk validation of all projects...")
    report = git.validate_projects(type="all")

    # Generate summary
    summary = report.to_markdown()

    # Write summary to a file
    with open("validation_summary.md", "w") as f:
        f.write(summary)

    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)

    print(f"Total: {report.total}")
    print(f"Success: {report.success_count} ✅")
    print(f"Failure: {report.failure_count} ❌")
    print(f"Skipped: {report.skipped_count} ⏭️")
    print("\nDetailed summary written to validation_summary.md")


if __name__ == "__main__":
    asyncio.run(main())
