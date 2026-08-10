"""CLI command adapters for Repository Manager.

Command modules marshal arguments to the existing Git, merge-queue, build,
manifest, and lane cores.  ``repository_manager.repository_manager.main``
remains the packaged entrypoint and delegates here.
"""

from repository_manager.cli_commands.build_queue import run_build_queue_cli
from repository_manager.cli_commands.concepts import run_concepts_cli
from repository_manager.cli_commands.context import CliRuntime, runtime_from_module
from repository_manager.cli_commands.lane import run_lane_cli
from repository_manager.cli_commands.merge_queue import run_merge_queue_cli
from repository_manager.cli_commands.parser import run
from repository_manager.cli_commands.remote_workers import run_remote_workers_cli

__all__ = [
    "CliRuntime",
    "run",
    "run_build_queue_cli",
    "run_concepts_cli",
    "run_lane_cli",
    "run_merge_queue_cli",
    "run_remote_workers_cli",
    "runtime_from_module",
]
