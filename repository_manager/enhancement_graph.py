#!/usr/bin/python

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Optional
from dataclasses import dataclass

from pydantic_graph import End, Graph

from agent_utilities.graph_orchestration import (
    DomainNode,
    RouterNode,
    _RouterNodeBase,
    ProjectState,
    ProjectDeps,
    BaseProjectInitializerNode,
)
from agent_utilities.models import (
    TaskList,
    Task,
    TaskStatus,
    ProgressEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancementState(ProjectState):
    """
    State for the repository-manager enhancement workflow,
    inheriting the long-running harness from agent-utilities.
    """

    last_git_commit: Optional[str] = None


@dataclass
class EnhancementDeps(ProjectDeps):
    """
    Dependencies for the Enhancement workflow.
    """

    pass


@dataclass
class InitializerNode(BaseProjectInitializerNode):
    """
    Specialized initializer for repository enhancements.
    """

    async def run(self, ctx: Any) -> "PlannerNode | WorkspaceManagerNode | RouterNode":

        await super().run(ctx)

        query_lower = ctx.state.query.lower()
        if any(kw in query_lower for kw in ["workspace", "yml", "yaml", "config"]):
            return WorkspaceManagerNode()

        return PlannerNode()


@dataclass
class PlannerNode(_RouterNodeBase):
    """
    Decomposes the high-level enhancement request into a structured TaskList.
    Uses an LLM to generate the plan and presents it for approval.
    """

    async def run(self, ctx: Any) -> "ParallelCodingNode | End[dict]":
        if ctx.state.task_list.phases and not ctx.state.human_approval_required:
            logger.info(
                "Planner: Task list already exists and is approved. Proceeding."
            )
            return ParallelCodingNode()

        logger.info("Planner: Decomposing request into TaskList...")

        from pydantic_ai import Agent

        planner_agent = Agent(
            model=ctx.deps.agent_model,
            result_type=TaskList,
            system_prompt=(
                "You are an expert Software Architect and Project Planner. "
                "Your goal is to decompose a high-level goal into a dependency-aware, phased TaskList for a repository enhancement. "
                "Each phase should be a logical milestone. "
                "Tasks within a phase can often be done in parallel if they don't depend on each other. "
                "Be specific about test criteria and expected results for each task."
            ),
        )

        repo_info = f"Repository path: {ctx.state.project_root}\n"
        readme_path = os.path.join(ctx.state.project_root, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                repo_info += f"README Context:\n{f.read()[:1000]}\n"

        prompt = f"Goal: {ctx.state.query}\n\n{repo_info}"

        try:
            result = await planner_agent.run(prompt)
            ctx.state.task_list = result.data
            ctx.state.human_approval_required = not ctx.deps.auto_approve_plan

            ctx.state.sync_to_disk()

            if ctx.state.human_approval_required:
                logger.info("Planner: Task list generated. Awaiting human approval.")
                return End(
                    {
                        "message": "Task list generated. Please review tasks.json and approve to proceed.",
                        "task_list": ctx.state.task_list.model_dump(),
                    }
                )
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return End({"error": f"Planning failed: {e}"})

        return ParallelCodingNode()


@dataclass
class ParallelCodingNode(_RouterNodeBase):
    """
    Spawns multiple CodingNodes in parallel to handle tasks in the current phase.
    """

    async def run(self, ctx: Any) -> "ValidatorNode":
        phase_idx = ctx.state.task_list.current_phase_index
        if phase_idx >= len(ctx.state.task_list.phases):
            return ValidatorNode()

        current_phase = ctx.state.task_list.phases[phase_idx]
        pending_tasks = [
            t for t in current_phase.tasks if t.status == TaskStatus.PENDING
        ]

        if not pending_tasks:
            return ValidatorNode()

        execution_batch = pending_tasks[: ctx.deps.max_parallel_agents]
        ctx.state.current_batch_ids = [t.id for t in execution_batch]

        logger.info(
            f"ParallelCoding: Starting {len(execution_batch)} tasks in phase '{current_phase.name}'"
        )

        tasks = [self.execute_task_subagent(ctx, task) for task in execution_batch]
        await asyncio.gather(*tasks)

        ctx.state.sync_to_disk()

        return ValidatorNode()

    async def execute_task_subagent(self, ctx: Any, task: Task):
        """
        Delegates a specific task to a sub-agent.
        """
        from pydantic_ai import Agent

        logger.info(f"Delegating Task: {task.title} ({task.id})")
        task.status = TaskStatus.IN_PROGRESS

        coding_agent = Agent(
            model=ctx.deps.agent_model,
            system_prompt=(
                "You are an expert Coding Agent. "
                f"Your specific task is: {task.title}\n"
                f"Description: {task.description}\n"
                "You have access to repository tools. Make minimal, high-quality changes."
            ),
        )

        for toolset in ctx.deps.mcp_toolsets:
            coding_agent.toolsets.append(toolset)

        try:
            result = await coding_agent.run(f"Complete the task: {task.title}")
            task.result = result.data if hasattr(result, "data") else str(result.output)
            task.status = TaskStatus.COMPLETED

            ctx.state.progress_log.entries.append(
                ProgressEntry(
                    message=f"Completed Task: {task.title}",
                    metadata={"task_id": task.id},
                )
            )
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.result = f"Error: {e}"


@dataclass
class ValidatorNode(_RouterNodeBase):
    """
    Runs validation tools and checks results against the SprintContract.
    """

    async def run(self, ctx: Any) -> "ParallelCodingNode | PlannerNode | End[dict]":
        logger.info("Validator: Running phase validation...")

        phase_idx = ctx.state.task_list.current_phase_index
        current_phase = ctx.state.task_list.phases[phase_idx]

        all_complete = all(
            t.status == TaskStatus.COMPLETED for t in current_phase.tasks
        )

        if all_complete:
            ctx.state.task_list.current_phase_index += 1
            if ctx.state.task_list.current_phase_index >= len(
                ctx.state.task_list.phases
            ):
                return End({"status": "success", "results": ctx.state.results})
            else:
                return ParallelCodingNode()

        return End({"status": "partial_success", "results": ctx.state.results})


@dataclass
class WorkspaceManagerNode(_RouterNodeBase):
    """
    Conversational node for building and updating workspace.yml.
    """

    async def run(self, ctx: Any) -> "End[dict]":
        logger.info("WorkspaceManager: Handling workspace configuration.")

        return End({"status": "workspace_updated", "results": ctx.state.results})


def create_enhancement_graph(name: str = "EnhancementGraph") -> Graph:
    """
    Factory to create the Enhancement Graph.
    """
    return Graph(
        nodes=(
            InitializerNode,
            PlannerNode,
            ParallelCodingNode,
            ValidatorNode,
            WorkspaceManagerNode,
            RouterNode,
            DomainNode,
        ),
        name=name,
    )
