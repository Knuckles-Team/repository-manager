import json

from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator


class AgentResponse(BaseModel):
    thoughts: str = Field(
        ..., description="Your internal reasoning and analysis of the situation."
    )
    plan: List[str] = Field(
        ..., description="A step-by-step checklist of what you are about to do."
    )
    action_taken: str = Field(
        ..., description="Summary of the actual tool calls or actions performed."
    )
    final_output: str = Field(..., description="The response to the user.")


class StepResult(BaseModel):
    """Generic container for agent steps."""

    thoughts: str
    plan: List[str]
    output: Any


class PlanStepResult(BaseModel):
    thoughts: str
    plan: List[str]
    output: Union["ImplementationPlan", "Clarification", "Task", str]


class TaskStepResult(BaseModel):
    thoughts: str
    plan: List[str]
    output: "Task"


class Task(BaseModel):
    id: int = Field(..., description="Unique identifier for the task, e.g. 1, 2, 3")
    description: str = Field(
        ..., description="Clear, concise statement of what needs to be done"
    )
    acceptance_criteria: List[str] = Field(
        ..., description="List of testable conditions for completion"
    )
    dependencies: Optional[List[int]] = Field(
        default=None, description="List of other task IDs this depends on"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Learnings and findings from the agent from their work on this task",
    )
    attempt_count: int = Field(
        default=0, description="Number of attempts made to complete this task"
    )
    status: str = Field(
        default="pending",
        description="Status: pending, in_progress, implemented, verified, failed",
    )

    @field_validator("dependencies", mode="before")
    def convert_dependencies(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except Exception as e:
                print(f"Unable to retrieve toolset: {e}")
                pass
        return [v]

    @field_validator("acceptance_criteria", mode="before")
    def ensure_list_of_strings(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except Exception:
                return [v]
        return [v]


class ImplementationPlan(BaseModel):
    summary: str = Field(
        default="", description="High-level summary of the implementation plan"
    )
    description: str = Field(default="", description="Detailed description of the plan")
    guardrails: List[str] = Field(
        default_factory=list, description="Global rules or constraints"
    )
    tasks: List[Task] = Field(..., description="List of tasks to implement")
    iteration_count: int = Field(
        default=0, description="Number of iterations completed"
    )

    @field_validator("tasks", mode="before")
    def ensure_list_tasks(cls, v):
        if isinstance(v, list):
            return v
        return [v]

    def is_complete(self) -> bool:
        """Check if all tasks are marked as verified."""
        return all(task.status == "verified" for task in self.tasks)

    def get_next_tasks(self) -> List[Task]:
        """Get all runnable tasks that have dependencies satisfied."""
        runnable = []
        completed_ids = {t.id for t in self.tasks if t.status == "verified"}

        for task in self.tasks:
            if task.status not in ["verified", "in_progress", "implemented"]:
                # Check dependencies
                deps_met = True
                if task.dependencies:
                    for dep_id in task.dependencies:
                        if dep_id not in completed_ids:
                            deps_met = False
                            break
                if deps_met:
                    runnable.append(task)
        return runnable

    def to_markdown(self) -> str:
        """Generate a markdown representation of the implementation plan."""
        md = []
        md.append(f"# Implementation Plan - {self.iteration_count} Iterations\n")
        md.append(f"**Summary:** {self.summary}\n")
        md.append(f"**Description:** {self.description}\n")

        if self.guardrails:
            md.append("## Guardrails")
            for g in self.guardrails:
                md.append(f"- {g}")
            md.append("")

        md.append("## Tasks")
        for task in self.tasks:
            status_icon = {
                "pending": "⏳",
                "in_progress": "🚧",
                "implemented": "✅",
                "verified": "🏁",
                "failed": "❌",
            }.get(task.status, "❓")

            md.append(f"### {status_icon} Task {task.id}: {task.description}")
            md.append(f"**Status:** {task.status} | **Attempts:** {task.attempt_count}")

            if task.dependencies:
                md.append(f"**Dependencies:** {task.dependencies}")

            if task.acceptance_criteria:
                md.append("**Acceptance Criteria:**")
                for ac in task.acceptance_criteria:
                    md.append(f"- {ac}")

            if task.notes:
                md.append(f"\n**Notes:**\n{task.notes}")

            md.append("\n---")

        return "\n".join(md)


class Clarification(BaseModel):
    """
    Represents a request for clarification from the user (Architect -> User).
    """

    question: str = Field(..., description="The question to ask the user.")
    context: Optional[str] = Field(
        default=None, description="Why this information is needed."
    )


class GitError(BaseModel):
    """Represents an error from a Git command."""

    message: str = Field(..., description="Error message from stderr")
    code: int = Field(..., description="Exit code of the command")


class GitMetadata(BaseModel):
    """Metadata about the executed Git command."""

    command: str = Field(..., description="The command executed")
    workspace: str = Field(..., description="The workspace directory")
    return_code: int = Field(..., description="Exit code")
    timestamp: str = Field(..., description="ISO formatted timestamp of execution")


class GitResult(BaseModel):
    """Result of a Git operation."""

    status: str = Field(
        ..., description="Status of the operation: 'success' or 'error'"
    )
    data: str = Field(..., description="Standard output from the command")
    error: Optional[GitError] = Field(
        default=None, description="Error details if status is 'error'"
    )
    metadata: GitMetadata = Field(..., description="Execution metadata")


class ReadmeResult(BaseModel):
    """Result of retrieving a README file."""

    content: str = Field(..., description="Content of the README file")
    path: str = Field(..., description="Path to the README file")
