import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Task(BaseModel):
    id: int = Field(..., description="Unique identifier for the task, e.g. 1, 2, 3")
    description: str = Field(
        ..., description="Clear, concise statement of what needs to be done"
    )
    acceptance_criteria: List[str] = Field(
        ..., description="List of testable conditions for completion"
    )
    passes: bool = Field(
        default=False, description="Flag indicating if the task is complete"
    )
    dependencies: Optional[List[int]] = Field(
        default=None, description="List of other task IDs this depends on"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or learnings from the agent"
    )
    priority: Optional[str] = Field(
        default=None, description="Priority level: high, medium, low"
    )
    execution_history: List[str] = Field(
        default_factory=list, description="Log of tool calls and execution steps"
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
            except Exception:
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


class PRD(BaseModel):
    project: str = Field(
        ...,
        description="The name of the project. This will also be used as the folder in the workspace. (Always lowercase)",
    )
    workspace: Optional[str] = Field(
        default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", "/documents"),
        description="The workspace location of the project",
    )
    summary: str = Field(
        default="", description="High-level summary of the project/feature"
    )
    description: str = Field(
        default="", description="Detailed description of the project/feature"
    )
    guardrails: List[str] = Field(
        default_factory=list, description="Global rules or constraints"
    )
    stories: List[Task] = Field(..., description="List of tasks/stories")
    iteration_count: int = Field(
        default=0, description="Number of iterations completed"
    )

    @field_validator("stories", mode="before")
    def ensure_list_stories(cls, v):
        if isinstance(v, list):
            return v
        return [v]

    @field_validator("project", mode="before")
    def ensure_project_lowercase(cls, v):
        if isinstance(v, str):
            return v.lower()
        else:
            return str(v).lower()

    def is_complete(self) -> bool:
        """Check if all tasks are marked as passed."""
        return all(task.passes for task in self.stories)

    def get_next_task(self) -> Optional[Task]:
        """Get the next undone task that has all dependencies satisfied."""
        for task in self.stories:
            if not task.passes:
                if task.dependencies is None or all(
                    self.stories[dep - 1].passes
                    for dep in task.dependencies
                    if dep - 1 < len(self.stories)
                ):
                    return task
        return None


class ElicitationRequest(BaseModel):
    """
    Represents a request for more information from the user (Product Manager -> User).
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
