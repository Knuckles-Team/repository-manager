#!/usr/bin/python
import sys

# coding: utf-8
import asyncio
import os
import argparse
import logging
import uvicorn
import json
import tempfile

from contextlib import asynccontextmanager
from typing import Optional, Any, List, Union, Dict
from pydantic_ai import (
    Agent,
    ModelSettings,
    RunContext,
    Tool,
    UsageLimitExceeded,
    ModelRetry,
    UsageLimits,
)
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai_skills import SkillsToolset
from fasta2a import Skill
from pydantic import ValidationError, Field
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from fastapi import FastAPI, Request
from starlette.responses import Response, StreamingResponse


from repository_manager.utils import (
    to_integer,
    to_boolean,
    to_float,
    to_list,
    to_dict,
    get_projects_file_path,
    get_skills_path,
    get_mcp_config_path,
    load_skills_from_directory,
    create_model,
    prune_large_messages,
)
from repository_manager.models import Task, PRD, ElicitationRequest

__version__ = "1.3.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console
)
logging.getLogger("pydantic_ai").setLevel(logging.INFO)
logging.getLogger("fastmcp").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(string=os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3-4b-2507")
DEFAULT_OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://host.docker.internal:1234/v1"
)
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_SKILLS_DIRECTORY = os.getenv("SKILLS_DIRECTORY", get_skills_path())
DEFAULT_PROJECTS_FILE = os.getenv("PROJECTS_FILE", get_projects_file_path())
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.getenv(
    "REPOSITORY_MANAGER_WORKSPACE", "/workspace"
)

# Model Settings
DEFAULT_MAX_TOKENS = to_integer(os.getenv("MAX_TOKENS", "8192"))
DEFAULT_TOTAL_TOKENS = to_integer(os.getenv("TOTAL_TOKENS", "128000"))
DEFAULT_TEMPERATURE = to_float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = to_float(os.getenv("TOP_P", "1.0"))
DEFAULT_TIMEOUT = to_float(os.getenv("TIMEOUT", "32400.0"))
DEFAULT_TOOL_TIMEOUT = to_float(os.getenv("TOOL_TIMEOUT", "32400.0"))
DEFAULT_PARALLEL_TOOL_CALLS = to_boolean(os.getenv("PARALLEL_TOOL_CALLS", "True"))
DEFAULT_SEED = to_integer(os.getenv("SEED", None))
DEFAULT_PRESENCE_PENALTY = to_float(os.getenv("PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = to_float(os.getenv("FREQUENCY_PENALTY", "0.0"))
DEFAULT_LOGIT_BIAS = to_dict(os.getenv("LOGIT_BIAS", None))
DEFAULT_STOP_SEQUENCES = to_list(os.getenv("STOP_SEQUENCES", None))
DEFAULT_EXTRA_HEADERS = to_dict(os.getenv("EXTRA_HEADERS", None))
DEFAULT_EXTRA_BODY = to_dict(os.getenv("EXTRA_BODY", None))

AGENT_NAME = "Repository Manager Supervisor"
AGENT_DESCRIPTION = (
    "A Supervisor Agent that orchestrates a team of child agents (Planner, PM, Executor, Validator) "
    "to manage git repositories using the Ralph Wiggum methodology. "
    "Capabilities include dynamic repository management, code execution, and strict Product Requirements Document (PRD) driven development."
)

# -------------------------------------------------------------------------
# System Prompts
# -------------------------------------------------------------------------


SUPERVISOR_SYSTEM_PROMPT = (
    "You are the Repository Manager Supervisor\n"
    "You orchestrate a team of agents to manage Code Repositories.\n\n"
    "Workflow: \n"
    "1. Assess the user's request and trigger the correct tool. Choose between `manage_repositories` and `run_full_prd_process` tools with the users request."
)

PLANNER_SYSTEM_PROMPT = (
    "You are a PRD (Product Requirements Document) Planner and Orchestrator.\n"
    "Your Goal: Break down a high-level request into a structured and detailed PRD, then orchestrate processing using tools.\n"
    "Modes:\n"
    "- If the prompt is a user request, generate and output a PRD object.\n"
    "- If the prompt is to elicit approval for a PRD, use the 'elicit_product_requirements_document' tool in a loop until approved, then output the final PRD.\n"
    "- If the prompt is to process a specific task, call 'execute_task' then 'validate_task', and output the updated Task.\n"
    "- If the prompt is to finalize a PRD, call 'update_documentation' if needed, and output a status string.\n"
    "Research Capabilities:\n"
    "- Use `list_projects` tool to check if a project already exists. If the project exists, focus on it; if it doesn't, plan to create it. "
    "Note: Ensure the project name gets set on the PRD object.\n"
    "Agent Strategy:\n"
    "- **Plan for Specialized**: IF a specialized agent is strictly needed, you MUST create a specific Task to 'Create [Role] Agent'.\n"
    "  - In the description of this Task, you MUST specify the `system_prompt` and `tool_names` for the new agent so the Supervisor can create it.\n"
    "  - For subsequent tasks that use this agent, explicitly state 'Use the [Role] Agent to...' in the description.\n"
    "Requirements:\n"
    "- Output MUST match the mode: PRD for planning/elicit, Task for processing, str for finalize.\n"
    "- Define `project` - The name of the project. This will also be used as the folder in the workspace.\n"
    "- Define `summary` - Overall summary of the goal.\n"
    "- Define `description` - Detailed description of the goal.\n"
    "- Define `guardrails` - These should be high level rules that apply to all tasks. Like maintaining the correct project reference.\n"
    "- Define `stories` a list of atomic (Tasks).\n"
    "- Each task must have clear `acceptance_criteria`, `description`, `id`, `priority`, `dependencies` (optional), `notes` (optional).\n"
    "- Dependencies: If a task depends on one or more tasks, it MUST be a LIST of integers, e.g. `dependencies=[1]`, NOT `dependencies=1`.\n"
)

PRODUCT_MANAGER_SYSTEM_PROMPT = (
    "You are a Product Manager.\n"
    "Your Goal: Review the PRD (Product Requirements Document) and ensure it is complete and unambiguous.\n"
    "Requirements:\n"
    "- If information is missing, Return an `ElicitationRequest` object with a question for the user.\n"
    "- If the PRD (Product Requirements Document) is good, Return the approved `PRD` object.\n"
    "- You can accept the user's input/answers to update the PRD (Product Requirements Document).\n"
)

EXECUTOR_SYSTEM_PROMPT = (
    "You are a Task Executor.\n"
    "Your Goal: Implement a single Task fully in code.\n"
    "Workflow:\n"
    "1. Run `list_projects` and validate if project in the PRD exists."
    "   - Initialize new projects by running `create_project(project=project)`. "
    "2. **Development & Testing**: \n"
    "   - Use `text_editor` to write the code.\n"
    "   - **Ensure you specify the command, workspace, project, and path.** The path is relative to the project e.g. `src/main.py` or `README.md`.\n"
    "   - **Iterate**: If validation fails, use `text_editor` to fix the workspace files. \n"
    "   - **Validate** If a project has a .pre-commit-config.yaml file, run `run_pre_commits` tool to validate the code changes and resolve issues detected in codebase from the pre-commit tool logs\n"
    "PRD Requirements:\n"
    "- Return the updated `Task` object.\n"
    "- Update `task.notes` with implementation details.\n"
    "- Set `task.passes = True` ONLY if you have verified it.\n"
    "Coding Principles (Karpathy's Guidelines):\n"
    "1. Think Before Coding: State assumptions. Present tradeoffs. Stop and ask if confused. Don't guess.\n"
    "2. Simplicity First: No overengineering. No 'flexibility' not asked for. If 200 lines can be 50, do 50.\n"
    "3. Surgical Changes: Touch only what is needed. Match existing style. Don't 'improve' unrelated code. Clean up your own unused imports/variables.\n"
    "4. Goal-Driven Execution: Transform tasks into verifiable goals (Step -> Verify). Loop until success is verified."
)

VALIDATOR_SYSTEM_PROMPT = (
    "You are a Strict Validator.\n"
    "Your Goal: Verify a completed Task against its acceptance criteria. You also need to review the code changes using git tools.\n"
    "Requirements:\n"
    "- Check EVERY criterion.\n"
    "- Use git tools like `git_action(command='git diff')` or `git_action(command='git status')` to review the code changes and inspect code for quality and correctness.\n"
    "- Verify the pre-commits are passing by running the `run_pre_commits` tool.\n"
    "- Return the updated `Task` object.\n"
    "- If verification fails, set `passes = False` and explain why in `notes`."
)

REPOSITORY_MANAGER_SYSTEM_PROMPT = (
    "You are a specialist agent with access to git tools.\n"
    "Your Goal: Manage git repositories. You have tools to manage pull and clone repositories in bulk.\n"
    "Requirements:\n"
    "- Use git tools to manage repositories.\n"
    "- You DO NOT have direct git access; use the specific repo tools.\n"
    "- Return the results of running the tools.\n"
    "- Set `prd.last_agent = 'Repository Manager'`."
)

DOCUMENTATION_AGENT_SYSTEM_PROMPT = (
    "You are a Documentation Specialist.\n"
    "Your Goal: Manage project documentation, specifically README.md files.\n"
    "Capabilities:\n"
    "- Read README.md files using `get_project_readme`.\n"
    "- Edit README.md files using `text_editor` (Ensure to pass project and path fields).\n"
    "Requirements:\n"
    "- Ensure every README has a ## Changelog section.\n"
    "- Reflect changes from the PRD in the documentation.\n"
    "- Verify required changes (Deprecations, examples, architecture diagrams, CLI tables, etc).\n"
    "- Return the updated `Task` object or status string.\n"
)

INSTRUCTIONS = (
    "The PRD object always contains the currently active project in its 'project' field.\n"
    "→ You MUST read the project value from PRD['project'] (or equivalent path) at the beginning of every conversation turn.\n"
    "→ For EVERY call to text_editor, git_action, or any other tool that accepts a 'project' parameter, "
    "you MUST pass exactly that value as the project argument.\n"
    "Never guess, hard-code, or use a project name from memory or previous messages."
)


# -------------------------------------------------------------------------
# Agent Creation
# -------------------------------------------------------------------------


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    workspace: str = DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
) -> Agent:

    # 1. Setup Model
    # 1. Setup Model
    model = create_model(provider, model_id, base_url, api_key)
    settings = ModelSettings(
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        timeout=DEFAULT_TIMEOUT,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
        seed=DEFAULT_SEED,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        logit_bias=DEFAULT_LOGIT_BIAS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
        extra_headers=DEFAULT_EXTRA_HEADERS,
        extra_body=DEFAULT_EXTRA_BODY,
    )

    # Dictionary to hold available toolsets by name for dynamic assignment
    # explicit_tools["name"] = [tool1, toolset1, ...]
    available_tools_registry: Dict[str, List[Any]] = {}

    # 2. Master Skills (Git, etc)
    master_skills = []
    if skills_directory and os.path.exists(skills_directory):
        logger.info(f"Loading skills from {skills_directory}")
        loaded_skills = SkillsToolset(directories=[str(skills_directory)])
        master_skills.append(loaded_skills)
        # Register for dynamic usage
        available_tools_registry["git_skills"] = [loaded_skills]

    # 3. Prepare Executor Tools (Repo Delegates)
    executor_tools_list = []
    executor_toolsets_list = []
    rm_tools = []

    # Add Master Skills (Shell, etc) to Executor
    if master_skills:
        executor_toolsets_list.extend(master_skills)

    # Variable to hold the repo manager agent if found
    repository_manager_agent = None

    # 4. Dynamic MCP Parsing & Registry Population
    if mcp_config and os.path.exists(mcp_config):
        try:
            with open(mcp_config, "r") as f:
                config_data = json.load(f)

            mcp_servers = config_data.get("mcpServers", {})

            for server_name, server_config in mcp_servers.items():

                # A. Repository Manager (Bulk Git Ops) -> Supervisor Tool
                if server_name == "repository-manager":
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as temp_config:
                        json.dump(
                            {"mcpServers": {server_name: server_config}}, temp_config
                        )
                        temp_config_path = temp_config.name
                    try:
                        rm_tools = load_mcp_servers(temp_config_path)
                        # Register tools if needed dynamically
                        available_tools_registry["repository_manager_tools"] = rm_tools
                    except Exception as e:
                        logger.error(f"Failed to load repository-manager tools: {e}")
                    finally:
                        if os.path.exists(temp_config_path):
                            os.remove(temp_config_path)

                    # Add RM tools to executor tools so it can use git directly
                    if rm_tools:
                        executor_toolsets_list.extend(rm_tools)
                        logger.info(
                            f"Added {len(rm_tools)} Repository Manager tools to Executor."
                        )

        except Exception as e:
            logger.error(f"Error parsing MCP config: {e}")

    # 4. Define Functional Agents
    planner_agent = Agent(
        model=model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        model_settings=settings,
        output_type=Union[PRD, Task, str],
        name="Planner",
        retries=3,
    )
    product_manager_agent = Agent(
        model=model,
        system_prompt=PRODUCT_MANAGER_SYSTEM_PROMPT,
        model_settings=settings,
        output_type=Union[PRD, ElicitationRequest],
        name="Product Manager",
        retries=3,
    )

    executor_agent = Agent(
        model=model,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=settings,
        tools=executor_tools_list,
        toolsets=executor_toolsets_list,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        output_type=Task,
        name="Executor",
        retries=3,
    )

    validator_agent = Agent(
        model=model,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        model_settings=settings,
        tools=executor_tools_list,
        toolsets=executor_toolsets_list,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        output_type=Task,
        name="Validator",
        retries=3,
    )

    repository_manager_agent = Agent(
        model=model,
        system_prompt=REPOSITORY_MANAGER_SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        toolsets=rm_tools + master_skills,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        model_settings=settings,
        output_type=Union[Task, str],
        name="Repository Manager",
        retries=3,
    )

    documentation_agent = Agent(
        model=model,
        system_prompt=DOCUMENTATION_AGENT_SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        toolsets=rm_tools + master_skills,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        model_settings=settings,
        output_type=Union[Task, str],
        name="Documentation Agent",
        retries=3,
    )

    logger.info("Created Specialist Agents")

    # 5. Supervisor Tools

    supervisor = Agent(
        model=model,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        name=AGENT_NAME,
        model_settings=settings,
        retries=3,
    )
    logger.info("Created Supervisor Agent")

    @supervisor.tool
    async def manage_repositories(ctx: RunContext[Any], instruction: str) -> str:
        """
        Execute commands
        - listing, cloning, bulk pulling repositories
        - creating, deleting, moving files and directories
        - creating new projects (always use lowercase project name)
        - running pre-commit scans
        - bumping versions
        """
        result = await repository_manager_agent.run(
            instruction, deps=ctx.deps, usage=ctx.usage
        )
        return str(result.output)

    @supervisor.tool
    async def run_full_prd_process(
        ctx: RunContext[Any],
        user_prompt: str,
        max_iterations: Optional[int] = 10,  # Safety cap
    ) -> str:
        """Trigger the complete processing of a Product Requirements Document, including planning, elicitation, and task execution/validation loop."""
        limits = UsageLimits(
            request_limit=50,
            tool_calls_limit=100,
            total_tokens_limit=100_000,
        )

        try:
            # Step 1: Planning via planner
            plan_result = await run_with_retry(
                planner_agent,
                f"{user_prompt} \n\nThe workspace is '{workspace}', Please extrapolate the project field from the user's prompt.",
                ctx,
                usage_limits=limits,
            )
            if not isinstance(plan_result.output, PRD):
                return f"Planning failed: {plan_result.output}"
            prd = plan_result.output

            # Step 2: Elicitation via planner delegation
            elicit_result = await run_with_retry(
                planner_agent,
                f"Elicit approval for PRD: {prd.model_dump_json()}",
                ctx,
                usage_limits=limits,
            )
            if not isinstance(elicit_result.output, PRD):
                return f"Elicitation failed: {elicit_result.output}"
            prd = elicit_result.output

            # Step 3: Task processing loop (per-task planner delegation)
            iteration = 0
            while not prd.is_complete() and iteration < max_iterations:
                task = prd.get_next_task()
                if task is None:
                    break  # Dependencies block or done

                process_result = await run_with_retry(
                    planner_agent,
                    f"Process task: {task.model_dump_json()}. Call execute_task then validate_task.",
                    ctx,
                    usage_limits=limits,
                )
                if not isinstance(process_result.output, Task):
                    logger.warning(
                        f"Task {task.id} processing failed: {process_result.output}"
                    )
                    continue  # Or raise

                # Update PRD with processed task
                for i, t in enumerate(prd.stories):
                    if t.id == process_result.output.id:
                        prd.stories[i] = process_result.output
                        break

                iteration += 1
                prd.iteration_count += 1

            # Step 4: Finalization via planner
            finalize_result = await run_with_retry(
                planner_agent,
                f"Finalize PRD for project {prd.project}: update documentation.",
                ctx,
                usage_limits=limits,
            )
            status = (
                finalize_result.output
                if isinstance(finalize_result.output, str)
                else "Finalization complete."
            )

            if prd.is_complete():
                return f"PRD processing completed for project '{prd.project}'. All tasks passed. {status}"
            else:
                return f"PRD processing partially completed after {iteration} iterations. Remaining tasks: {[t.id for t in prd.stories if not t.passes]}. {status}"

        except UsageLimitExceeded as e:
            logger.error(f"Usage limit hit in PRD process: {e}")
            return f"Safety limit hit: {str(e)}"
        except Exception as e:
            logger.exception(f"Error in PRD process: {e}")
            return f"Error in PRD process: {str(e)}"

    @planner_agent.tool
    async def elicit_product_requirements_document(
        ctx: RunContext[Any], prd: PRD
    ) -> PRD:
        """Elicit approval for the PRD (Product Requirements Document) from the User."""
        current_prd = prd
        for _ in range(10):
            result = await product_manager_agent.run(
                f"Review this PRD:\n{current_prd.model_dump_json()}",
                usage=ctx.usage,
                deps=ctx.deps,
            )
            output = result.output

            if isinstance(output, PRD):
                return output

            elif isinstance(output, ElicitationRequest):
                logger.info(f"PM Elicitation Request: {output.question}")
                try:
                    user_response = input(
                        f"[PM asks]: {output.question} (Context: {output.context})\n> "
                    )
                except EOFError:
                    user_response = "Proceed"

                current_prd = await _feedback_loop_pm(
                    current_prd, user_response, ctx=ctx
                )

        return current_prd

    async def _feedback_loop_pm(
        prd: PRD, user_response: str, ctx: RunContext[Any]
    ) -> PRD:
        res = await product_manager_agent.run(
            f"Current PRD (Product Requirements Document): {prd.model_dump_json()}\nUser Answer to your question: {user_response}\nUpdate and Approve.",
            usage=ctx.usage,  # NEW
            deps=ctx.deps,  # NEW
        )
        if isinstance(res.output, ElicitationRequest):
            pass
        return res.output

    async def run_with_retry(
        agent: Agent,
        prompt: str,
        ctx: RunContext[Any],
        max_attempts: int = 3,
        usage_limits: Optional[UsageLimits] = None,
    ) -> Any:
        for attempt in range(1, max_attempts + 1):
            try:
                return await agent.run(
                    prompt,
                    usage=ctx.usage,
                    deps=ctx.deps,
                    usage_limits=usage_limits,
                )
            except UsageLimitExceeded:
                raise  # Don't retry limits
            except ValidationError as e:
                if attempt == max_attempts:
                    raise
                raise ModelRetry(
                    f"Output validation failed: {str(e)}. Refine and retry."
                )
            except Exception as e:
                if attempt == max_attempts:
                    raise
                logger.warning(f"Retry {attempt} for {agent.name}: {str(e)}")
                await asyncio.sleep(2**attempt)

    @planner_agent.tool
    async def execute_task(ctx: RunContext[Any], task: Task) -> Task:
        """Execute one task from the PRD."""
        try:
            result = await run_with_retry(
                executor_agent, f"Execute this task:\n{task.model_dump_json()}", ctx
            )

            # Capture history
            history_log = []
            for msg in result.all_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        # Check for ToolCallPart
                        if part.part_kind == "tool-call":
                            history_log.append(
                                f"Executor: {part.tool_name}({part.part_kind})[{part.timestamp}]"
                            )

            task_out = result.output
            if isinstance(task_out, Task):
                task_out.execution_history.extend(history_log)
                return task_out

            # Fallback if output is not Task
            return task
        except Exception as e:
            logger.exception(f"Error executing task: {e}")
            if task.notes:
                task.notes += f"\n\n[FATAL ERROR]: {e}"
            else:
                task.notes = f"[FATAL ERROR]: {e}"
            task.passes = False
            return task

    @planner_agent.tool
    async def validate_task(ctx: RunContext[Any], task: Task) -> Task:
        """Validate a completed task."""
        try:
            result = await run_with_retry(
                validator_agent, f"Validate this task:\n{task.model_dump_json()}", ctx
            )

            # Capture history
            history_log = []
            for msg in result.all_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        # Check for ToolCallPart
                        if part.part_kind == "tool-call":
                            history_log.append(
                                f"Validator: {part.tool_name}({part.part_kind})[{part.timestamp}]"
                            )

            task_out = result.output
            if isinstance(task_out, Task):
                task_out.execution_history.extend(history_log)
                return task_out
            return task
        except Exception as e:
            logger.exception("Error validating task")
            if task.notes:
                task.notes += f"\n\n[FATAL ERROR]: {e}"
            else:
                task.notes = f"[FATAL ERROR]: {e}"
            task.passes = False
            return task

    @planner_agent.tool
    async def update_documentation(ctx: RunContext[Any], instruction: str) -> str:
        """
        Delegate documentation tasks to the Documentation Agent.
        Use this to read, review, or update README.md files.
        """
        result = await run_with_retry(documentation_agent, instruction, ctx)
        return str(result.output)

    # Register supervisor tools for dynamic usage
    available_tools_registry["execute_task"] = [Tool(execute_task)]
    available_tools_registry["validate_task"] = [Tool(validate_task)]
    available_tools_registry["update_documentation"] = [Tool(update_documentation)]

    @planner_agent.tool
    async def create_specialized_agent(
        ctx: RunContext[Any],
        name: str = Field(..., description="Short lowercase name for the new agent"),
        system_prompt: str = Field(
            ..., description="System prompt for the child agent"
        ),
        tool_names: List[str] = Field(
            ...,
            description="Names of available toolsets to give. Options: 'git_skills', 'python_sandbox', 'repository_manager_tools', 'execute_task', 'validate_task', 'update_documentation'",
        ),
    ) -> str:
        """Create a new specialized child agent with specific tools and add it as a tool to the supervisor."""

        # Resolve tools from registry
        selected_tools = []
        for t_name in tool_names:
            if t_name in available_tools_registry:
                selected_tools.extend(available_tools_registry[t_name])
            else:
                return f"Error: Toolset '{t_name}' not found. Available: {list(available_tools_registry.keys())}"

        child_model = create_model(provider, model_id, base_url, api_key)

        # Split tools and toolsets
        child_tools = []
        child_toolsets = []
        for t in selected_tools:
            if isinstance(t, Tool) or callable(t):
                child_tools.append(t)
            else:
                child_toolsets.append(t)

        child_agent = Agent(
            system_prompt=system_prompt,
            model=child_model,
            model_settings=settings,
            tools=child_tools,
            toolsets=child_toolsets,
            tool_timeout=DEFAULT_TOOL_TIMEOUT,
            name=name,
            retries=3,
        )

        @planner_agent.tool(name=f"run_{name}")
        async def dynamic_tool(input_data: str) -> str:
            """Execute the specialized agent."""
            result = await child_agent.run(input_data)
            return str(result.output)

        return f"Created tool run_{name} with tools: {tool_names}"

    return supervisor


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}")


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    debug: Optional[bool] = DEFAULT_DEBUG,
    host: Optional[str] = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    enable_web_ui: bool = DEFAULT_ENABLE_WEB_UI,
):
    logger.info(
        f"Starting {AGENT_NAME} with provider={provider}, model={model_id}, mcp={mcp_url} | {mcp_config}"
    )

    agent = create_agent(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_config=mcp_config,
        skills_directory=skills_directory,
    )

    if skills_directory and os.path.exists(skills_directory):
        skills = load_skills_from_directory(skills_directory)
        logger.info(f"Loaded {len(skills)} skills from {skills_directory}")
    else:
        skills = [
            Skill(
                id="repository_manager_supervisor",
                name="Repository Manager Supervisor",
                description="Supervisor for Ralph Wiggum style repository management",
                tags=["repository_manager"],
                input_modes=["text"],
                output_modes=["text"],
            )
        ]

    a2a_app = agent.to_a2a(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        version=__version__,
        skills=skills,
        debug=debug,
    )

    # acp_app = agent.to_acp(name=AGENT_NAME, description=AGENT_DESCRIPTION)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title=f"{AGENT_NAME}",
        description=AGENT_DESCRIPTION,
        debug=debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "OK"}

    app.mount("/a2a", a2a_app)

    # app.mount("/acp", acp_app)

    @app.post("/ag-ui")
    async def ag_ui_endpoint(request: Request) -> Response:
        accept = request.headers.get("accept", SSE_CONTENT_TYPE)
        try:
            run_input = AGUIAdapter.build_run_input(await request.body())
        except ValidationError as e:
            return Response(
                content=json.dumps(e.json()),
                media_type="application/json",
                status_code=422,
            )

        # Prune large messages from history
        if hasattr(run_input, "messages"):
            run_input.messages = prune_large_messages(run_input.messages)
        limits = UsageLimits(total_tokens_limit=DEFAULT_TOTAL_TOKENS) if debug else None
        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream(usage_limits=limits)
        sse_stream = adapter.encode_stream(event_stream)
        return StreamingResponse(sse_stream, media_type=accept)

    if enable_web_ui:
        web_ui = agent.to_web(instructions=SUPERVISOR_SYSTEM_PROMPT)
        app.mount("/", web_ui)

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def agent_server():
    print(f"Repository Manager Agent v{__version__}")
    parser = argparse.ArgumentParser(
        add_help=False, description=f"Run the {AGENT_NAME} Server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    parser.add_argument("--debug", type=bool, default=DEFAULT_DEBUG, help="Debug")
    parser.add_argument("--reload", action="store_true", help="Reload")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--base-url", default=DEFAULT_OPENAI_BASE_URL, help="Base URL")
    parser.add_argument("--api-key", default=DEFAULT_OPENAI_API_KEY, help="API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP URL")
    parser.add_argument("--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Config")
    parser.add_argument("--workspace", default=os.getcwd(), help="Repository Workspace")
    parser.add_argument(
        "--web", action="store_true", default=DEFAULT_ENABLE_WEB_UI, help="Web UI"
    )
    parser.add_argument("--help", action="store_true", help="Show usage")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:

        usage()

        sys.exit(0)

    if "PROJECTS_FILE" not in os.environ and DEFAULT_PROJECTS_FILE:
        os.environ["PROJECTS_FILE"] = DEFAULT_PROJECTS_FILE
        logger.info(f"Set PROJECTS_FILE to default: {DEFAULT_PROJECTS_FILE}")

    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG, force=True)
        # ... logs ...

    create_agent_server(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config,
        debug=args.debug,
        host=args.host,
        port=args.port,
        enable_web_ui=args.web,
    )


def usage():
    print(
        f"Repository Manager ({__version__}): CLI Tool\n\n"
        "Usage:\n"
        "--host          [ Host ]\n"
        "--port          [ Port ]\n"
        "--debug         [ Debug ]\n"
        "--reload        [ Reload ]\n"
        "--provider      [ Provider ]\n"
        "--model-id      [ Model ID ]\n"
        "--base-url      [ Base URL ]\n"
        "--api-key       [ API Key ]\n"
        "--mcp-url       [ MCP URL ]\n"
        "--mcp-config    [ MCP Config ]\n"
        "--workspace     [ Repository Workspace ]\n"
        "--web           [ Web UI ]\n"
        "\n"
        "Examples:\n"
        "  [Simple]  repository-manager-agent \n"
        '  [Complex] repository-manager-agent --host "value" --port "value" --debug "value" --reload --provider "value" --model-id "value" --base-url "value" --api-key "value" --mcp-url "value" --mcp-config "value" --workspace "value" --web\n'
    )


if __name__ == "__main__":
    agent_server()
