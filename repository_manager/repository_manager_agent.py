#!/usr/bin/python
# coding: utf-8
import os
import argparse
import logging
import uvicorn
import json
import tempfile

from contextlib import asynccontextmanager
from typing import Optional, Any, List, Union, Dict
from pathlib import Path
from pydantic_ai import Agent, ModelSettings, RunContext, Tool
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai_skills import SkillsToolset
from fasta2a import Skill
from pydantic import ValidationError, Field
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from fastapi import FastAPI, Request
from starlette.responses import Response, StreamingResponse

try:
    from mcp_run_python import code_sandbox
except ImportError:
    code_sandbox = None
from repository_manager.utils import (
    to_integer,
    to_boolean,
    get_projects_file_path,
    get_skills_path,
    get_mcp_config_path,
    load_skills_from_directory,
    create_model,
    generate_mermaid_diagram,
    fetch_pyodide_packages,
)
from repository_manager.models import Task, PRD, ElicitationRequest

__version__ = "1.2.17"

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
DEFAULT_SMART_CODING_MCP_ENABLE = to_boolean(
    string=os.getenv("SMART_CODING_MCP_ENABLE", "True")
)
DEFAULT_PYTHON_SANDBOX_ENABLE = to_boolean(
    string=os.getenv("PYTHON_SANDBOX_ENABLE", "True")
)
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_GITLAB_AGENT_ENABLE = to_boolean(os.getenv("GITLAB_AGENT_ENABLE", "True"))
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.getenv(
    "REPOSITORY_MANAGER_WORKSPACE", "/workspace"
)

AGENT_NAME = "Repository Manager Supervisor"
AGENT_DESCRIPTION = (
    "A Supervisor Agent that orchestrates a team of child agents (Planner, PM, Executor, Validator) "
    "to manage git repositories using the Ralph Wiggum methodology. "
    "Capabilities include dynamic repository management, code execution, and strict Product Requirements Document (PRD) driven development."
)

# -------------------------------------------------------------------------
# System Prompts
# -------------------------------------------------------------------------

KARPATHY_GUIDELINES = (
    "\n\nCORE PRINCIPLES (KARPATHY GUIDELINES):\n"
    "1. Think Before Coding: State assumptions. Present tradeoffs. Stop and ask if confused. Don't guess.\n"
    "2. Simplicity First: No overengineering. No 'flexibility' not asked for. If 200 lines can be 50, do 50.\n"
    "3. Surgical Changes: Touch only what is needed. Match existing style. Don't 'improve' unrelated code. Clean up your own unused imports/variables.\n"
    "4. Goal-Driven Execution: Transform tasks into verifiable goals (Step -> Verify). Loop until success is verified."
)

SUPERVISOR_SYSTEM_PROMPT = (
    "You are the Repository Manager Supervisor — persistent and adaptive.\n"
    "You orchestrate a team of agents to manage git repositories through ad hoc commands and/or complete coding tasks based on a PRD (Product Requirements Document) for more complex tasks.\n\n"
    "You should determine if the user's request is a simple repository management task or a complex coding task while preferring simple tasks to save on time and resources.\n"
    "Workflow: Simple Repository Management\n"
    "1. Delegate git commands to the repository agent for things like creating a new project, bulk cloning and pulling, cloning or pulling a project, or getting the git status of a project.\n"
    "Workflow: Complex Coding Tasks\n"
    "1. If no PRD exists, call `plan_product_requirements_document` with the user's initial prompt.\n"
    "2. Call `elicit_product_requirements_document` to get user approval. The Product Manager may ask questions; "
    "you must bubble these up to the user via the tool's interaction pattern.\n"
    "3. Once PRD (Product Requirements Document) is approved, loop through tasks:\n"
    "   - **Agent Creation**: If the task appears to be about creating a new agent (e.g. 'Create SQL Agent'), CALL `create_specialized_agent` with the details (name, prompt, tools) found in the task description.\n"
    "   - **Specialized Execution**: If the task is designated for a specialized agent (e.g. 'Use SQL Agent to...'), CALL the dynamic tool `run_{agent_name}` directly.\n"
    "   - **Standard Execution**: For all other tasks, CALL `execute_task` to have the Executor implement it.\n"
    "   - **Validation**: After execution, always CALL `validate_task` to verify it.\n"
    "4. Do not stop until the PRD (Product Requirements Document) is fully complete (is_complete=True).\n"
    "5. Once the PRD is complete, Call `finalize_prd_and_diagram` to aggregate execution history and generate a diagram.\n"
    "6. Then Call `update_documentation` to reflect all changes in the README.md including the diagram.\n"
    "Memory Management:\n"
    "- Use `add_memory` to save important context like the current active project, workspace path, or user preferences.\n"
    "- Use `search_memories` to retrieve this context if it is missing.\n"
)

PLANNER_SYSTEM_PROMPT = (
    "You are a PRD (Product Requirements Document) Planner.\n"
    "Your Goal: Break down a high-level request into a structured PRD (Product Requirements Document).\n"
    "Research Capabilities:\n"
    "- Use `list_projects` tool to check if a project already exists. If the project exists, focus on it; if it doesn't, plan to create it.\n"
    "- Use `match_pyodide_packages` to find the correct package names for the python-sandbox (e.g. `pygame-ce` for `pygame`). If no package is found, we can assume it will be installed as a native python pip.\n"
    "- Use `search_memories` to check for existing PRD requirements, project context, or user preferences to inform your plan.\n"
    "Agent Strategy:\n"
    "- **Evaluate Team**: Determine if the request requires a specialized team of agents (e.g., SQL related -> SQL Agent) or if the current team (Executor) uses standard capabilities (Python, Files, Git).\n"
    "- **Prioritize Standard**: ALWAYS prefer the standard Executor team if they can meet the requirements.\n"
    "- **Plan for Specialized**: IF a specialized agent is strictly needed, you MUST create a specific Task to 'Create [Role] Agent'.\n"
    "  - In the description of this Task, you MUST specify the `system_prompt` and `tool_names` for the new agent so the Supervisor can create it.\n"
    "  - For subsequent tasks that usage this agent, explicitly state 'Use the [Role] Agent to...' in the description.\n"
    "Requirements:\n"
    "- Output MUST be a valid PRD (Product Requirements Document) object.\n"
    "- Define `project_name`.\n"
    "- Define `overall_goal`, `guardrails`, and a list of atomic `stories` (Tasks).\n"
    "- Each task must have clear `acceptance_criteria`.\n"
    "- **MANDATORY PROCEDURAL STEPS**: You MUST explicitly include the following procedural steps in the `description` or `acceptance_criteria` of the tasks:\n"
    "  1. **Initialization**: \n"
    "     - If NEW project: validation step to run `create_project` tool with the project name and workspace -> `create_directory` tool for structure within the project like src, tests, docs, etc. -> `text_editor` tool to create and update files.\n"
    "  2. **Research**: For existing codebase, include a task/step to use `smart-coding-*` tools to research the codebase. Skip this step if its a new project.\n"
    "  3. **Development**: \n"
    "     - **Write to Workspace (CRITICAL)**: Mandate using `text_editor` to write code to the ACTUAL project files in the workspace. **YOU MUST SPECIFY FULL PATHS INCLUDING THE PROJECT DIRECTORY WHICH WAS THE PROJECT NAME** (e.g. `/workspace/my_project/src/main.py`, NOT just `src/main.py`).\n"
    "     - **Conditional Sandbox Usage**: ONLY IF the user expressly asks to run/test in a sandbox, include a step to use `run_python_in_sandbox` to EXECUTE the code from the workspace files. Otherwise, rely on the filesystem.\n"
    "  4. **Validation**: \n"
    "     - Check for `.pre-commit` configuration and run `run_pre_commits` if present.\n"
    "     - Create and run `pytest` tests to validate logic (run via `run_command` or similar if environment permits, otherwise rely on pre-commits).\n"
    "- Identify and list any `required_packages` (Python dependencies) needed for each task.\n"
    "- **CRITICAL**: Use the `match_pyodide_packages` tool to ensure package names are compatible with Pyodide (e.g., use 'pygame-ce', not 'pygame') IF sandbox usage is planned.\n"
    "- **Dependencies**: If a task depends on one or more tasks, it MUST be a LIST of integers, e.g. `dependencies=[1]`, NOT `dependencies=1`.\n"
    "- Always include a final task for **Documentation Review** to ensure the README.md is updated with the changes.\n"
    "- Set `prd.last_agent = 'Planner'`."
)

PRODUCT_MANAGER_SYSTEM_PROMPT = (
    "You are a Product Manager.\n"
    "Your Goal: Review the PRD (Product Requirements Document) and ensure it is complete and unambiguous.\n"
    "Requirements:\n"
    "- If information is missing, Return an `ElicitationRequest` object with a question for the user.\n"
    "- If the PRD (Product Requirements Document) is good, Return the approved `PRD` object.\n"
    "- You can accept the user's input/answers to update the PRD (Product Requirements Document).\n"
    "- Set `prd.last_agent = 'Product Manager'`."
)

EXECUTOR_SYSTEM_PROMPT = (
    "You are a Task Executor.\n"
    "Your Goal: Implement a single Task fully in code.\n"
    "Workflow:\n"
    "1. **Initialize**: Identify if this is an existing project, or new project. "
    "   - Initialize new projects by running `create_project`. "
    "   - Navigate to existing projects by running `run_command` 'cd <project_name>'.\n"
    "2. **Research**: Use `smart-coding-*` tools to research the codebase and find relevant information pertaining to the task.\n"
    "3. **Development & Testing**: \n"
    "   - **Write to Workspace (CRITICAL PATH WARNING)**: Use `text_editor` to write the code to the actual project files.\n"
    "     - **YOU MUST ALWAYS USE THE FULL PROJECT PATH**. Example: If project is 'snake', write to `/workspace/snake/src/main.py`.\n"
    "     - **DO NOT** write to `src/main.py` directly as this goes to the workspace root. **ALWAYS PREPEND THE PROJECT NAME AND WORKSPACE**.\n"
    "   - **Conditional Sandbox Test**: IF the task specifically instructs to use the sandbox, read the content of the files you just wrote and use `run_python_in_sandbox` to execute/validate them.\n"
    "   - **Iterate**: If validation fails, use `text_editor` to fix the workspace files.\n"
    "Capabilities:\n"
    "- `run_python_in_sandbox`: For temporary development, executing and testing code within a secure isolation. **CRITICAL**: Check `task.required_packages` and pass them as the `dependencies` argument (e.g., ['numpy', 'pygame-ce']). "
    "You can call it multiple times, adjusting deps or code based on previous results. Use this ONLY if instructed.\n"
    "- `text_editor`: For writing the final, tested code to the project files.\n"
    "- `smart-coding-*`: To semantic search code repositories. There is a different instance of smart-coding-* for each project in the workspace so ensure you are using the correct one when using this tool.\n"
    "- `git-*`: To interact with git (via specific repo tools).\n"
    "- `add_memory` / `search_memories`: To save or retrieve context about the execution or project state.\n"
    "Requirements:\n"
    "- Return the updated `Task` object.\n"
    "- Update `task.notes` with implementation details.\n"
    "- Set `task.passes = True` ONLY if you have verified it and written code to the workspace.\n"
    "- Set `task.last_agent = 'Executor'`."
) + KARPATHY_GUIDELINES

VALIDATOR_SYSTEM_PROMPT = (
    "You are a Strict Validator.\n"
    "Your Goal: Verify a completed Task against its acceptance criteria. You also need to review the code changes using git tools.\n"
    "Requirements:\n"
    "- Check EVERY criterion.\n"
    "- Use git tools to review the code changes and inspect code for quality and correctness.\n"
    "- Verify the pre-commits are passing by running the `run_pre_commits` tool.\n"
    "- Return the updated `Task` object.\n"
    "- If verification fails, set `passes = False` and explain why in `notes`."
    "- Set `task.last_agent = 'Validator'`."
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
    "- Edit README.md files using `text_editor`.\n"
    "Requirements:\n"
    "- Ensure every README has a ## Changelog section.\n"
    "- Reflect changes from the PRD in the documentation.\n"
    "- Verify required changes (Deprecations, examples, architecture diagrams, CLI tables, etc).\n"
    "- Return the updated `Task` object or status string.\n"
    "- Set `prd.last_agent = 'Documentation Agent'`."
)


# -------------------------------------------------------------------------
# Agent Creation
# -------------------------------------------------------------------------


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    enable_gitlab_agent: bool = DEFAULT_GITLAB_AGENT_ENABLE,
    workspace: str = DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
) -> Agent:

    # 1. Setup Model
    model = create_model(provider, model_id, base_url, api_key)
    settings = ModelSettings(timeout=32400.0)

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

    # 3. Prepare Executor Tools (Sandbox + Repo Delegates)
    executor_tools_list = []
    executor_toolsets_list = []
    memory_tools = []

    # Dynamic Sandbox Tool
    async def run_python_in_sandbox(dependencies: List[str], code: str) -> str:
        """
        Run Python code in a sandboxed environment.
        Specify dependencies (e.g., ['numpy', 'pygame-ce']) to install them automatically via micropip.
        The sandbox is isolated and supports Pyodide-compatible packages.
        Use 'pygame-ce' for pygame support.
        """
        try:
            if code_sandbox is None:
                return "Error: mcp_run_python not installed."
            async with code_sandbox(dependencies=dependencies) as sandbox:
                result = await sandbox.eval(code)
                output = f"Stdout: {result.stdout}\nStderr: {result.stderr}\nReturn Value: {result.return_value}"
                if result.error:
                    output += f"\nError: {result.error}"
                return output
        except Exception as e:
            return f"Error executing sandbox: {str(e)}"

    # WRAP IN TOOL OBJECT TO PREVENT PYDANTIC-AI FROM TREATING AS DYNAMIC TOOLSET
    sandbox_tool = Tool(run_python_in_sandbox)
    executor_tools_list.append(sandbox_tool)
    # Register for dynamic usage
    available_tools_registry["python_sandbox"] = [sandbox_tool]

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
                # B. Mem0 Memory Server
                elif server_name == "mem0":
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as temp_config:
                        json.dump(
                            {"mcpServers": {server_name: server_config}}, temp_config
                        )
                        temp_config_path = temp_config.name
                    try:
                        loaded_mem_tools = load_mcp_servers(temp_config_path)
                        memory_tools = loaded_mem_tools
                        available_tools_registry["memory_tools"] = loaded_mem_tools
                        # Add to executor tools
                        executor_tools_list.extend(loaded_mem_tools)
                        logger.info(
                            f"Loaded {len(loaded_mem_tools)} Mem0 Memory tools."
                        )
                    except Exception as e:
                        logger.error(f"Failed to load mem0 tools: {e}")
                    finally:
                        if os.path.exists(temp_config_path):
                            os.remove(temp_config_path)

                # C. Smart Coding Repos
                elif server_name.startswith("smart-coding-"):
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as temp_config:
                        json.dump(
                            {"mcpServers": {server_name: server_config}}, temp_config
                        )
                        temp_config_path = temp_config.name

                    try:
                        codebase_tools = load_mcp_servers(temp_config_path)

                        # Create Child Agent (for Delegation)
                        codebase_agent = Agent(
                            model=model,
                            system_prompt=f"You are the {server_name} Codebase Agent.\nGoal: Manage the repository '{server_name}'.\nYou have full access to search and modify THIS repository.",
                            toolsets=codebase_tools,
                            tool_timeout=32400.0,
                            model_settings=settings,
                            name=server_name,
                            retries=3,
                        )
                        logger.info(f"Created Child Agent for {server_name}")

                        # Create Delegate Tool
                        def make_repo_delegator(agent_instance, s_name):
                            async def delegate_to_repo(
                                ctx: RunContext[Any], instruction: str
                            ) -> str:
                                """
                                Delegate a specific instruction to the Code Base Agent.
                                The Codebase Agent can search code, read files, and manage git.
                                """
                                result = await agent_instance.run(instruction)
                                return str(result.output)

                            delegate_to_repo.__name__ = (
                                f"search_codebase_{s_name.replace('-', '_')}"
                            )
                            return delegate_to_repo

                        delegator = make_repo_delegator(codebase_agent, server_name)
                        executor_tools_list.append(delegator)

                        # Register the DELEGATOR for dynamic usage (so other agents can manage this repo)
                        available_tools_registry[f"manage_{server_name}"] = [delegator]
                        # Also register the RAW tools if someone wants direct access (risky but permitted)
                        available_tools_registry[f"tools_{server_name}"] = (
                            codebase_tools
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to setup repo agent for {server_name}: {e}"
                        )
                    finally:
                        if os.path.exists(temp_config_path):
                            os.remove(temp_config_path)

        except Exception as e:
            logger.error(f"Error parsing MCP config: {e}")

    async def match_pyodide_packages(ctx: RunContext[Any], query_package: str) -> str:
        """
        Check if a package is available in Pyodide.
        Returns the exact package name/version from the supported list, or suggests closest match.
        Use this to validate dependencies like 'numpy', 'pandas', 'pygame' (which might be 'pygame-ce').
        Example: query_package="pygame" -> Returns info about "pygame-ce".
        """
        all_packages = fetch_pyodide_packages()  # ← your function that gets the list

        # 1. Check for known remappings
        remappings = {
            "pygame": "pygame-ce",
            "opencv": "opencv-python",
            "sklearn": "scikit-learn",
            "pil": "pillow",
        }

        query_lower = query_package.lower().strip()

        # If user asks for 'pygame', we immediately suggest 'pygame-ce' if present
        if query_lower in remappings:
            remapped_name = remappings[query_lower]
            # Verify if remapped name is actually in the list
            # The list format is "Name (Version)"
            for pkg in all_packages:
                if pkg.split(" ")[0].lower() == remapped_name:
                    return f"Found match for '{query_package}' (remapped to '{remapped_name}'): {pkg}"

        # 2. Native Search
        matches = []
        for pkg in all_packages:
            name = pkg.split(" ")[0].lower()
            # Exact match?
            if name == query_lower:
                return f"Found exact match for '{query_package}': {pkg}"

            # Partial match
            if query_lower in name:
                matches.append(pkg)

        if matches:
            # Sort matches to prioritize shorter names (likely closer to exact)
            matches.sort(key=len)
            return f"Found matches for '{query_package}': {', '.join(matches[:10])}"  # Limit to top 10
        else:
            # Fallback: Just return the original name, but warn
            return (
                f"No direct match found for '{query_package}' in Pyodide built-in list. "
                "It might still be installable via micropip pure python wheels, "
                "or it is not supported. Use the original name if you are sure."
            )

    # 2. Optionally wrap it in a Tool object (gives more control, but not always needed)
    pyodide_matcher_tool = Tool(match_pyodide_packages)
    executor_tools_list.append(pyodide_matcher_tool)

    # 4. Define Functional Agents
    planner_agent = Agent(
        model=model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        model_settings=settings,
        output_type=PRD,
        tools=[pyodide_matcher_tool] + memory_tools,
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
        model_settings=settings,
        tools=executor_tools_list,
        toolsets=executor_toolsets_list,
        tool_timeout=32400.0,
        output_type=Task,
        name="Executor",
        retries=3,
    )

    validator_agent = Agent(
        model=model,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        model_settings=settings,
        tools=executor_tools_list,
        toolsets=executor_toolsets_list,
        tool_timeout=32400.0,
        output_type=Task,
        name="Validator",
        retries=3,
    )

    repository_manager_agent = Agent(
        model=model,
        system_prompt=REPOSITORY_MANAGER_SYSTEM_PROMPT,
        toolsets=rm_tools + master_skills,
        tool_timeout=32400.0,
        model_settings=settings,
        name="Repository Manager",
        retries=3,
    )

    documentation_agent = Agent(
        model=model,
        system_prompt=DOCUMENTATION_AGENT_SYSTEM_PROMPT,
        toolsets=rm_tools + master_skills,
        tool_timeout=32400.0,
        model_settings=settings,
        name="Documentation Agent",
        retries=3,
    )

    logger.info("Created Specialist Agents")

    # 5. Supervisor Tools

    supervisor = Agent(
        model=model,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        tools=memory_tools,
        name=AGENT_NAME,
        model_settings=settings,
        retries=3,
    )
    logger.info("Created Supervisor Agent")

    @planner_agent.tool
    @supervisor.tool
    async def manage_repositories(ctx: RunContext[Any], instruction: str) -> str:
        """
        Delegate general repository management tasks (listing, cloning, bulk pulling) to the Repository Manager.
        Use this for git operations that affect multiple repos or the workspace.
        """
        result = await repository_manager_agent.run(instruction)
        return str(result.output)

    @supervisor.tool
    async def plan_product_requirements_document(
        ctx: RunContext[Any], user_prompt: str
    ) -> PRD:
        """Generate an initial PRD (Product Requirements Document) from the user's high-level prompt."""
        logger.info(
            f"[PLANNER] Starting plan_product_requirements_document. Prompt: {user_prompt[:50]}..."
        )
        try:
            # Inject workspace into context
            enhanced_prompt = (
                f"{user_prompt}\n\nContext: The workspace root is '{workspace}'."
            )
            result = await planner_agent.run(enhanced_prompt)

            # Force project_root path
            prd = result.output
            if prd.project_name:
                import os

                prd.project_root = os.path.join(workspace, prd.project_name)

            logger.info("[PLANNER] Successfully generated PRD.")
            return prd
        except Exception as e:
            logger.error(f"[PLANNER] Failed to generate PRD: {e}", exc_info=True)
            return f"Error generating PRD: {str(e)}"

    @supervisor.tool
    async def elicit_product_requirements_document(
        ctx: RunContext[Any], prd: PRD
    ) -> PRD:
        """Elicit approval for the PRD (Product Requirements Document) from the User."""
        current_prd = prd
        for _ in range(10):
            result = await product_manager_agent.run(
                f"Review this PRD:\n{current_prd.model_dump_json()}"
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

                current_prd = await _feedback_loop_pm(current_prd, user_response)

        return current_prd

    async def _feedback_loop_pm(prd: PRD, user_response: str) -> PRD:
        res = await product_manager_agent.run(
            f"Current PRD (Product Requirements Document): {prd.model_dump_json()}\nUser Answer to your question: {user_response}\nUpdate and Approve."
        )
        if isinstance(res.output, ElicitationRequest):
            pass
        return res.output

    @supervisor.tool
    async def execute_task(ctx: RunContext[Any], task: Task) -> Task:
        """Execute one task."""
        try:
            result = await executor_agent.run(
                f"Execute this task:\n{task.model_dump_json()}"
            )

            # Capture history
            history_log = []
            for msg in result.all_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        # Check for ToolCallPart
                        if part.part_kind == "tool-call":
                            history_log.append(
                                f"Executor: {part.tool_name}({part.args})"
                            )

            task_out = result.output
            if isinstance(task_out, Task):
                task_out.execution_history.extend(history_log)
                return task_out

            # Fallback if output is not Task
            return task
        except Exception as e:
            logger.exception("Error executing task")
            if task.notes:
                task.notes += f"\n\n[FATAL ERROR]: {e}"
            else:
                task.notes = f"[FATAL ERROR]: {e}"
            task.passes = False
            return task

    @supervisor.tool
    async def validate_task(ctx: RunContext[Any], task: Task) -> Task:
        """Validate a completed task."""
        try:
            result = await validator_agent.run(
                f"Validate this task:\n{task.model_dump_json()}"
            )

            # Capture history
            history_log = []
            for msg in result.all_messages():
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        # Check for ToolCallPart
                        if part.part_kind == "tool-call":
                            history_log.append(
                                f"Validator: {part.tool_name}({part.args})"
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

    @supervisor.tool
    async def update_documentation(ctx: RunContext[Any], instruction: str) -> str:
        """
        Delegate documentation tasks to the Documentation Agent.
        Use this to read, review, or update README.md files.
        """
        result = await documentation_agent.run(instruction)
        return str(result.output)

    @supervisor.tool
    async def finalize_prd_and_diagram(ctx: RunContext[Any], prd: PRD) -> PRD:
        """
        Aggregate execution history from all tasks in the PRD and generate a Mermaid diagram.
        Call this BEFORE update_documentation or as the final step.
        """
        full_history = []
        for story in prd.stories:
            full_history.extend(story.execution_history)

        prd.execution_history = full_history
        prd.mermaid_diagram = generate_mermaid_diagram(full_history)
        prd.last_agent = "Supervisor"
        return prd

    # Register supervisor tools for dynamic usage
    available_tools_registry["execute_task"] = [Tool(execute_task)]
    available_tools_registry["validate_task"] = [Tool(validate_task)]
    available_tools_registry["update_documentation"] = [Tool(update_documentation)]

    @supervisor.tool
    async def create_specialized_agent(
        ctx: RunContext[Any],
        name: str = Field(..., description="Short lowercase name for the new agent"),
        system_prompt: str = Field(
            ..., description="System prompt for the child agent"
        ),
        tool_names: List[str] = Field(
            default_factory=list,
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
            tool_timeout=32400.0,
            name=name,
            retries=3,
        )

        @supervisor.tool(name=f"run_{name}")
        async def dynamic_tool(ctx: RunContext[Any], input_data: str) -> str:
            """Execute the specialized agent."""
            result = await child_agent.run(input_data)
            return str(result.output)

        return f"Created tool run_{name} with tools: {tool_names}"

    ## Re-implement agent later once tool overlap issue is resolved. Create project triggers agent to create project, not the local file git repo project creation. Need to fine tune the prompts.
    # # 6. Register External Agents (Standard Pattern)
    # if enable_gitlab_agent:
    #     try:
    #         # Dynamic import to avoid hard dependency
    #         from gitlab_api.gitlab_agent import create_agent as create_gitlab_agent

    #         # Create the instance
    #         gitlab_agent_instance = create_gitlab_agent()

    #         @supervisor.tool
    #         async def run_gitlab_agent(ctx: RunContext[Any], instruction: str) -> str:
    #             """
    #             Delegate functionality to the GitLab Agent.
    #             This agent can interact with the GitLab API (issues, merge requests, pipelines, etc).
    #             """
    #             # Run the agent with the instruction
    #             result = await gitlab_agent_instance.run(instruction)
    #             return str(result.output)

    #         logger.info("External GitLab Agent enabled and registered as 'run_gitlab_agent'.")

    #     except ImportError:
    #         logger.warning(
    #             "gitlab-api module not found. GitLab Agent disabled. "
    #             "Install with 'pip install gitlab-api[all]'"
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to initialize GitLab Agent: {e}")

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
    enable_gitlab_agent: bool = DEFAULT_GITLAB_AGENT_ENABLE,
):
    logger.info(
        f"Starting {AGENT_NAME} with provider={provider}, model={model_id}, mcp={mcp_url} | {mcp_config}"
    )

    agent = create_agent(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        skills_directory=skills_directory,
        enable_gitlab_agent=enable_gitlab_agent,
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hasattr(a2a_app, "router"):
            async with a2a_app.router.lifespan_context(a2a_app):
                yield
        else:
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

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream()
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


def configure_mcp_servers(
    base_directory: str,
    mcp_config_path: str,
    enable_smart_coding: bool = False,
    enable_python_sandbox: bool = False,
):
    if not enable_smart_coding and not enable_python_sandbox:
        return

    config = {}
    if os.path.exists(mcp_config_path):
        try:
            with open(mcp_config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading MCP config from {mcp_config_path}: {e}")
            return

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    updates_made = False

    if enable_smart_coding:
        if not os.path.exists(base_directory):
            logger.warning(f"Directory not found: {base_directory}")
        else:
            git_projects = []
            base_path = Path(base_directory)
            try:
                for item in base_path.iterdir():
                    if item.is_dir() and (item / ".git").exists():
                        git_projects.append(item)
            except Exception as e:
                logger.error(f"Error scanning directory {base_directory}: {e}")

            if git_projects:
                logger.info(
                    f"Found {len(git_projects)} git projects for Smart Coding MCP."
                )
                for project in git_projects:
                    server_name = f"smart-coding-{project.name}"
                    if server_name not in config["mcpServers"]:
                        config["mcpServers"][server_name] = {
                            "command": "smart-coding-mcp",
                            "args": ["--workspace", str(project.absolute())],
                            "timeout": 200000,
                        }
                        updates_made = True
                        logger.info(f"Added MCP server configuration for {server_name}")
                    else:
                        config["mcpServers"][server_name][
                            "command"
                        ] = "smart-coding-mcp"
                        config["mcpServers"][server_name]["args"] = [
                            "--workspace",
                            str(project.absolute()),
                        ]
                        config["mcpServers"][server_name]["timeout"] = 200000
                        updates_made = True

    if enable_python_sandbox:
        if "python-sandbox" not in config["mcpServers"]:
            config["mcpServers"]["python-sandbox"] = {
                "command": "uvx",
                "args": ["mcp-run-python", "stdio"],
                "timeout": 200000,
            }
            updates_made = True
            logger.info("Added MCP server configuration for python-sandbox")

    if updates_made:
        try:
            with open(mcp_config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Updated MCP config at {mcp_config_path}")
        except Exception as e:
            logger.error(f"Error writing MCP config: {e}")


def agent_server():
    print(f"Repository Manager Agent v{__version__}")
    parser = argparse.ArgumentParser(description=f"Run the {AGENT_NAME} Server")
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
    parser.add_argument(
        "--smart-coding-mcp-enable",
        action="store_true",
        default=DEFAULT_SMART_CODING_MCP_ENABLE,
    )
    parser.add_argument(
        "--python-sandbox-enable",
        action="store_true",
        default=DEFAULT_PYTHON_SANDBOX_ENABLE,
    )
    parser.add_argument("--workspace", default=os.getcwd(), help="Repository Workspace")
    parser.add_argument(
        "--web", action="store_true", default=DEFAULT_ENABLE_WEB_UI, help="Web UI"
    )
    parser.add_argument(
        "--gitlab-agent-enable",
        action="store_true",
        default=DEFAULT_GITLAB_AGENT_ENABLE,
    )
    parser.add_argument(
        "--no-gitlab-agent-enable",
        action="store_false",
        dest="gitlab_agent_enable",
    )
    args = parser.parse_args()

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
        enable_gitlab_agent=args.gitlab_agent_enable,
    )


if __name__ == "__main__":
    agent_server()
