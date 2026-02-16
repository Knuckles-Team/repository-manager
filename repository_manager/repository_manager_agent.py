#!/usr/bin/python

# coding: utf-8
import asyncio
import os
import argparse
import logging
import uvicorn
import json
import tempfile
import re
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager
from typing import Optional, Any, List, Dict
from pydantic_ai import (
    Agent,
    ModelSettings,
    RunContext,
    UsageLimitExceeded,
    ModelRetry,
    UsageLimits,
)
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai_skills import SkillsToolset
from fasta2a import Skill
from pydantic import ValidationError
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
from repository_manager.models import (
    Task,
    ImplementationPlan,
    Clarification,
    PlanStepResult,
    TaskStepResult,
)
from repository_manager.prompts import (
    CORE_SYSTEM_PROMPT,
    ARCHITECT_SYSTEM_PROMPT,
    ENGINEER_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
)

__version__ = "1.3.10"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("pydantic_ai").setLevel(logging.INFO)
logging.getLogger("fastmcp").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(string=os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3-coder-next")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_SKILLS_DIRECTORY = os.getenv("SKILLS_DIRECTORY", get_skills_path())
DEFAULT_PROJECTS_FILE = os.getenv("PROJECTS_FILE", get_projects_file_path())
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_SSL_VERIFY = to_boolean(os.getenv("SSL_VERIFY", "True"))
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.getenv(
    "REPOSITORY_MANAGER_WORKSPACE", "/workspace"
)

DEFAULT_MAX_TOKENS = to_integer(os.getenv("MAX_TOKENS", "16384"))
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
    "A Supervisor Agent that orchestrates a team of child agents (Architect, Engineer, QA) "
    "to manage git repositories using parallel execution."
)


SUPERVISOR_SYSTEM_PROMPT = (
    "You are the Repository Manager Supervisor.\n"
    "You orchestrate a team of agents to manage Code Repositories.\n\n"
    "Workflow: \n"
    "1. **Architect**: Designs an implementation plan.\n"
    "2. **Engineer**: Executes tasks in parallel.\n"
    "3. **QA**: Verifies completed tasks.\n\n"
    "Use `run_implementation_process` to start the full workflow."
)


INSTRUCTIONS = (
    "The ImplementationPlan object contains a summary and a list of tasks.\n"
    "→ For EVERY call to repository tools, ensure you use the correct project name if applicable.\n"
    "Never guess, hard-code, or use a project name from memory or previous messages."
)


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    workspace: str = DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
) -> Agent:

    model = create_model(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        ssl_verify=ssl_verify,
        timeout=DEFAULT_TIMEOUT,
    )
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

    available_tools_registry: Dict[str, List[Any]] = {}

    master_skills = []
    if skills_directory and os.path.exists(skills_directory):
        logger.info(f"Loading skills from {skills_directory}")
        loaded_skills = SkillsToolset(directories=[str(skills_directory)])
        master_skills.append(loaded_skills)
        available_tools_registry["git_skills"] = [loaded_skills]

    engineer_tools_list = []
    engineer_toolsets_list = []
    rm_tools = []

    if master_skills:
        engineer_toolsets_list.extend(master_skills)

    if mcp_config and os.path.exists(mcp_config):
        try:
            with open(mcp_config, "r") as f:
                config_data = json.load(f)

            mcp_servers = config_data.get("mcpServers", {})

            for server_name, server_config in mcp_servers.items():

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
                        available_tools_registry["repository_manager_tools"] = rm_tools
                    except Exception as e:
                        logger.error(f"Failed to load repository-manager tools: {e}")
                    finally:
                        if os.path.exists(temp_config_path):
                            os.remove(temp_config_path)

                    if rm_tools:
                        engineer_toolsets_list.extend(rm_tools)
                        logger.info(
                            f"Added {len(rm_tools)} Repository Manager tools to Engineer."
                        )

        except Exception as e:
            logger.error(f"Error parsing MCP config: {e}")

    # --- Architect Agent ---
    architect_agent = Agent(
        model=model,
        system_prompt=f"{CORE_SYSTEM_PROMPT}\n\n{ARCHITECT_SYSTEM_PROMPT}",
        model_settings=settings,
        output_type=PlanStepResult,
        name="Architect",
        retries=3,
        toolsets=rm_tools,  # Architect needs tools to check existing projects (list_projects)
    )

    # --- Engineer Agent ---
    engineer_agent = Agent(
        model=model,
        system_prompt=f"{CORE_SYSTEM_PROMPT}\n\n{ENGINEER_SYSTEM_PROMPT}",
        instructions=INSTRUCTIONS,
        model_settings=settings,
        tools=engineer_tools_list,
        toolsets=engineer_toolsets_list,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        output_type=TaskStepResult,
        name="Engineer",
        retries=3,
    )

    # --- QA Agent ---
    qa_agent = Agent(
        model=model,
        system_prompt=f"{CORE_SYSTEM_PROMPT}\n\n{QA_SYSTEM_PROMPT}",
        instructions=INSTRUCTIONS,
        model_settings=settings,
        tools=engineer_tools_list,
        toolsets=engineer_toolsets_list,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        output_type=TaskStepResult,
        name="QA",
        retries=3,
    )

    logger.info("Created Specialist Agents: Architect, Engineer, QA")

    supervisor = Agent(
        model=model,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        name=AGENT_NAME,
        model_settings=settings,
        retries=3,
    )
    logger.info("Created Supervisor Agent")

    @supervisor.tool
    async def run_implementation_process(
        ctx: RunContext[Any],
        user_prompt: str,
        max_iterations: Optional[int] = 15,
    ) -> str:
        """Trigger the complete implementation process: Plan -> Execute (Parallel) -> Verify."""
        limits = UsageLimits(
            request_limit=50,
            tool_calls_limit=200,
            total_tokens_limit=DEFAULT_TOTAL_TOKENS,
        )

        plan_filename_cache: Dict[str, str] = {}

        def save_plan(plan: ImplementationPlan):
            try:
                if "filename" not in plan_filename_cache:
                    # Create a slug from summary
                    slug = re.sub(r"[^a-z0-9]+", "_", plan.summary.lower()).strip("_")
                    if not slug:
                        slug = "plan"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plan_filename_cache["filename"] = (
                        f"implementation_plan_{slug}_{timestamp}.md"
                    )

                filename = plan_filename_cache["filename"]
                # Save to workspace root
                path = Path(workspace) / filename
                with open(path, "w") as f:
                    f.write(plan.to_markdown())

                # Also symlink or copy to 'implementation_plan.md' for consistency if desired?
                # The user specifically asked for implementation_plan_<summary>_<timestamp>.md
            except Exception as e:
                logger.error(f"Failed to save plan artifact: {e}")

        try:
            # 1. PLAN (Architect)
            architect_prompt = f"{user_prompt} \n\nThe workspace is '{workspace}'. Design an ImplementationPlan."
            plan_result = await run_with_retry(
                architect_agent,
                architect_prompt,
                ctx,
                usage_limits=limits,
            )

            # Initial output could be a Plan or Clarification
            plan_output = plan_result.output.output

            # Clarification Loop
            clarification_attempts = 0
            while isinstance(plan_output, Clarification) and clarification_attempts < 3:
                # In non-interactive mode, we might just have to stop or assume.
                # For now, we simulate user input or just fail.
                # Ideally, this should use notify_user if connected to a real user,
                # but for an autonomous run, we can try to self-answer or abort.
                # Here we assume the user might provide input via a side-channel or we log it.
                logger.info(
                    f"Architect requested clarification: {plan_output.question}"
                )

                # Mock response for autonomous run if user_prompt didn't cover it
                # In a real tool we might pause execution.
                # For now, we treat it as an error if we can't get clarification.
                return f"Architect requested clarification: {plan_output.question}"

            if not isinstance(plan_output, ImplementationPlan):
                return f"Planning failed. Output was: {type(plan_output)}"

            plan = plan_output
            logger.info(f"Plan created: {plan.summary} with {len(plan.tasks)} tasks.")
            save_plan(plan)

            # 2. EXECUTE & VERIFY LOOP
            loop_count = 0
            MAX_LOOPS = 25  # Safety break
            MAX_TASK_RETRIES = 3

            while not plan.is_complete() and loop_count < MAX_LOOPS:
                logger.info(f"--- Loop {loop_count + 1} / {MAX_LOOPS} ---")

                # A. Handle FAILURES (Planning adjustment)
                failed_tasks = [t for t in plan.tasks if t.status == "failed"]
                if failed_tasks:
                    logger.warning(
                        f"Tasks failed: {[t.id for t in failed_tasks]}. Requesting Re-Plan."
                    )

                    replan_prompt = (
                        f"The following tasks have failed after multiple attempts:\n"
                        f"{[t.model_dump_json() for t in failed_tasks]}\n\n"
                        f"Current Plan Context:\n{plan.model_dump_json(exclude={'tasks'})}\n\n"
                        f"Please revise the ImplementationPlan to achieve the goal despite these failures. "
                        f"You may remove the failed tasks and add new ones, or modify the approach."
                    )

                    replan_result = await run_with_retry(
                        architect_agent,
                        replan_prompt,
                        ctx,
                        # Don't pass usage limits to avoid blowing up parent context
                    )

                    new_plan_output = replan_result.output.output
                    if isinstance(new_plan_output, ImplementationPlan):
                        # Merge or Replace?
                        # If we replace, we lose status of other tasks unless Architect is smart.
                        # For simplicity, we assume Architect returns a FULL plan including old (completed) tasks if they are still relevant.
                        # But Architect might not know about 'verified' status if we only passed failed ones.
                        # To be safe, we should pass FULL plan to Architect.
                        # Retrying with FULL plan context in prompt above.

                        # We hope Architect preserves IDs and statuses of non-failed tasks.
                        # Usage of 'merge' logic might be safer but complex.
                        # Let's trust the Architect to regenerate a valid full plan.
                        plan = new_plan_output
                        logger.info(f"Plan revised: {len(plan.tasks)} tasks.")
                        save_plan(plan)
                        # Reset failed tasks loop to process new plan
                        failed_tasks = []
                    else:
                        logger.error("Architect failed to revise plan.")
                        return f"Implementation failed. Architect could not revise plan for failed tasks: {[t.id for t in failed_tasks]}"

                # B. EXECUTION (Engineer)
                # Identify pending tasks
                runnable_tasks = []
                # Simple dependency check
                # completed_ids was unused

                # However, we only mark 'verified' after QA.
                # Let's use get_next_tasks() from model which logic checks 'passes' (which typically means verified/done).
                # But 'passes' is set by QA agent (or Engineer if self-verifying?).
                # In our model, 'passes' is bool.

                next_tasks = plan.get_next_tasks()
                for t in next_tasks:
                    if t.status == "pending":
                        if t.attempt_count >= MAX_TASK_RETRIES:
                            t.status = "failed"
                            t.notes = (
                                (t.notes or "")
                                + f"\n[FAILED]: Max retries ({MAX_TASK_RETRIES}) reached."
                            )
                            # Trigger replan in next loop
                        else:
                            t.status = "in_progress"
                            t.attempt_count += 1
                            runnable_tasks.append(t)

                if runnable_tasks:
                    logger.info(
                        f"Spawning Engineers for tasks: {[t.id for t in runnable_tasks]}"
                    )

                    async def run_engineer(t: Task):
                        # Use a fresh limit for child agent to prevent 'Total Tokens Exceeded' on the parent
                        # arising from child usage.
                        # We DO NOT pass ctx.usage to avoid accumulating child usage onto supervisor's limit.
                        child_limits = UsageLimits(
                            request_limit=50,
                            total_tokens_limit=None,  # Let child run as much as needed per task
                        )

                        try:
                            # Note: We are creating a NEW prompt for every run.
                            # engineer_agent is stateless (no history passed).
                            res = await engineer_agent.run(
                                f"Implement this task:\n{t.model_dump_json()}\n\nFull Plan Context:\n{plan.model_dump_json(exclude={'tasks'})}",
                                deps=ctx.deps,
                                usage_limits=child_limits,
                            )
                            # res.data is TaskStepResult
                            task_out = res.data.output
                            if isinstance(task_out, Task):
                                task_out.status = "implemented"
                                return task_out
                            return t
                        except Exception as e:
                            logger.error(f"Engineer failed task {t.id}: {e}")
                            t.notes = (t.notes or "") + f"\n[ERROR]: {str(e)}"
                            t.status = "pending"  # Reset to pending to retry (attempt_count was already incremented)
                            return t

                    engineer_results = await asyncio.gather(
                        *[run_engineer(t) for t in runnable_tasks]
                    )

                    # Update Plan
                    for res_task in engineer_results:
                        for i, t in enumerate(plan.tasks):
                            if t.id == res_task.id:
                                plan.tasks[i] = res_task
                                break

                    save_plan(plan)

                # C. VERIFICATION (QA)
                verify_tasks = [t for t in plan.tasks if t.status == "implemented"]
                if verify_tasks:
                    logger.info(
                        f"Spawning QA for tasks: {[t.id for t in verify_tasks]}"
                    )

                    async def run_qa(t: Task):
                        child_limits = UsageLimits(
                            request_limit=50, total_tokens_limit=None
                        )
                        try:
                            res = await qa_agent.run(
                                f"Verify this task:\n{t.model_dump_json()}\n\nFull Plan Context:\n{plan.model_dump_json(exclude={'tasks'})}",
                                deps=ctx.deps,
                                usage_limits=child_limits,
                            )
                            # res.data is TaskStepResult
                            task_out = res.data.output
                            if isinstance(task_out, Task):
                                if task_out.status == "verified":
                                    pass  # Already marked as verified by QA
                                else:
                                    task_out.status = "pending"
                                    # Note: We don't increment attempt_count here, but we could if we consider QA failure a 'try'.
                                    # Since we send back to Engineer, the Engineer's next run will increment attempt_count.
                                    # This effectively counts Implementation+QA cycles.
                                    task_out.notes = (
                                        task_out.notes or ""
                                    ) + "\n[QA FAILED]: Verification failed."
                                return task_out
                            return t
                        except Exception as e:
                            logger.error(f"QA failed task {t.id}: {e}")
                            # If QA crashes, maybe we assume it's NOT verified?
                            # Leave as implemented? Or retry QA?
                            # Let's retry QA by leaving as implemented (it will be picked up next loop)
                            # But we need to avoid infinite loop.
                            # Let's push back to pending to be safe.
                            t.status = "pending"
                            t.notes = (t.notes or "") + f"\n[QA ERROR]: {str(e)}"
                            return t

                    qa_results = await asyncio.gather(
                        *[run_qa(t) for t in verify_tasks]
                    )

                    # Update Plan
                    for res_task in qa_results:
                        for i, t in enumerate(plan.tasks):
                            if t.id == res_task.id:
                                plan.tasks[i] = res_task
                                break

                    save_plan(plan)

                # D. Check Progress
                loop_count += 1
                plan.iteration_count = loop_count

                # If no tasks ran, check if we are stuck
                if not runnable_tasks and not verify_tasks and not plan.is_complete():
                    # Maybe dependencies are blocked?
                    # Check if any tasks are pending but blocked
                    pending = [t for t in plan.tasks if t.status == "pending"]
                    if pending:
                        # Blocked tasks?

                        logger.warning(
                            "No tasks ran this iteration but plan is incomplete. Dependencies might be blocked."
                        )
                        # Break loop to avoid infinite spin if logic is stuck
                        # But 'failed' logic top of loop should handle if max retries hit.
                        if all(t.attempt_count == 0 for t in pending):
                            # Nothing started?
                            pass
                        else:
                            # We might be waiting for something that never finishes?
                            pass
                    else:
                        # e.g. all 'failed' and waiting for replan?
                        pass

            # 4. FINALIZE
            if plan.is_complete():
                return f"Implementation completed: {plan.summary}. All tasks verified."
            else:
                remaining = [t.id for t in plan.tasks if t.status != "verified"]
                return f"Implementation stopped after {loop_count} loops. Remaining/Failed tasks: {remaining}."

        except UsageLimitExceeded as e:
            logger.error(f"Usage limit hit in process: {e}")
            return f"Safety limit hit: {str(e)}"
        except Exception as e:
            logger.exception(f"Error in process: {e}")
            return f"Error in process: {str(e)}"

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
                raise
            except ValidationError as e:
                # Basic output structure validation failed
                if attempt == max_attempts:
                    raise
                logger.warning(f"Validation error: {e}. Retrying.")
                raise ModelRetry(
                    f"Output validation failed: {str(e)}. Refine and retry."
                )
            except Exception as e:
                if attempt == max_attempts:
                    raise
                logger.warning(f"Retry {attempt} for {agent.name}: {str(e)}")
                await asyncio.sleep(2**attempt)

    @supervisor.tool
    async def manage_repositories(ctx: RunContext[Any], instruction: str) -> str:
        """
        Directly access repository tools for ad-hoc requests.
        Delegates to the Engineer agent.
        """
        result = await engineer_agent.run(
            f"Execute this instruction: {instruction} (Task ID: 0, status: pending)",  # Dummy task context
            deps=ctx.deps,
            usage=ctx.usage,
        )
        if isinstance(result.output, TaskStepResult):
            return f"Result: {result.output.output.action}\nNotes: {result.output.output.notes}"
        return str(result.output)

    return supervisor


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    debug: Optional[bool] = DEFAULT_DEBUG,
    host: Optional[str] = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    enable_web_ui: bool = DEFAULT_ENABLE_WEB_UI,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
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
        ssl_verify=ssl_verify,
        # timeout=DEFAULT_TIMEOUT, # create_agent doesn't accept timeout? Check signature.
    )

    if skills_directory and os.path.exists(skills_directory):
        skills = load_skills_from_directory(skills_directory)
        logger.info(f"Loaded {len(skills)} skills from {skills_directory}")
    else:
        skills = [
            Skill(
                id="repository_manager_supervisor",
                name="Repository Manager Supervisor",
                description="Supervisor for Repository Management",
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
        if hasattr(a2a_app, "router") and hasattr(a2a_app.router, "lifespan_context"):
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
    args = parser.parse_args()

    create_agent_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    agent_server()
