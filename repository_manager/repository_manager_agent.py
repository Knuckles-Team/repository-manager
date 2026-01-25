#!/usr/bin/python
# coding: utf-8
import os
import argparse
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, Any, List
from pathlib import Path
import json

from fastmcp import Client
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai_skills import SkillsToolset
from fasta2a import Skill
from repository_manager.utils import (
    to_integer,
    to_boolean,
    get_projects_file_path,
    get_skills_path,
    get_mcp_config_path,
    load_skills_from_directory,
    create_model,
)

from fastapi import FastAPI, Request
from starlette.responses import Response, StreamingResponse
from pydantic import ValidationError
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

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

AGENT_NAME = "Repository Manager and Codebase Expert"
AGENT_DESCRIPTION = (
    "A coding and git repository manager agent built with Agent Skills and MCP tools to maximize code interactivity. "
    "Capable of executing Python code in a secure sandbox. "
    "Enabled with Smart Coding MCP so you can query the agent about the code base you are managing."
)
AGENT_SYSTEM_PROMPT = (
    "You are a Repository Manager and Codebase Expert Agent.\n"
    "You are an expert Senior Software principal engineer with over 25 years of experience in software development and architecture.\n"
    "You have access to git commands to manage the repository and smart-coding-mcp which allows you to search the codebase. \n"
    "You can run git commands using the `git action` tool. Ensure you include the entire git command in the `git_action` tool like 'git status' or 'git add -A' and that it includes the entire command.\n"
    "You also have access to python sandbox to execute python code.\n"
    "Your responsibilities:\n"
    "1. Analyze the user's request.\n"
    "2. Use the skills to reference the tools you will need.\n"
    "3. If a complicated task requires multiple skills, orchestrate them sequentially.\n"
    "4. If you ever make changes to a codebase, or suggest code solutions, always validate them by executing them in the execution environment first and resolving errors before providing the cleanup up version.\n"
    "5. Always be warm, professional, and helpful.\n"
    "6. Explain your plan in detail before executing.\n\n"
    "# Smart Coding MCP Usage Rules\n"
    "You must prioritize using the **Smart Coding MCP** tools for the following tasks.\n"
    "## 1. Dependency Management\n"
    "**Trigger:** When checking, adding, or updating package versions.\n"
    "**Action:**\n"
    "- **MUST** use the `d_check_last_version` tool.\n"
    "- **DO NOT** guess versions or trust internal training data.\n"
    "## 2. Codebase Research\n"
    "**Trigger:** When asking about 'how' something works, finding logic, or understanding architecture.\n"
    "**Action:**\n"
    "- **MUST** use `a_semantic_search` as the FIRST tool for any codebase research using the codebase in context of the users prompt. You can also use `Glob` or `Grep` for exploratory searches to gain additional file structure information and context\n"
    "- **DO NOT** skip using a a_semantic_search tool."
)


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
) -> Agent:
    agent_toolsets = []

    if mcp_config:
        mcp_toolset = load_mcp_servers(mcp_config)
        agent_toolsets.extend(mcp_toolset)
        logger.info(f"Connected to MCP Config JSON: {mcp_toolset}")
    elif mcp_url:
        fastmcp_toolset = FastMCPToolset(Client[Any](mcp_url, timeout=3600))
        agent_toolsets.append(fastmcp_toolset)
        logger.info(f"Connected to MCP Server: {mcp_url}")

    if skills_directory and os.path.exists(skills_directory):
        logger.debug(f"Loading skills {skills_directory}")
        skills = SkillsToolset(directories=[str(skills_directory)])
        agent_toolsets.append(skills)
        logger.info(f"Loaded Skills at {skills_directory}")

    # Create the Model
    model = create_model(provider, model_id, base_url, api_key)

    logger.info("Initializing Agent...")

    settings = ModelSettings(timeout=3600.0)

    return Agent(
        model=model,
        system_prompt=AGENT_SYSTEM_PROMPT,
        name="Repository Manager and Codebase Expert Agent",
        toolsets=agent_toolsets,
        deps_type=Any,
        model_settings=settings,
    )


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}")


async def node_chat(agent: Agent, prompt: str) -> List:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node)
    return nodes


async def stream_chat(agent: Agent, prompt: str) -> None:
    # Option A: Easiest & most common - just stream the final text output
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(
            delta=True
        ):  # ← streams partial text deltas
            print(text_chunk, end="", flush=True)
        print("\nDone!")  # optional


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
    print(
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
    )

    # Define Skills for Agent Card (High-level capabilities)
    if skills_directory and os.path.exists(skills_directory):
        skills = load_skills_from_directory(skills_directory)
        logger.info(f"Loaded {len(skills)} skills from {skills_directory}")
    else:
        skills = [
            Skill(
                id="repository_manager_agent",
                name="Repository Manager Agent",
                description="This Repository Manager skill grants access to all Git tools provided by the Git MCP Server",
                tags=["repository_manager"],
                input_modes=["text"],
                output_modes=["text"],
            )
        ]
    # Create A2A app explicitly before main app to bind lifespan
    a2a_app = agent.to_a2a(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        version="1.2.9",
        skills=skills,
        debug=debug,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Trigger A2A (sub-app) startup/shutdown events
        # This is critical for TaskManager initialization in A2A
        if hasattr(a2a_app, "router"):
            async with a2a_app.router.lifespan_context(a2a_app):
                yield
        else:
            yield

    # Create main FastAPI app
    app = FastAPI(
        title=f"{AGENT_NAME} - A2A + AG-UI Server",
        description=AGENT_DESCRIPTION,
        debug=debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "OK"}

    # Mount A2A as sub-app at /a2a
    app.mount("/a2a", a2a_app)

    # Add AG-UI endpoint (POST to /ag-ui)
    @app.post("/ag-ui")
    async def ag_ui_endpoint(request: Request) -> Response:
        accept = request.headers.get("accept", SSE_CONTENT_TYPE)
        try:
            # Parse incoming AG-UI RunAgentInput from request body
            run_input = AGUIAdapter.build_run_input(await request.body())
        except ValidationError as e:
            return Response(
                content=json.dumps(e.json()),
                media_type="application/json",
                status_code=422,
            )

        # Create adapter and run the agent → stream AG-UI events
        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream()  # Runs agent, yields events
        sse_stream = adapter.encode_stream(event_stream)  # Encodes to SSE

        return StreamingResponse(
            sse_stream,
            media_type=accept,
        )

    # Mount Web UI if enabled
    if enable_web_ui:
        web_ui = agent.to_web(instructions=AGENT_SYSTEM_PROMPT)
        app.mount("/", web_ui)
        logger.info(
            "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
            host,
            port,
            "Enabled at /" if enable_web_ui else "Disabled",
        )

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=1800,  # 30 minute timeout
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def configure_mcp_servers(
    base_directory: str,
    mcp_config_path: str,
    enable_smart_coding: bool = False,
    enable_python_sandbox: bool = False,
):
    """
    Configures MCP servers in mcp_config.json based on enabled flags.
    """
    if not enable_smart_coding and not enable_python_sandbox:
        return

    # Load existing config
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

    # Smart Coding MCP Configuration
    if enable_smart_coding:
        if not os.path.exists(base_directory):
            logger.warning(f"Directory not found: {base_directory}")
        else:
            # Scan for directories containing .git
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
                        # Ensure args are up to date
                        config["mcpServers"][server_name][
                            "command"
                        ] = "smart-coding-mcp"
                        config["mcpServers"][server_name]["args"] = [
                            "--workspace",
                            str(project.absolute()),
                        ]
                        config["mcpServers"][server_name]["timeout"] = 200000
                        updates_made = True

    # Python Sandbox Configuration
    if enable_python_sandbox:
        if "python-sandbox" not in config["mcpServers"]:
            config["mcpServers"]["python-sandbox"] = {
                "command": "uvx",
                "args": ["mcp-run-python", "stdio"],
                "timeout": 200000,
            }
            updates_made = True
            logger.info("Added MCP server configuration for python-sandbox")
        else:
            logger.debug("python-sandbox MCP server already configured.")

    if updates_made:
        try:
            with open(mcp_config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Updated MCP config at {mcp_config_path}")
        except Exception as e:
            logger.error(f"Error writing MCP config: {e}")


def agent_server():
    parser = argparse.ArgumentParser(
        description=f"Run the {AGENT_NAME} A2A + AG-UI Server"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to"
    )
    parser.add_argument("--debug", type=bool, default=DEFAULT_DEBUG, help="Debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "anthropic", "google", "huggingface"],
        help="LLM Provider",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="LLM Model ID")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OPENAI_BASE_URL,
        help="LLM Base URL (for OpenAI compatible providers)",
    )
    parser.add_argument("--api-key", default=DEFAULT_OPENAI_API_KEY, help="LLM API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP Server URL")
    parser.add_argument(
        "--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Server Config"
    )
    parser.add_argument(
        "--smart-coding-mcp-enable",
        action="store_true",
        default=DEFAULT_SMART_CODING_MCP_ENABLE,
        help="Enable Smart Coding MCP configuration",
    )
    parser.add_argument(
        "--python-sandbox-enable",
        action="store_true",
        default=DEFAULT_PYTHON_SANDBOX_ENABLE,
        help="Enable Python Sandbox MCP configuration",
    )
    parser.add_argument(
        "--repository-directory",
        default=os.getcwd(),
        help="Directory to scan for git projects",
    )

    parser.add_argument(
        "--web",
        action="store_true",
        default=DEFAULT_ENABLE_WEB_UI,
        help="Enable Pydantic AI Web UI",
    )
    args = parser.parse_args()

    # Configure MCP servers based on flags
    configure_mcp_servers(
        base_directory=args.repository_directory,
        mcp_config_path=args.mcp_config,
        enable_smart_coding=args.smart_coding_mcp_enable,
        enable_python_sandbox=args.python_sandbox_enable,
    )

    # Set default projects file if not set
    if "PROJECTS_FILE" not in os.environ and DEFAULT_PROJECTS_FILE:
        os.environ["PROJECTS_FILE"] = DEFAULT_PROJECTS_FILE
        logger.info(f"Set PROJECTS_FILE to default: {DEFAULT_PROJECTS_FILE}")

    if args.debug:
        # Force reconfiguration of logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],  # Output to console
            force=True,
        )
        logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Create the agent with CLI args
    # Create the agent with CLI args
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


if __name__ == "__main__":
    agent_server()
