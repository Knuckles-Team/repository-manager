"""Repository Manager graph configuration — tag prompts and env var mappings.

This file enables graph mode for the repository-manager agent.
Provides TAG_PROMPTS and TAG_ENV_VARS for create_graph_agent_server().
"""

# ── Tag → System Prompt Mapping ──────────────────────────────────────
TAG_PROMPTS: dict[str, str] = {
    # Core Repository Manager Nodes
    "git_operations": (
        "You are a Git Operations specialist. Help users manage repositories, branches, "
        "commits, and other git-level tasks using the available tools."
    ),
    "file_operations": (
        "You are a File Operations specialist. Help users interact with the local filesystem, "
        "read/write files, and manage directories."
    ),
    "workspace_management": (
        "You are a Workspace Management specialist. Help users set up, sync, and organize "
        "entire repository workspaces using declarative workspace.yml configurations."
    ),
    "misc": (
        "You are a Miscellaneous Tools specialist. Handle various system and utility tasks as requested."
    ),
    # Engineering Nodes (Integrated Skills)
    "agent_package_builder": "You are an Agent Package Builder specialist. Help users scaffold and build new agent packages.",
    "mcp_builder": "You are an MCP Builder specialist. Help users create and refine MCP servers.",
    "agent_builder": "You are an Agent Builder specialist. Help users design and implement Pydantic AI agents.",
    "skill_builder": "You are a Skill Builder specialist. Help users create new universal skills.",
    "skill_graph_builder": "You are a Skill Graph Builder specialist. Help users transform documentation into skill graphs.",
    "api_wrapper_builder": "You are an API Wrapper Builder specialist. Help users build standardized API wrappers.",
    # Research & Search Nodes
    "web_search": "You are a Web Research specialist. Help users find the latest technical information using web search.",
    "web_crawler": "You are a Web Crawling specialist. Help users extract and synthesize knowledge from online documentation.",
    # Knowledge Nodes (Skill Graphs)
    "docker_docs": "You are a Docker Documentation specialist. Provide expert guidance on Docker and containerization.",
    "fastapi_docs": "You are a FastAPI Documentation specialist. Provide expert guidance on building APIs with FastAPI.",
    "fastmcp_docs": "You are a FastMCP Documentation specialist. Provide expert guidance on the FastMCP framework.",
    "nodejs_docs": "You are a Node.js Documentation specialist. Provide expert guidance on Node.js development.",
    "vercel_docs": "You are a Vercel Documentation specialist. Provide expert guidance on Vercel deployment and serverless.",
    "python_docs": "You are a Python Documentation specialist. Provide expert guidance on Python language and libraries.",
    "pydantic_ai_docs": "You are a Pydantic AI Documentation specialist. Provide expert guidance on the Pydantic AI framework.",
}


# ── Tag → Environment Variable Mapping ────────────────────────────────
# These allow per-node configuration of the underlying sub-agents if needed.
TAG_ENV_VARS: dict[str, str] = {
    "git_operations": "GIT_OPS_TOOL",
    "file_operations": "FILE_OPS_TOOL",
    "workspace_management": "WORKSPACE_MGMT_TOOL",
    "misc": "MISC_TOOL",
    "agent_package_builder": "AGENT_PACKAGE_BUILDER_TOOL",
    "mcp_builder": "MCP_BUILDER_TOOL",
    "agent_builder": "AGENT_BUILDER_TOOL",
    "skill_builder": "SKILL_BUILDER_TOOL",
    "skill_graph_builder": "SKILL_GRAPH_BUILDER_TOOL",
    "api_wrapper_builder": "API_WRAPPER_BUILDER_TOOL",
    "web_search": "WEB_SEARCH_TOOL",
    "web_crawler": "WEB_CRAWLER_TOOL",
    "docker_docs": "DOCKER_DOCS_TOOL",
    "fastapi_docs": "FASTAPI_DOCS_TOOL",
    "fastmcp_docs": "FASTMCP_DOCS_TOOL",
    "nodejs_docs": "NODEJS_DOCS_TOOL",
    "vercel_docs": "VERCEL_DOCS_TOOL",
    "python_docs": "PYTHON_DOCS_TOOL",
    "pydantic_ai_docs": "PYDANTIC_AI_DOCS_TOOL",
}
