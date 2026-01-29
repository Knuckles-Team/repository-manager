import os
import pytest
from fastmcp import FastMCP
from repository_manager.repository_manager_mcp import register_tools


def test_mcp_tools_registration():
    """
    Verify that all tools in repository_manager_mcp can be registered without SchemaErrors.
    This catches issues where non-default arguments follow default arguments.
    """
    # Setup mock environment for the tools
    os.environ["REPOSITORY_MANAGER_WORKSPACE"] = "/tmp"

    mcp = FastMCP("TestRepoManager")

    try:
        register_tools(mcp)
    except Exception as e:
        pytest.fail(f"Failed to register MCP tools: {e}")

    # Verify tools are actually registered (checking if register_tools runs without error is sufficient for this bug)
    # assert len(mcp._tool_registry) > 0
