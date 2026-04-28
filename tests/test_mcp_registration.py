import os
import pytest
from fastmcp import FastMCP
from repository_manager.mcp_server import get_mcp_instance


def test_mcp_tools_registration():
    """
    Verify that all tools in repository_manager_mcp can be registered without SchemaErrors.
    This catches issues where non-default arguments follow default arguments.
    """
    os.environ["REPOSITORY_MANAGER_WORKSPACE"] = "/tmp"

    try:
        # get_mcp_instance internally calls all register_*_tools functions
        get_mcp_instance()
    except Exception as e:
        pytest.fail(f"Failed to register MCP tools: {e}")
