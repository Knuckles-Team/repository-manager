"""MCP tools for misc operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


def register_misc_tools(mcp: FastMCP):
    """Register miscellaneous tools like health check."""

    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})
