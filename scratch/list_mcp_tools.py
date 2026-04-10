import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def list_tools():
    # Use the command from mcp_config.json
    server_params = StdioServerParameters(
        command="repository-manager-mcp",
        args=["--transport", "stdio"],
        env={
            **os.environ,
            "REPOSITORY_MANAGER_WORKSPACE": os.getcwd(),
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(json.dumps([t.name for t in tools.tools], indent=2))


if __name__ == "__main__":
    asyncio.run(list_tools())
